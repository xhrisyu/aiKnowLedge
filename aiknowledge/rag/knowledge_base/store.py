"""
Store the documents in the database [MongoDB] & Create the Lucene index
Embed chunk data and knowledge_base in the database [Qdrant]

1. document metadata
2. chunk data
3. construct Lucene index from chunk data
"""

from datetime import datetime
import json
import os
import time
from pymongo import MongoClient
from tqdm import tqdm

from aiknowledge.config import app_config
from aiknowledge.db import KBQdrantClient
from aiknowledge.llm import OpenAILLM

mongo_config = app_config.get("mongo")
qdrant_config = app_config.get("qdrant")
openai_config = app_config.get("openai")

mongo_client = MongoClient(mongo_config['uri'])
vecdb_client = KBQdrantClient(
    url=qdrant_config["url"],
    collection_name=qdrant_config['collection_name']["general"],
    embedding_dim=qdrant_config["embedding_dim"]
)
llm_client = OpenAILLM(api_key=openai_config["api_key"], )


def store_metadatas(
        doc_metadata: dict,
        database_name: str,
        collection_name: str,
):
    """
    Store document metadata and chunk data in MongoDB
    """

    database = mongo_client[database_name]
    collection = database[collection_name]

    # Add create time
    doc_metadata["create_time"] = datetime.now()

    # Insert document metadata
    # Check if the document already exists based on the doc_name
    if not collection.find_one({"doc_name": doc_metadata["doc_name"]}):
        collection.insert_one(doc_metadata)
        return 1
    else:
        print(f"Document {doc_metadata['doc_name']} already exists in the database.")
        return 0


def store_chunks(
        chunk_data_list: list,
        database_name: str,
        collection_name: str,
):
    """
    Store chunk data in MongoDB

    :param chunk_data_list: list of chunk data
    :param database_name: name of the database
    :param collection_name: name of the collection
    """
    database = mongo_client[database_name]
    collection = database[collection_name]

    # Insert chunk data
    # Check if the chunk already exists based on the doc_name and chunk_id
    for chunk_data in chunk_data_list:
        if not collection.find_one(
                {
                    "doc_name": chunk_data["doc_name"],
                    "chunk_id": chunk_data["chunk_id"]
                }
        ):
            collection.insert_one(chunk_data)
        else:
            print(f"Chunk {chunk_data['doc_name']}<{chunk_data['chunk_id']}> already exists in the database.")


def cleanup_chunks(
        doc_name: str,
        database_name: str,
        collection_name: str,
):
    """
    Remove the chunks of the document from the database

    :param doc_name: name of the document
    :param database_name: name of the database
    :param collection_name: name of the collection
    """
    database = mongo_client[database_name]
    collection = database[collection_name]

    # Remove the chunks of the document
    collection.delete_many({"doc_name": doc_name})


def construct_lucene_index(
        mongo_database_name: str,
        mongo_collection_name: str,
        document_json_dir: str,
        index_dir: str = "indexes/chunk_data"
) -> None:
    """
    Construct Lucene index from chunk data

    :param mongo_database_name: name of the MongoDB database
    :param mongo_collection_name: name of the MongoDB collection
    :param document_json_dir: dir of json file of the document chunk
    :param index_dir: dir of index file output
    :return:
    """

    # Get all the chunks from the MongoDB
    database = mongo_client[mongo_database_name]
    collection = database[mongo_collection_name]

    # Save the chunks into json
    if not os.path.exists(document_json_dir):
        os.makedirs(document_json_dir, exist_ok=True)

    with open(os.path.join(document_json_dir, f"{mongo_collection_name}.jsonl"), "w", encoding="utf-8") as f:
        for chunk in collection.find():
            f.write(
                json.dumps(
                    {
                        "id": str(chunk["chunk_id"]),
                        "contents": chunk["content"]
                    },
                    ensure_ascii=False
                ) + "\n"
            )

    # Create lucene index
    os.system(f"python -m pyserini.index.lucene \
              --collection JsonCollection \
              --input {document_json_dir} \
              --generator DefaultLuceneDocumentGenerator \
              --language zh \
              --index {index_dir} \
              --threads 1 \
              --storePositions \
              --storeDocvectors \
              --storeRaw"
              )


def store_chunks_vectors(
        mongo_database_name: str,
        mongo_collection_name: str,
        qdrant_collection_name: str
):
    """
    Store chunk data in the database [Qdrant]

    :param mongo_database_name:
    :param mongo_collection_name:
    :param qdrant_collection_name: name of the Qdrant collection
    :return: number of inserted vectors

    > chunk_data_list structure:
    [
        {
            "doc_id": doc_id(uuid)
            "doc_name": doc_name,
            "chunk_id": chunk_id(uuid, linked to Qdrant points' id),
            "chunk_seq": chunk_seq(chunk sequence in same doc),
            "content": content,
            "in_vector_db": False
        },
        ...
    ]
    """

    # Get all the chunk from Mongo
    mongo_collection = mongo_client[mongo_database_name][mongo_collection_name]
    chunk_data_list = list(mongo_collection.find())

    # Checkout to the target Qdrant collection
    vecdb_client.checkout_collection(qdrant_collection_name)

    max_tries = 5
    retry_delay = 0.5  # Retry delay in seconds
    total_inserted_num = 0

    for chunk_data in tqdm(chunk_data_list):
        print(f"Processing chunk {chunk_data['chunk_id']} in [{chunk_data['doc_name']}]...")

        # Check whether the chunk has been embedded and inserted
        if chunk_data.get("in_vector_db"):  # if inserted, skip
            print(f"> Chunk {chunk_data['chunk_id']} has been inserted in [{chunk_data['doc_name']}]. Skip.")
            continue

        # Get content of the chunk
        content = chunk_data["content"]

        # Get content embedding, retry until success
        for attempt in range(max_tries):
            print(f"> {attempt + 1} attempt to embed chunk...")
            try:
                chunk_vector = llm_client.get_text_embedding(content).content
                break
            except Exception as e:
                print(f"Failed to embed chunk <{chunk_data['chunk_id']}> in [{chunk_data['doc_name']}]. Exception: {e}")
                time.sleep(retry_delay)
        else:  # if meets `break` condition, skip the following code
            print(f"Failed to embed chunk <{chunk_data['chunk_id']}> in [{chunk_data['doc_name']}] after {max_tries} attempts.")
            continue
        print(f"> Chunk {chunk_data['chunk_id']} has been embedded successfully!")

        # Add to vectors_data, retry until success
        for attempt in range(max_tries):
            print(f">> {attempt + 1} attempt to insert chunk...")
            try:
                qdrant_inserted_num = vecdb_client.insert_vector(
                    vec_id=chunk_data["chunk_id"],
                    vector=chunk_vector
                )
                if qdrant_inserted_num == 1:
                    print(f">> Chunk {chunk_data['chunk_id']} has been inserted successfully!")
                    break
            except Exception as e:
                print(f"Failed to insert chunk {chunk_data['chunk_id']} in file [{chunk_data['doc_name']}]. Exception: {e}")
                time.sleep(retry_delay)

        # Update the number of inserted vectors
        total_inserted_num += 1

        # Update the chunk `in_vector_db` status
        mongo_collection.update_one(
            {"chunk_id": chunk_data["chunk_id"]},
            {"$set": {"in_vector_db": True}}
        )
        print(f">>> Chunk <{chunk_data['chunk_id']}> status has been updated in MongoDB.")

    print(f" ------ Total inserted vectors: {total_inserted_num} ------")

    return total_inserted_num
