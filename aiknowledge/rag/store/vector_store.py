"""
Embed chunk data and store in the database [Qdrant]
"""
import time
from pymongo import MongoClient

from aiknowledge.config import app_config
from aiknowledge.db import KBQdrantClient
from aiknowledge.llm import OpenAILLM

# Get Qdrant client
qdrant_config = app_config.get("qdrant")
vecdb_client = KBQdrantClient(
    url=qdrant_config["url"],
    collection_name=qdrant_config['collection_name']["general"],
    embedding_dim=qdrant_config["embedding_dim"]
)

# Get LLM client
openai_config = app_config.get("openai")
llm_client = OpenAILLM(api_key=openai_config["api_key"], )

# Get Mongo client
mongo_config = app_config.get("mongo")
mongo_client = MongoClient(mongo_config['uri'])


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

    for chunk_data in chunk_data_list:
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
