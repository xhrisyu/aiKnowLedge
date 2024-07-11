"""
Store the documents in the database [MongoDB] & Create the Lucene index

1. document metadata
2. chunk data
3. construct Lucene index from chunk data
"""

from pymongo import MongoClient
from datetime import datetime
import json
import os

from aiknowledge.config import app_config

mongo_config = app_config.get("mongo")
mongo_client = MongoClient(mongo_config['uri'])


def store_document(
        doc_metadata: dict,
        database_name: str,
        collection_name: str,
):
    """
    Store document metadata and chunk data in MongoDB

    :param doc_metadata: metadata of the document
    :param database_name: name of the database
    :param collection_name: name of the collection
    """
    database = mongo_client[database_name]
    collection = database[collection_name]

    # Add create time
    doc_metadata["create_time"] = datetime.now()

    # Insert document metadata
    # Check if the document already exists based on the doc_name
    if not collection.find_one({"doc_name": doc_metadata["doc_name"]}):
        collection.insert_one(doc_metadata)
    else:
        print(f"Document {doc_metadata['doc_name']} already exists in the database.")


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


def construct_lucene_index(
        database_name: str,
        collection_name: str,
        document_json_dir: str,
        index_dir: str = "indexes/lucene-index"
) -> None:
    """
    Construct Lucene index from chunk data

    :param database_name: name of the MongoDB database
    :param collection_name: name of the MongoDB collection
    :param document_json_dir: dir of json file of the document chunk
    :param index_dir: dir of index file output
    :return:
    """

    # Get all the chunks from the MongoDB
    database = mongo_client[database_name]
    collection = database[collection_name]

    # Save the chunks into json
    if not os.path.exists(document_json_dir):
        os.makedirs(document_json_dir, exist_ok=True)

    with open(os.path.join(document_json_dir, "documents.jsonl"), "w", encoding="utf-8") as f:
        for doc in collection.find():
            f.write(
                json.dumps(
                    {
                        "id": f'{str(doc["doc_id"])}<{str(doc["chunk_id"])}>',
                        "contents": doc["content"]
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


def store_qa_document():
    pass

