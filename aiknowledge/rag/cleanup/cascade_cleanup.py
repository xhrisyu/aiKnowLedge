import time
from pymongo import MongoClient

from aiknowledge.config import app_config
from aiknowledge.db import QAQdrantClient


# Get Mongo client
mongo_config = app_config.get("mongo")
mongo_client = MongoClient(mongo_config['uri'])


# Get Qdrant client
qdrant_config = app_config.get("qdrant")
vecdb_client = QAQdrantClient(
    url=qdrant_config["url"],
    collection_name=qdrant_config['collection_name']["intflex_audit"],
    embedding_dim=qdrant_config["embedding_dim"]
)


def delete_certain_document(doc_name: str):
    """
    Delete certain document in the MongoDB

    :param doc_name: name of the document
    """
    # Get doc_id from Mongo
    doc_id = mongo_client["intflex_audit"]["doc_metadata"].find_one({"doc_name": doc_name})["doc_id"]

    # Get chunk_id that belongs to the doc_id
    chunk_id_list = [chunk["chunk_id"] for chunk in mongo_client["intflex_audit"]["chunk_data"].find({"doc_id": doc_id})]

    # Delete vector by points' id (linked to chunk_id) from Qdrant
    vecdb_client.remove_vectors_by_id(chunk_id_list)

    # Delete metadata and chunk data from MongoDB
    mongo_client["intflex_audit"]["doc_metadata"].delete_one({"doc_name": doc_name})
    mongo_client["intflex_audit"]["chunk_data"].delete_many({"doc_name": doc_name})

