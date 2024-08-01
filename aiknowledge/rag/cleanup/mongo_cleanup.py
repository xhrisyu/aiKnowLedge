import time
from pymongo import MongoClient

from aiknowledge.config import app_config


# Get Mongo client
mongo_config = app_config.get("mongo")
mongo_client = MongoClient(mongo_config['uri'])


# Reset chunk attribute `in_vecdb_db` to False
def reset_chunk_in_vecdb_db(
        mongo_database_name: str,
        mongo_collection_name: str
):
    """
    Reset chunk attribute `in_vector_db` to False

    :param mongo_database_name:
    :param mongo_collection_name:
    :return: number of updated documents
    """
    mongo_collection = mongo_client[mongo_database_name][mongo_collection_name]
    result = mongo_collection.update_many(
        {"in_vector_db": True},
        {"$set": {"in_vector_db": False}}
    )
    return result.modified_count
