
from aiknowledge.config import app_config
from aiknowledge.db import QAQdrantClient

# Get Qdrant client
qdrant_config = app_config.get("qdrant")
vecdb_client = QAQdrantClient(
    url=qdrant_config["url"],
    collection_name=qdrant_config['collection_name']["general"],
    embedding_dim=qdrant_config["embedding_dim"]
)


# Drop the collection
def drop_collection(collection_name: str):
    """
    Drop the collection in the Qdrant database

    :param collection_name: name of the collection
    """
    vecdb_client.checkout_collection(collection_name)
    vecdb_client.drop_collection()


if __name__ == '__main__':
    drop_collection("intflex_audit")

