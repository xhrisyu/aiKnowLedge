import time
from pymongo import MongoClient

from aiknowledge.config import app_config
from aiknowledge.db import KBQdrantClient


# Get Mongo client
mongo_config = app_config.get("mongo")
mongo_client = MongoClient(mongo_config['uri'])


# Get Qdrant client
qdrant_config = app_config.get("qdrant")
vecdb_client = KBQdrantClient(
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
    doc_metadata_result = mongo_client["intflex_audit"]["doc_metadata"].find_one({"doc_name": doc_name})
    if doc_metadata_result:
        doc_id = doc_metadata_result["doc_id"]
    else:
        return

    # Get chunk_id that belongs to the doc_id
    chunk_id_list = [chunk["chunk_id"] for chunk in mongo_client["intflex_audit"]["chunk_data"].find({"doc_id": doc_id})]

    if not chunk_id_list:
        return

    # Delete vector by points' id (linked to chunk_id) from Qdrant
    vecdb_client.remove_vectors_by_id(chunk_id_list)

    # Delete metadata and chunk data from MongoDB
    mongo_client["intflex_audit"]["doc_metadata"].delete_one({"doc_name": doc_name})
    mongo_client["intflex_audit"]["chunk_data"].delete_many({"doc_name": doc_name})


if __name__ == "__main__":
    """
    ZC-1-M-284 PCB天准LDI参数设定表（A1）
    ZC-2-M-088测试不良标识及打孔操作规范（B1）
    ZC-2-M-153 PCB产品工序运输标准作业指导书（A1）
    ZC-2-M-164 PCB AOI扫描标准作业指导书（A）
    ZC-M-004 信息安全管理手册 (A)
    ZC-QP-079信息安全风险识别与评价管理程序（A）
    """
    doc_names = [
        "ZC-1-M-284 PCB天准LDI参数设定表（A1）",
        "ZC-2-M-088测试不良标识及打孔操作规范（B1）",
        "ZC-2-M-153 PCB产品工序运输标准作业指导书（A1）",
        "ZC-2-M-164 PCB AOI扫描标准作业指导书（A）",
        "ZC-M-004 信息安全管理手册 (A)",
        "ZC-QP-079信息安全风险识别与评价管理程序（A）",
        "QSA通用V4.0（光迅）",
        "QPA-PCB V2.0（光迅）"
    ]
    for doc_name in doc_names:
        delete_certain_document(doc_name)
        time.sleep(1)
