"""
构建则成外审文件的知识库存储
1. 文档载入 & 格式转换预处理
2. 文档分块 & MongoDB存储文档的元数据和分块 & 构建BM25的索引
3. Qdrant存储分块的向量
"""

import os
from pprint import pprint

from aiknowledge.utils.file_converter import docx2markdown
from aiknowledge.utils.tools import get_file_name, get_file_extension
from aiknowledge.rag.store.loader import load_and_split
from aiknowledge.rag.store.doc_store import store_document, store_chunks, construct_lucene_index
from aiknowledge.rag.store.vector_store import store_chunks_vectors


audit_file_folder = "../../document/intflex_audit"
store_folder = "../uploaded_file/intflex_audit"


def load_raw_document_and_preprocess():

    # 1. 文档载入和预处理

    docx_file_list = []
    for root, dirs, files in os.walk(audit_file_folder):
        for file in files:
            if file.endswith(".docx"):
                docx_file_list.append(os.path.abspath(os.path.join(root, file)))

    # sort file name
    docx_file_list.sort()

    for docx_file in docx_file_list:
        file_name = get_file_name(docx_file, with_extension=False)
        # if not os.path.exists(os.path.join(store_folder, file_name)):
        #     print(f"Converting {docx_file} to markdown file...")
        #     docx2markdown(docx_file, output_dir=os.path.join(store_folder, file_name))
        # else:
        #     print(f"{file_name} already exists in the store folder.")

        print(f"Converting {docx_file} to markdown file...")
        docx2markdown(docx_file, output_dir=os.path.join(store_folder, file_name))


def split_chunk_and_store(given_file_name_list: list[str] = None):

    # 2. 文档分块

    file_list = []
    for root, dirs, files in os.walk(store_folder):
        for file in files:
            if file.endswith(".md") or file.endswith(".txt"):
                file_list.append(os.path.abspath(os.path.join(root, file)))

    file_list.sort()

    for file_path in file_list:
        file_name = get_file_name(file_path, with_extension=False)

        if given_file_name_list and file_name not in given_file_name_list:
            continue

        print(f"Processing {file_name}...")

        doc_type = get_file_extension(file_path, with_dot=False, upper_case=True)
        print(f"Splitting {file_path} into chunks...")

        # Get doc metadata and chunk data
        if doc_type == "MD":
            doc_metadata, chunk_data_list = load_and_split(file_path, 300, 50, ["\n\n"])
        else:
            doc_metadata, chunk_data_list = load_and_split(file_path, 300, 50, ["\n\n", "\n", "。"])
        print(f">> {file_name} has been split into chunks.")

        # Store doc metadata to MongoDB
        store_document(doc_metadata, "intflex_audit", "doc_metadata")
        print(f">> {file_name} metadata has been stored in MongoDB.")

        # Store chunk data to MongoDB
        if file_name in ["QSA通用V4.0（光迅）", "QPA-PCB V2.0（光迅）"]:
            store_chunks(chunk_data_list, "intflex_audit", "qa")
        else:
            store_chunks(chunk_data_list, "intflex_audit", "chunk_data")
        print(f">> {file_name} chunks have been stored in MongoDB.")


def create_lucene_index():

    construct_lucene_index(
        mongo_database_name="intflex_audit",
        mongo_collection_name="chunk_data",
        document_json_dir=os.path.abspath("../uploaded_file/document_json"),
        index_dir=os.path.abspath("../uploaded_file/indexes/chunk_data_index")
    )

    construct_lucene_index(
        mongo_database_name="intflex_audit",
        mongo_collection_name="qa",
        document_json_dir=os.path.abspath("../uploaded_file/document_json"),
        index_dir=os.path.abspath("../uploaded_file/indexes/qa_index")
    )

    print(f"Lucene index has been constructed.")


def store_vectors_to_qdrant():
    # store_chunks_vectors(
    #     mongo_database_name="intflex_audit", mongo_collection_name="chunk_data",
    #     qdrant_collection_name="intflex_audit"
    # )

    store_chunks_vectors(
        mongo_database_name="intflex_audit", mongo_collection_name="chunk_data",
        qdrant_collection_name="intflex_audit"
    )

    # store_chunks_vectors(
    #     mongo_database_name="intflex_audit", mongo_collection_name="qa",
    #     qdrant_collection_name="intflex_audit_qa"
    # )


if __name__ == "__main__":

    # load_raw_document_and_preprocess()

    # to_process_file_name_list = [
    #     "ZC-1-M-284 PCB天准LDI参数设定表（A1）",
    #     "ZC-2-M-088测试不良标识及打孔操作规范（B1）",
    #     "ZC-2-M-153 PCB产品工序运输标准作业指导书（A1）",
    #     "ZC-2-M-164 PCB AOI扫描标准作业指导书（A）",
    #     "ZC-M-004 信息安全管理手册 (A)",
    #     "ZC-QP-079信息安全风险识别与评价管理程序（A）",
    #     "QSA通用V4.0（光迅）",
    #     "QPA-PCB V2.0（光迅）"
    # ]
    # split_chunk_and_store(given_file_name_list=to_process_file_name_list)
    # split_chunk_and_store()

    # store_vectors_to_qdrant()

    create_lucene_index()