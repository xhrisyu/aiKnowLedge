"""
构建则成外审文件的知识库存储
1. 文档载入 & 格式转换预处理
2. 文档分块 & MongoDB存储文档的元数据和分块 & 构建BM25的索引
3. Qdrant存储分块的向量
"""

import os
from pprint import pprint

from aiknowledge.utils.file_converter import docx2markdown
from aiknowledge.utils.tools import get_file_name
from aiknowledge.rag.store.loader import load_and_split
from aiknowledge.rag.store.doc_store import store_document, store_chunks, construct_lucene_index
from aiknowledge.rag.store.vector_store import store_chunks_vectors


audit_file_folder = "../../document/audit_file"
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


def split_chunk_and_store(chunk_size: int = 250, overlap_size: int = 60, separators=None):

    # 2. 文档分块

    if separators is None:
        separators = ["\n\n", "\n"]

    md_file_list = []
    for root, dirs, files in os.walk(store_folder):
        for file in files:
            if file.endswith(".md"):
                md_file_list.append(os.path.abspath(os.path.join(root, file)))

    md_file_list.sort()
    # pprint(md_file_list)

    for md_file in md_file_list:
        file_name = get_file_name(md_file, with_extension=False)
        print(f"Splitting {md_file} into chunks...")

        # Get doc metadata and chunk data
        doc_metadata, chunk_data_list = load_and_split(md_file, chunk_size, overlap_size, separators)
        print(f">> {file_name} has been split into chunks.")

        # Store doc metadata to MongoDB
        store_document(doc_metadata, "intflex_audit", "doc_metadata")
        print(f">> {file_name} metadata has been stored in MongoDB.")

        # Store chunk data to MongoDB
        store_chunks(chunk_data_list, "intflex_audit", "chunk_data")
        print(f">> {file_name} chunks have been stored in MongoDB.")

    # Create Lucene index
    document_json_dir = "../uploaded_file/document_json"
    index_dir = "../uploaded_file/indexes/lucene-index"

    construct_lucene_index(
        database_name="intflex_audit",
        collection_name="chunk_data",
        document_json_dir=os.path.abspath(document_json_dir),
        index_dir=os.path.abspath(index_dir)
    )
    print(f"Lucene index has been constructed.")


def store_vectors_to_qdrant():
    store_chunks_vectors(
        mongo_database_name="intflex_audit",
        mongo_collection_name="chunk_data",
        qdrant_collection_name="intflex_audit"
    )


# load_raw_document_and_preprocess()
# split_chunk_and_store(250, 60, ["\n\n", "\n"])
store_vectors_to_qdrant()
