LOADER_DICT = {
    "UnstructuredFileLoader": ['.txt', '.xml'],
    "CSVLoader": [".csv"],
    # "FilteredCSVLoader": [".csv"], 如果使用自定义分割csv
    "RapidOCRPDFLoader": [".pdf"],
    "UnstructuredWordDocumentLoader": ['.docx', '.uploaded_file'],
}
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]

# MONGO_DATABASE_NAME = "aiknowledge"
# MONGO_COLLECTION_DEFAULT_NAME = "rag"
# QDRANT_COLLECTION_DEFAULT_NAME = "general"

MONGO_DATABASE_INTFLEX_AUDIT = "intflex_audit"
MONGO_COLLECTION_INTFLEX_AUDIT_CHUNK_DATA = "chunk_data"
MONGO_COLLECTION_INTFLEX_AUDIT_QA = "qa"

QDRANT_COLLECTION_INTFLEX_AUDIT_CHUNK_DATA = "intflex_audit"
QDRANT_COLLECTION_INTFLEX_AUDIT_QA = "intflex_audit_qa"

LUCENE_INDEX_DIR_INTFLEX_AUDIT_CHUNK_DATA = "./aiknowledge/uploaded_file/indexes/chunk_data"
LUCENE_INDEX_DIR_INTFLEX_AUDIT_QA = "./aiknowledge/uploaded_file/indexes/qa"
