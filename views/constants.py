LOADER_DICT = {
    "UnstructuredFileLoader": ['.txt', '.xml'],
    "CSVLoader": [".csv"],
    # "FilteredCSVLoader": [".csv"], 如果使用自定义分割csv
    "RapidOCRPDFLoader": [".pdf"],
    "UnstructuredWordDocumentLoader": ['.docx', '.doc'],
}
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]

MONGO_DATABASE_NAME = "aiknowledge"
MONGO_COLLECTION_DEFAULT_NAME = "kb"
QDRANT_COLLECTION_DEFAULT_NAME = "general"

CHAT_HISTORY_LEN = 10
