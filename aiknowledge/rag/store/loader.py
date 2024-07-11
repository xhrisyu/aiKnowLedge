from typing import Optional
from uuid import uuid4
from langchain_community.document_loaders import TextLoader, PyPDFium2Loader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from aiknowledge.utils.tools import get_file_name, get_file_extension, convert_escaped_chars_to_original_chars


def load_and_split(
        file_path: str,
        chunk_size: int,
        overlap_size: int,
        separators: Optional[list],
) -> tuple[dict, list]:

    # Get file name and extension
    doc_name = get_file_name(file_path, with_extension=False)
    doc_type = get_file_extension(file_path, with_dot=False, upper_case=True)

    # Generate the unique doc_id
    doc_id = str(uuid4())

    # Load file (by LangChain)
    if doc_type == "TXT":
        loader = TextLoader(file_path)
    elif doc_type == "CSV":
        loader = TextLoader(file_path)
    elif doc_type == "PDF":
        loader = PyPDFium2Loader(file_path)
    elif doc_type == "DOCX":
        loader = TextLoader(file_path)
    elif doc_type == "MD":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        loader = TextLoader(file_path)

    loaded_document = loader.load()

    # Check separators: the input separators are list of escaped characters, convert them to original characters
    if separators is not None:
        separators = convert_escaped_chars_to_original_chars(separators)

    # Split text by LangChain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        separators=separators
    )
    # <default structure>: [Document(page_content="**chunk_1**", metadata={"source": "file_abs_path"}), ...]
    split_document = text_splitter.split_documents(loaded_document)

    # Format document metadata
    """
    document = {
        "doc_id": uuid,
        "doc_name": <name of the original file>,
        "doc_type": <PDF, XLSX, DOCX, ...>,
        "chunk_size": int,
        "overlap_size": int,
        "separators": Optional[List[str]],
        "create_time": <the time document inserted into mongo>,
        "location": <local file path>,
    }
    """
    doc_metadata = {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "doc_type": doc_type,
        "chunk_size": chunk_size,
        "overlap_size": overlap_size,
        "separators": separators,
        "location": file_path,
    }

    # Add extra data to each chunk
    """
    chunk = {
        "doc_id": uuid,
        "doc_name": <name of the original file>,
        "chunk_id": <chunk seq in this doc, same doc_id file have multiple chunks>,
        "content": "这是当前chunk的原文片段",
        "in_vector_db": <True of False>, 
    }
    """
    chunk_data_list = []
    for idx, doc in enumerate(split_document):
        payload = {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "chunk_id": idx + 1,
            "content": doc.page_content,
            "in_vector_db": False
        }
        chunk_data_list.append(payload)

    return doc_metadata, chunk_data_list


# class DocumentLoader:
#     """
#     Class for processing text vectors
#     1. Convert text to vector
#     2. Add metadata
#     3. Format the vector and metadata for vector database insertion
#     """
#
#     def __init__(self,
#                  document_id: str,
#                  file_path: str,
#                  chunk_size: int,
#                  overlap_size: int,
#                  separators: list | None,
#                  embedding_client: OpenAILLM | None
#                  ):
#         self.document_id = document_id
#         self.file_path = file_path
#         self.chunk_size = chunk_size
#         self.overlap_size = overlap_size
#         self.separators = separators
#         self.embedding_client = embedding_client
#         self.file_extension = get_file_extension(file_path, with_dot=False, upper_case=True)
#
#     def process(self):
#         # Load file by LangChain:
#         if self.file_extension == "TXT":
#             loader = TextLoader(self.file_path)
#         elif self.file_extension == "CSV":
#             loader = TextLoader(self.file_path)
#         elif self.file_extension == "PDF":
#             loader = PyPDFium2Loader(self.file_path)
#         else:  # "docx":
#             loader = TextLoader(self.file_path)
#         document = loader.load()
#
#         # Check separators: the input separators are list of escaped characters, convert them to original characters
#         if self.separators is not None:
#             self.separators = convert_escaped_chars_to_original_chars(self.separators)
#
#         # Split text by LangChain
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.overlap_size,
#             separators=self.separators
#         )
#         print("Text Splitter is ready...")
#         split_document = text_splitter.split_documents(
#             document)  # structure: [Document(page_content="**chunk_1**", metadata={"source": "file_abs_path"}), ...]
#         print(split_document)
#
#         # Add metadata to each store, and reconstruct data to fit the format of Qdrant insertion
#         vector_points = []
#         for idx, doc in enumerate(split_document):
#             vec_id = str(uuid4())  # Generate vector_id in Vector DB by uuid
#             page_content = doc.page_content
#             document_name = get_file_name(doc.metadata["source"])
#             page = doc.metadata.get('page')
#             try:
#                 vector = self.embedding_client.get_text_embedding(page_content)
#             except Exception as e:
#                 print(f"Failed to get embedding for chunk {idx + 1} in file [{document_name}]. Exception: {e}")
#                 continue
#             payload = {
#                 "document_id": self.document_id,
#                 "document_name": document_name,
#                 "source": doc.metadata["source"],
#                 "chunk_id": idx + 1,
#                 "page": page,
#                 "page_content": page_content,
#                 "params": {
#                     "chunk_size": self.chunk_size,
#                     "overlap_size": self.overlap_size
#                 }
#             }
#             vector_points.append({"vec_id": vec_id, "vector": vector, "payload": payload})
#
#         return vector_points
