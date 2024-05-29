from uuid import uuid4
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm import OpenAILLM
from .tools import get_file_name, get_file_extension, convert_escaped_chars_to_original_chars


class TextVectorProcessor:
    """
    Class for processing text vectors
    1. Convert text to vector
    2. Add metadata
    3. Format the vector and metadata for vector database insertion
    """

    def __init__(self,
                 document_id: str,
                 file_path: str,
                 chunk_size: int, overlap_size: int, separators: list | None,
                 embedding_client: OpenAILLM | None
                 ):
        self.document_id = document_id
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.separators = separators
        self.embedding_client = embedding_client
        self.file_extension = get_file_extension(file_path, with_dot=False, upper=True)

    def process(self):
        # Load file by LangChain:
        if self.file_extension == "txt":
            loader = TextLoader(self.file_path)
        elif self.file_extension == "csv":
            loader = TextLoader(self.file_path)
        else:  # "docx":
            loader = TextLoader(self.file_path)
        document = loader.load()

        # Check separators: the input separators are list of escaped characters, convert them to original characters
        if self.separators is not None:
            self.separators = convert_escaped_chars_to_original_chars(self.separators)

        # Split text by LangChain
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap_size,
            separators=self.separators
        )
        split_document = text_splitter.split_documents(document)  # structure: [Document(page_content="**chunk_1**", metadata={"source": "file_abs_path"}), ...]

        # Add metadata to each document, and reconstruct data to fit the format of Qdrant insertion
        vector_points = []
        for idx, doc in enumerate(split_document):
            vec_id = str(uuid4())  # Generate vector_id in Vector DB by uuid
            page_content = doc.page_content
            document_name = get_file_name(doc.metadata["source"])
            try:
                vector = self.embedding_client.get_text_embedding(page_content)
            except Exception as e:
                print(f"Failed to get embedding for chunk {idx + 1} in file [{document_name}]. Exception: {e}")
                continue
            payload = {
                "document_id": self.document_id,
                "document_name": document_name,
                "source": doc.metadata["source"],
                "chunk_id": idx + 1,
                "page_content": page_content
            }
            vector_points.append({"vec_id": vec_id, "vector": vector, "payload": payload})

        return vector_points


# if __name__ == "__main__":
#     # Example usage
#     file_path = "/Users/yu/Projects/AIGC/aiknowledge/doc_temp/ZC-1-M-001 厂区温湿度管控规范（B）.txt"
#     file_extension = "txt"
#     chunk_size = 100
#     overlap_size = 20
#     separators = ['\n\n', '\n', '。']
#     # embedding_client = OpenAILLM()
#     processor = TextVectorProcessor(
#         document_id="test_doc_id",
#         file_path=file_path,
#         file_extension=file_extension,
#         chunk_size=chunk_size,
#         overlap_size=overlap_size,
#         separators=separators)
#     processed_document = processor.process()
#     from pprint import pprint
#     pprint(processed_document)
