from uuid import uuid4
from langchain_community.document_loaders import TextLoader, PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm.client import OpenAILLM
from utils.tools import get_file_name, get_file_extension, convert_escaped_chars_to_original_chars


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
                 chunk_size: int,
                 overlap_size: int,
                 separators: list | None,
                 embedding_client: OpenAILLM | None
                 ):
        print(f"Entering TextVectorProcessor init...")
        self.document_id = document_id
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.separators = separators
        self.embedding_client = embedding_client
        self.file_extension = get_file_extension(file_path, with_dot=False, upper=True)
        print(f"document_id: {self.document_id}\nfile_path: {self.file_path}\nchunk_size: {self.chunk_size}\n")
        print(f"overlap_size: {self.overlap_size}\nseparators: {self.separators}\n")
        print(f"embedding_client: {self.embedding_client}\nfile_extension: {self.file_extension}\n")
        print("=" * 50 + "\n")

    def process(self):
        print(f"Entering TextVectorProcessor process...")
        # Load file by LangChain:
        if self.file_extension == "TXT":
            loader = TextLoader(self.file_path)
        elif self.file_extension == "CSV":
            loader = TextLoader(self.file_path)
        elif self.file_extension == "PDF":
            print("Entering PDF process...")
            loader = PyPDFium2Loader(self.file_path)
        else:  # "docx":
            loader = TextLoader(self.file_path)
        print("Start to load document...")
        document = loader.load()
        print(f"Loaded document: {document}\n")

        # Check separators: the input separators are list of escaped characters, convert them to original characters
        if self.separators is not None:
            self.separators = convert_escaped_chars_to_original_chars(self.separators)

        print("Start to split text...")
        # Split text by LangChain
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap_size,
            separators=self.separators
        )
        print("text_splitter is ready...")
        split_document = text_splitter.split_documents(
            document)  # structure: [Document(page_content="**chunk_1**", metadata={"source": "file_abs_path"}), ...]

        print(split_document)

        # Add metadata to each document, and reconstruct data to fit the format of Qdrant insertion
        vector_points = []
        for idx, doc in enumerate(split_document):
            vec_id = str(uuid4())  # Generate vector_id in Vector DB by uuid
            page_content = doc.page_content
            document_name = get_file_name(doc.metadata["source"])
            page = doc.metadata.get('page')
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
                "page": page,
                "page_content": page_content
            }
            vector_points.append({"vec_id": vec_id, "vector": vector, "payload": payload})

        return vector_points


if __name__ == "__main__":
    # Example usage
    # file_path = "/Users/yu/Projects/AIGC/aiknowledge/uploaded_file/uploaded_file_temp/洛杉矶湖人.pdf"
    file_path = "uploaded_file/kb/NBA - 维基百科，自由的百科全书.pdf"
    file_extension = "PDF"
    chunk_size = 200
    overlap_size = 50
    separators = ['\n\n', '\n', '。']

    from config import app_config
    embedding_client = OpenAILLM(api_key=app_config.get("openai")["api_key"])
    processor = TextVectorProcessor(
        document_id="test_doc_id",
        file_path=file_path,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        separators=separators,
        embedding_client=embedding_client
    )
    processed_document = processor.process()

    from pprint import pprint
    pprint(processed_document)
