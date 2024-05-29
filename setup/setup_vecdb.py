from dotenv import load_dotenv
import os
import yaml
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PointStruct
from pprint import pprint


# ============== OpenAI Config ==============
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
# openai.proxy = "http://127.0.0.1:7890"
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
openai_config = yaml.safe_load(open("../config/config.yaml", "r"))['openai']
EMBEDDING_MODEL = openai_config["embedding_model"]

# ============== qdrant Config ==============
qdrant_config = yaml.safe_load(open("../config/config.yaml", "r"))['qdrant']
QDRANT_URL = qdrant_config["url"]
EMBEDDING_DIM = qdrant_config["embedding_dim"]
STORAGE_PATH = qdrant_config['storage_path']
GENERAL_COLLECTION = qdrant_config['collection_name']['general']
PROJECT_COLLECTION = qdrant_config['collection_name']['project']
VOCABULARY_COLLECTION = qdrant_config['collection_name']['vocabulary']

"""
general文件
chunk_size=400,
chunk_overlap=100,
separators=["\n\n"]

线路板专业术语（流程英文版）.txt：
chunk_size=5,
chunk_overlap=0,
separators=["\n"]
"""


def langchain_setup_vecdb(
        file_path: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: list[str] | None,
        qdrant_collection_name: str
) -> None:
    """
    Using Langchain framework to split document from folder or , embedding text and store in Vector DB
    """
    if os.path.isdir(file_path):
        loader = DirectoryLoader(file_path, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    elif os.path.isfile(file_path):
        loader = TextLoader(file_path)
    else:
        raise ValueError("`file_path` must be a directory or a file")

    raw_doc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    doc = text_splitter.split_documents(raw_doc)
    if doc is None:
        return

    # Add metadata to each document
    new_doc = [doc[0]]
    new_doc[0].metadata['id'] = 1
    cur_id = 1
    for i in range(1, len(doc)):
        pre_doc, cur_doc = doc[i - 1], doc[i]
        if cur_doc.metadata['source'] == pre_doc.metadata['source']:
            cur_id += 1
        else:
            cur_id = 1
        cur_doc.metadata['id'] = cur_id
        new_doc.append(cur_doc)

    # Set source name without suffix
    for _doc in new_doc:
        _doc.metadata['source'] = get_file_name_without_suffix(_doc.metadata['source'])

    pprint(new_doc)
    # new_doc = [new_doc[0]]

    # Store in Vector DB
    Qdrant.from_documents(
        documents=new_doc,
        embedding=OpenAIEmbeddings(),
        url=QDRANT_URL,
        collection_name=qdrant_collection_name,
        force_recreate=True,
    )


if __name__ == "__main__":
    DOC_CHUNK_SIZE = 300
    DOC_CHUNK_OVERLAP = 50

    langchain_setup_vecdb(
        file_path="../doc/kb",
        chunk_size=DOC_CHUNK_SIZE,
        chunk_overlap=DOC_CHUNK_OVERLAP,
        separators=['\n\n', '\n'],
        qdrant_collection_name=GENERAL_COLLECTION
    )

    # langchain_setup_vecdb(
    #     file_path="doc/kb_1_general",
    #     chunk_size=300, chunk_overlap=100, separators=["\n\n"],
    #     qdrant_collection_name=GENERAL_COLLECTION
    # )
    # langchain_setup_vecdb(
    #     file_path="doc/kb_3_vocabulary",
    #     chunk_size=5, chunk_overlap=0, separators=["\n"],
    #     qdrant_collection_name=VOCABULARY_COLLECTION
    # )
    # langchain_setup_vecdb(
    #     file_path="doc/kb_2_project",
    #     chunk_size=500, chunk_overlap=100, separators=['\n\n', '\n', '。'],
    #     qdrant_collection_name=PROJECT_COLLECTION
    # )

    # user_question = "在公司打架有什么后果"
    # user_question = "在厂区打架的处罚"
    # user_question = "上班期间炒股会如何处罚"
    # qdrant_client = VecDBClient(
    #     url=QDRANT_URL,
    #     collection_name=COLLECTION,
    #     embedding_dim=EMBEDDING_DIM
    # )
    # embedded_query = get_text_embedding(text=user_question, embedding_model_name=EMBEDDING_MODEL_NAME)
    # searched_points = qdrant_client.retrieve_similar_vectors(query_vec=embedded_query, top_k=5, sim_lower_bound=0.65)
    # searched_res = {}
    # for point in searched_points:
    #     source_name = point['metadata']['source']
    #     match = re.search(r'/([^/]+)\s*\.', source_name)
    #     if match:
    #         source_name = match.group(1)
    #     source_content = point['page_content'].strip()
    #     if source_name not in searched_res:
    #         searched_res[source_name] = [source_content]
    #     else:
    #         searched_res[source_name].append(source_content)
    # dominant_source = max(searched_res, key=lambda x: len(searched_res[x]))
    # dominant_text = "\n\n".join(searched_res[dominant_source][:3])
    # context_text = f"文档来源: {dominant_source}\n文档片段: \n{dominant_text}"
    # print(context_text)
    # openai_response = get_openai_qa_response(
    #     context_text=context_text,
    #     user_question=user_question,
    #     llm_model_name=CHAT_MODEL_NAME,
    #     is_stream=False,
    #     chat_history=None,
    # )
