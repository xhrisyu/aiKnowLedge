from typing import Any, Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from aiknowledge.db import KBQdrantClient, KBMongoClient
from aiknowledge.llm import OpenAILLM
from aiknowledge.rag.retriever.bm25 import BM25Searcher
from aiknowledge.rag.retriever.hybrid_search import ReRanking
from aiknowledge.utils.tools import remove_overlap
from aiknowledge.config import app_config


def retrieve(
        retrieve_method: str,
        query_input: str | List[float],
        retrieve_client: KBQdrantClient | BM25Searcher,
        retrieve_client_scope_name: str,
        mongo_client: KBMongoClient,
        mongo_database_name: str,
        mongo_collection_name: str,
        top_k: int
) -> List[Dict]:
    """
    Format the retrieve payloads by adding the document name, chunk sequence, previous content, content, next content

    Using ThreadPoolExecutor to process the retrieve payloads in parallel.

    :param retrieve_method: The method used to retrieve the similar vectors, enum in ["cosine_similarity", "bm25"]
    :param query_input: The query input, either a string or an embedded vector list
    :param retrieve_client: The retrieve client, KBQdrantClient or a BM25Searcher
    :param retrieve_client_scope_name: The retrieve client scope name. [KBQdrantClient: collection name | BM25Searcher: index dir]
    :param mongo_client: The mongo client
    :param mongo_database_name: The mongo database name
    :param mongo_collection_name: The mongo collection name
    :param top_k: The number of top k retrieve payloads to return
    :return: The formatted retrieve payloads

    > input retrieve payloads structure:
    [
        {
            "score": <float number>,
            "chunk_id": <uuid>,
        },
        ...
    ]

    > return formatted retrieve payloads structure:
    [
        {
            "score": 0.1,
            "chunk_id": "",
            "doc_name": "",
            "chunk_seq": 1,
            "pre_content": "",
            "content": "",
            "next_content": "",
        },
    ]
    """

    top_k = 5

    if retrieve_method not in ["cosine_similarity", "bm25"]:
        raise ValueError("Invalid retrieve method")

    if retrieve_method == "cosine_similarity":
        retrieve_client.checkout_collection(retrieve_client_scope_name)
        retrieve_payloads = retrieve_client.retrieve_similar_vectors_simply(
            query_vector=query_input,
            top_k=top_k,
        )
    else:  # retrieve_method == "bm25":
        retrieve_client.checkout_index(retrieve_client_scope_name)
        retrieve_payloads = retrieve_client.search(
            query=query_input,
            top_k=top_k
        )

    def get_payload_task(retrieve_payload: dict[str, Any]) -> dict[str, Any]:
        doc_name, chunk_seq, pre_content, content, next_content = mongo_client.get_neighbour_chunk_content_by_chunk_id(
            chunk_id=retrieve_payload["chunk_id"],
            database_name=mongo_database_name,
            collection_name=mongo_collection_name
        )
        retrieve_payload["doc_name"] = doc_name
        retrieve_payload["chunk_seq"] = chunk_seq
        retrieve_payload["pre_content"] = remove_overlap(pre_content, content)
        retrieve_payload["content"] = content
        retrieve_payload["next_content"] = remove_overlap(next_content, content)
        retrieve_payload["ranking_method"] = retrieve_method

        return retrieve_payload

    # Create thread pool executor to process the retrieve payloads in parallel
    with ThreadPoolExecutor(max_workers=top_k) as executor:
        futures = [executor.submit(get_payload_task, retrieve_payload) for retrieve_payload in retrieve_payloads]
        processed_retrieve_payloads = [future.result() for future in as_completed(futures)]

    processed_retrieve_payloads = sorted(processed_retrieve_payloads, key=lambda x: x["score"], reverse=True)[:top_k]

    return processed_retrieve_payloads


def retrieve_pipeline(
        user_query: str,
        entity_list: List[str],
        llm_client: OpenAILLM,
        qdrant_client: KBQdrantClient,
        keyword_retriever: BM25Searcher,
        mongo_client: KBMongoClient,
        vector_search_params: Dict[str, Any],
        keyword_search_params: Dict[str, Any],
        top_k: int,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
    """

    :param user_query:
    :param entity_list:
    :param llm_client:
    :param qdrant_client:
    :param keyword_retriever:
    :param mongo_client:
    :param vector_search_params:
    :param keyword_search_params:
    :param top_k:

    :return:

    `vector_search_params` structure:
    {
        "chunk_data": {
            "qdrant_collection_name": "intflex_audit",
            "mongo_database_name": "intflex_audit",
            "mongo_collection_name": "chunk_data"
        },
        "qa": {
            "qdrant_collection_name": "intflex_audit_qa",
            "mongo_database_name": "intflex_audit",
            "mongo_collection_name": "qa"
        }
    }

    `keyword_search_params` structure:
    {
        "chunk_data": {
            "lucene_index_dir": "./aiknowledge/uploaded_file/indexes/chunk_data_index",
            "mongo_database_name": "intflex_audit",
            "mongo_collection_name": "chunk_data"
        },
        "qa": {
            "lucene_index_dir": "./aiknowledge/uploaded_file/indexes/qa_index",
            "mongo_database_name": "intflex_audit",
            "mongo_collection_name": "qa"
        }
    }
    """

    ###############################################
    # Embedding the user query
    ###############################################
    embedding_response = llm_client.get_text_embedding(text=user_query)
    embedding_user_question = embedding_response.content

    ###############################################
    # Vector Search & Keyword Search
    ###############################################
    def task_vector_search_1():
        # Vector search in `chunk data`
        return retrieve(
            retrieve_method="cosine_similarity",
            query_input=embedding_user_question,
            retrieve_client=qdrant_client,
            retrieve_client_scope_name=vector_search_params["chunk_data"]["qdrant_collection_name"],
            mongo_client=mongo_client,
            mongo_database_name=vector_search_params["chunk_data"]["mongo_database_name"],
            mongo_collection_name=vector_search_params["chunk_data"]["mongo_collection_name"],
            top_k=top_k
        )

    def task_vector_search_2():
        # Vector search in `qa`
        return retrieve(
            retrieve_method="cosine_similarity",
            query_input=embedding_user_question,
            retrieve_client=qdrant_client,
            retrieve_client_scope_name=vector_search_params["qa"]["qdrant_collection_name"],
            mongo_client=mongo_client,
            mongo_database_name=vector_search_params["qa"]["mongo_database_name"],
            mongo_collection_name=vector_search_params["qa"]["mongo_collection_name"],
            top_k=top_k
        )

    def task_keyword_search_1():
        # Keyword search in `chunk data`
        return retrieve(
            retrieve_method="bm25",
            query_input=" ".join(entity_list),
            retrieve_client=keyword_retriever,
            retrieve_client_scope_name=keyword_search_params["chunk_data"]["lucene_index_dir"],
            mongo_client=mongo_client,
            mongo_database_name=keyword_search_params["chunk_data"]["mongo_database_name"],
            mongo_collection_name=keyword_search_params["chunk_data"]["mongo_collection_name"],
            top_k=top_k
        )

    def task_keyword_search_2():
        # Keyword search in `qa`
        return retrieve(
            retrieve_method="bm25",
            query_input=" ".join(entity_list),
            retrieve_client=keyword_retriever,
            retrieve_client_scope_name=keyword_search_params["qa"]["lucene_index_dir"],
            mongo_client=mongo_client,
            mongo_database_name=keyword_search_params["qa"]["mongo_database_name"],
            mongo_collection_name=keyword_search_params["qa"]["mongo_collection_name"],
            top_k=top_k
        )

    # Create thread pool executor to process the retrieve payloads in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_vector_search_1 = executor.submit(task_vector_search_1)
        future_vector_search_2 = executor.submit(task_vector_search_2)
        future_keyword_search_1 = executor.submit(task_keyword_search_1)
        future_keyword_search_2 = executor.submit(task_keyword_search_2)

        vector_retrieve_payloads = future_vector_search_1.result()
        qa_vector_retrieve_payloads = future_vector_search_2.result()
        keyword_retrieve_payloads = future_keyword_search_1.result()
        qa_keyword_retrieve_payloads = future_keyword_search_2.result()

    ###############################################
    # Hybrid Search
    ###############################################
    cohere_config = app_config.get("cohere")

    def task_reranking_1():
        # Reranking chunk data
        chunks_candidates = vector_retrieve_payloads + keyword_retrieve_payloads

        # Init hybrid retriever
        hybrid_retriever = ReRanking(api_key=cohere_config["api_key"])
        hybrid_retriever.setup_reranking(rankings=chunks_candidates, k=top_k)
        reranking_items = hybrid_retriever.get_cross_encoder_scores(query=user_query)

        # Get original content from `chunks_candidates`
        for reranking_item in reranking_items:
            chunk_id = reranking_item["chunk_id"]
            for doc in chunks_candidates:
                if doc["chunk_id"] == chunk_id:
                    reranking_item["doc_name"] = doc["doc_name"]
                    reranking_item["chunk_seq"] = doc["chunk_seq"]
                    reranking_item["pre_content"] = doc.get("pre_content", "")
                    reranking_item["content"] = doc.get("content", "")
                    reranking_item["next_content"] = doc.get("next_content", "")

        return reranking_items

    def task_reranking_2():
        # Reranking qa
        qas_candidates = qa_vector_retrieve_payloads + qa_keyword_retrieve_payloads

        # Init hybrid retriever
        hybrid_retriever = ReRanking(api_key=cohere_config["api_key"])
        hybrid_retriever.setup_reranking(rankings=qas_candidates, k=top_k)
        reranking_items = hybrid_retriever.get_cross_encoder_scores(query=user_query)

        # Get original content from `qas_candidates`
        for reranking_item in reranking_items:
            chunk_id = reranking_item["chunk_id"]
            for doc in qas_candidates:
                if doc["chunk_id"] == chunk_id:
                    reranking_item["doc_name"] = doc["doc_name"]
                    reranking_item["chunk_seq"] = doc["chunk_seq"]
                    reranking_item["pre_content"] = doc.get("pre_content", "")
                    reranking_item["content"] = doc.get("content", "")
                    reranking_item["next_content"] = doc.get("next_content", "")

        return reranking_items

    # Create thread pool executor to process the reranking in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_reranking_1 = executor.submit(task_reranking_1)
        future_reranking_2 = executor.submit(task_reranking_2)

        reranking_item_list = future_reranking_1.result()
        reranking_item_list_qa = future_reranking_2.result()

    return vector_retrieve_payloads, qa_vector_retrieve_payloads, keyword_retrieve_payloads, qa_keyword_retrieve_payloads, reranking_item_list, reranking_item_list_qa
