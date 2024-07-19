from typing import Any, Optional, List, Dict, Tuple

from aiknowledge.db import KBQdrantClient, KBMongoClient
from aiknowledge.llm import OpenAILLM
from aiknowledge.rag.retriever.bm25 import BM25Searcher
from aiknowledge.rag.retriever.hybrid_search import ReRanking
from aiknowledge.utils.tools import remove_overlap
from concurrent.futures import ThreadPoolExecutor, as_completed


def format_retrieve_payload(
        retrieve_method: str,
        retrieve_payloads: list[dict[str, Any]],
        mongo_client: KBMongoClient,
        mongo_database_name: str,
        mongo_collection_name: str,
        top_k: int = 5
) -> list[dict]:
    """
    Format the retrieve payloads by adding the document name, chunk sequence, previous content, content, next content

    :param retrieve_method: The method used to retrieve the similar vectors, enum in ["cosine_similarity", "bm25"]
    :param retrieve_payloads: The retrieve payloads, each payload is a dict containing the chunk_id and score
    :param mongo_client: The mongo client
    :param mongo_database_name: The mongo database name
    :param mongo_collection_name: The mongo collection name
    :param top_k: The number of top k retrieve payloads to return
    :return: The formatted retrieve payloads
    """

    processed_retrieve_payloads = retrieve_payloads
    for retrieve_payload in processed_retrieve_payloads:
        doc_name, chunk_seq, pre_content, content, next_content = mongo_client.get_neighbour_chunk_content_by_chunk_id(
            chunk_id=retrieve_payload["chunk_id"],
            database_name=mongo_database_name,
            collection_name=mongo_collection_name
        )
        retrieve_payload["doc_name"] = doc_name
        retrieve_payload["chunk_seq"] = chunk_seq
        retrieve_payload["pre_content"] = remove_overlap(pre_content, content).replace("\n", "")
        retrieve_payload["content"] = content.replace("\n", "")
        retrieve_payload["next_content"] = remove_overlap(next_content, content).replace("\n", "")
        retrieve_payload["ranking_method"] = retrieve_method

    processed_retrieve_payloads = sorted(processed_retrieve_payloads, key=lambda x: x["score"], reverse=True)[:top_k]

    return processed_retrieve_payloads


def format_retrieve_payload_parallel(
        retrieve_method: str,
        retrieve_payloads: list[dict[str, Any]],
        mongo_client: KBMongoClient,
        mongo_database_name: str,
        mongo_collection_name: str,
        top_k: int = 5
) -> list[dict]:
    """
    Format the retrieve payloads by adding the document name, chunk sequence, previous content, content, next content

    Using ThreadPoolExecutor to process the retrieve payloads in parallel.

    :param retrieve_method: The method used to retrieve the similar vectors, enum in ["cosine_similarity", "bm25"]
    :param retrieve_payloads: The retrieve payloads, each payload is a dict containing the chunk_id and score
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

    def process_payload(retrieve_payload: dict[str, Any]) -> dict[str, Any]:
        doc_name, chunk_seq, pre_content, content, next_content = mongo_client.get_neighbour_chunk_content_by_chunk_id(
            chunk_id=retrieve_payload["chunk_id"],
            database_name=mongo_database_name,
            collection_name=mongo_collection_name
        )
        retrieve_payload["doc_name"] = doc_name
        retrieve_payload["chunk_seq"] = chunk_seq
        retrieve_payload["pre_content"] = remove_overlap(pre_content, content).replace("\n", "")
        retrieve_payload["content"] = content.replace("\n", "")
        retrieve_payload["next_content"] = remove_overlap(next_content, content).replace("\n", "")
        retrieve_payload["ranking_method"] = retrieve_method
        return retrieve_payload

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_payload, retrieve_payload) for retrieve_payload in retrieve_payloads]
        processed_retrieve_payloads = [future.result() for future in as_completed(futures)]

    processed_retrieve_payloads = sorted(processed_retrieve_payloads, key=lambda x: x["score"], reverse=True)[:top_k]

    return processed_retrieve_payloads


def retrieve_payloads_and_format(
        retrieve_method: str,
        query_input: str | list[float],
        retrieve_client: KBQdrantClient | BM25Searcher,
        retrieve_client_scope_name: str,
        mongo_client: KBMongoClient,
        mongo_database_name: str,
        mongo_collection_name: str,
        top_k: int
) -> list[dict]:
    if retrieve_method not in ["cosine_similarity", "bm25"]:
        raise ValueError("Invalid retrieve method")

    if retrieve_method == "cosine_similarity":
        retrieve_client.checkout_collection(retrieve_client_scope_name)
        retrieve_payloads = retrieve_client.retrieve_similar_vectors_simply(
            query_vector=query_input,
            top_k=5,
        )
    else:  # retrieve_method == "bm25":
        retrieve_client.checkout_index(retrieve_client_scope_name)
        retrieve_payloads = retrieve_client.search(
            query=query_input,
            top_k=5
        )

    formatted_retrieve_payloads = format_retrieve_payload_parallel(
        retrieve_method=retrieve_method,
        retrieve_payloads=retrieve_payloads,
        mongo_client=mongo_client,
        mongo_database_name=mongo_database_name,
        mongo_collection_name=mongo_collection_name,
        top_k=top_k
    )

    return formatted_retrieve_payloads


def process_query(
        user_query_type: int,
        user_query: str,
        entity_list: List[str],
        llm_client: OpenAILLM,
        qdrant_client: KBQdrantClient,
        keyword_retriever: BM25Searcher,
        mongo_client: KBMongoClient,
        vector_search_params: Dict[str, Any],
        keyword_search_params: Dict[str, Any],
        top_k: int,
        hybrid_retriever: ReRanking,
) -> Tuple[Optional[List[Dict]], Optional[List[Dict]], Optional[List[Dict]], Optional[List[Dict]], Optional[List[Dict]], Optional[List[Dict]]]:
    """

    :param user_query_type:
    :param user_query:
    :param entity_list:
    :param llm_client:
    :param qdrant_client:
    :param keyword_retriever:
    :param mongo_client:
    :param vector_search_params:
    :param keyword_search_params:
    :param top_k:
    :param hybrid_retriever:

    :return:

    : `vector_search_params` structure:
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

    : `keyword_search_params` structure:
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

    if user_query_type == 0:  # Casual Chat
        return None, None, None, None, None, None

    # Embedding the user query
    embedding_response = llm_client.get_text_embedding(text=user_query)
    embedding_user_question = embedding_response.content

    ###############################################
    # Vector Search
    ###############################################
    # Vector search in chunk data
    vector_retrieve_payloads = retrieve_payloads_and_format(
        retrieve_method="cosine_similarity",
        query_input=embedding_user_question,
        retrieve_client=qdrant_client,
        retrieve_client_scope_name=vector_search_params["chunk_data"]["qdrant_collection_name"],
        mongo_client=mongo_client,
        mongo_database_name=vector_search_params["chunk_data"]["mongo_database_name"],
        mongo_collection_name=vector_search_params["chunk_data"]["mongo_collection_name"],
        top_k=top_k
    )

    # Vector search in QA
    qa_vector_retrieve_payloads = retrieve_payloads_and_format(
        retrieve_method="cosine_similarity",
        query_input=embedding_user_question,
        retrieve_client=qdrant_client,
        retrieve_client_scope_name=vector_search_params["qa"]["qdrant_collection_name"],
        mongo_client=mongo_client,
        mongo_database_name=vector_search_params["qa"]["mongo_database_name"],
        mongo_collection_name=vector_search_params["qa"]["mongo_collection_name"],
        top_k=top_k
    )

    ###############################################
    # Keyword Search
    ###############################################
    # Keyword search in chunk data
    keyword_retrieve_payloads = retrieve_payloads_and_format(
        retrieve_method="bm25",
        query_input=" ".join(entity_list),
        retrieve_client=keyword_retriever,
        retrieve_client_scope_name=keyword_search_params["chunk_data"]["lucene_index_dir"],
        mongo_client=mongo_client,
        mongo_database_name=keyword_search_params["chunk_data"]["mongo_database_name"],
        mongo_collection_name=keyword_search_params["chunk_data"]["mongo_collection_name"],
        top_k=top_k
    )
    # Keyword search in QA
    qa_keyword_retrieve_payloads = retrieve_payloads_and_format(
        retrieve_method="bm25",
        query_input=" ".join(entity_list),
        retrieve_client=keyword_retriever,
        retrieve_client_scope_name=keyword_search_params["qa"]["lucene_index_dir"],
        mongo_client=mongo_client,
        mongo_database_name=keyword_search_params["qa"]["mongo_database_name"],
        mongo_collection_name=keyword_search_params["qa"]["mongo_collection_name"],
        top_k=top_k
    )

    ###############################################
    # Hybrid Search
    ###############################################
    # Reranking chunk data
    chunks_candidate = vector_retrieve_payloads + keyword_retrieve_payloads
    hybrid_retriever.setup_reranking(rankings=chunks_candidate, k=top_k)
    reranking_item_list = hybrid_retriever.get_cross_encoder_scores(query=user_query)
    # Get original content from `chunks_candidate`
    for reranking_item in reranking_item_list:
        chunk_id = reranking_item["chunk_id"]
        for doc in chunks_candidate:
            if doc["chunk_id"] == chunk_id:
                reranking_item["doc_name"] = doc["doc_name"]
                reranking_item["chunk_seq"] = doc["chunk_seq"]
                reranking_item["pre_content"] = doc.get("pre_content", "")
                reranking_item["content"] = doc.get("content", "")
                reranking_item["next_content"] = doc.get("next_content", "")

    # Reranking qa
    qas_candidate = qa_vector_retrieve_payloads + qa_keyword_retrieve_payloads
    hybrid_retriever.setup_reranking(rankings=qas_candidate, k=top_k)
    reranking_item_list_qa = hybrid_retriever.get_cross_encoder_scores(query=user_query)
    # Get original content from `qas_candidate`
    for reranking_item in reranking_item_list_qa:
        chunk_id = reranking_item["chunk_id"]
        for doc in qas_candidate:
            if doc["chunk_id"] == chunk_id:
                reranking_item["doc_name"] = doc["doc_name"]
                reranking_item["chunk_seq"] = doc["chunk_seq"]
                reranking_item["pre_content"] = doc.get("pre_content", "")
                reranking_item["content"] = doc.get("content", "")
                reranking_item["next_content"] = doc.get("next_content", "")

    return vector_retrieve_payloads, qa_vector_retrieve_payloads, keyword_retrieve_payloads, qa_keyword_retrieve_payloads, reranking_item_list, reranking_item_list_qa

