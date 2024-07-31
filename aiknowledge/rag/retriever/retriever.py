from typing import Any, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from aiknowledge.db import KBQdrantClient, KBMongoClient
from aiknowledge.llm import OpenAILLM
from aiknowledge.rag.retriever.bm25 import BM25
from aiknowledge.rag.retriever.rerank import Rerank
from aiknowledge.utils.tools import remove_overlap
from aiknowledge.config import app_config


def get_chunk_payload_task(
        retrieve_payload: Dict[str, Any],
        mongo_client: KBMongoClient,
        mongo_database_name: str,
        mongo_collection_name: str,
) -> Dict[str, Any]:
    """
    > input `retrieve_payload` structure:
    {
        "score": <float number>,
        "chunk_id": <uuid>,
    }

    > return formatted `retrieve_payload` structure:
    {
        "score": 0.1,
        "chunk_id": "",
        "doc_name": "",
        "chunk_seq": 1,
        "pre_content": "",
        "content": "",
        "next_content": "",
    }
    """
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

    return retrieve_payload


def vector_search(
        query_input: List[float],
        vector_db_client: KBQdrantClient,
        vector_collection_name: str,
        mongo_client: KBMongoClient,
        mongo_database_name: str,
        mongo_collection_name: str,
) -> List[Dict]:
    """
    Format the vector retrieved payloads by adding the document name, chunk sequence, previous content, content, next content
    Using ThreadPoolExecutor to process the retrieve payloads in parallel.

    :param query_input: The query input, either a string or an embedded vector list
    :param vector_db_client: Vector database client
    :param vector_collection_name: The vector collection name
    :param mongo_client: The mongo client
    :param mongo_database_name: The mongo database name
    :param mongo_collection_name: The mongo collection name
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
    vector_db_client.checkout_collection(vector_collection_name)
    retrieve_payloads = vector_db_client.retrieve_similar_vectors_simply(
        query_vector=query_input,
        top_k=top_k,
    )
    # Create thread pool executor to process the retrieve payloads in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_chunk_payload_task,
                                   retrieve_payload,
                                   mongo_client,
                                   mongo_database_name,
                                   mongo_collection_name,)
                   for retrieve_payload in retrieve_payloads]
        processed_retrieve_payloads = [future.result() for future in as_completed(futures)]

    # processed_retrieve_payloads = []
    # for retrieve_payload in retrieve_payloads:
    #     processed_retrieve_payload = get_chunk_payload_task(
    #         retrieve_payload=retrieve_payload,
    #         mongo_client=mongo_client,
    #         mongo_database_name=mongo_database_name,
    #         mongo_collection_name=mongo_collection_name,
    #     )
    #     processed_retrieve_payloads.append(processed_retrieve_payload)

    processed_retrieve_payloads = sorted(processed_retrieve_payloads, key=lambda x: x["score"], reverse=True)[:top_k]
    return processed_retrieve_payloads


def bm25_search(
        query_input: str,
        bm25_client: BM25,
        index_dir: str,
        mongo_client: KBMongoClient,
        mongo_database_name: str,
        mongo_collection_name: str,
) -> List[Dict]:
    """
    Format the keyword retrieved payloads by adding the document name, chunk sequence, previous content, content, next content
    Using ThreadPoolExecutor to process the retrieve payloads in parallel.

    :param query_input: The query input, either a string or an embedded vector list
    :param bm25_client: BM25 client
    :param index_dir: The bm25 index name
    :param mongo_client: The mongo client
    :param mongo_database_name: The mongo database name
    :param mongo_collection_name: The mongo collection name
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
    bm25_client.checkout_index(index_dir)
    retrieve_payloads = bm25_client.search(
        query=query_input,
        top_k=top_k
    )
    # Create thread pool executor to process the retrieve payloads in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(get_chunk_payload_task,
                                   retrieve_payload,
                                   mongo_client,
                                   mongo_database_name,
                                   mongo_collection_name)
                   for retrieve_payload in retrieve_payloads]
        processed_retrieve_payloads = [future.result() for future in as_completed(futures)]

    # processed_retrieve_payloads = []
    # for retrieve_payload in retrieve_payloads:
    #     processed_retrieve_payload = get_chunk_payload_task(
    #         retrieve_payload=retrieve_payload,
    #         mongo_client=mongo_client,
    #         mongo_database_name=mongo_database_name,
    #         mongo_collection_name=mongo_collection_name,
    #     )
    #     processed_retrieve_payloads.append(processed_retrieve_payload)

    processed_retrieve_payloads = sorted(processed_retrieve_payloads, key=lambda x: x["score"], reverse=True)[:top_k]
    return processed_retrieve_payloads


def retrieve_pipeline_parallel(
        user_query: str,
        entity_list: List[str],
        llm_client: OpenAILLM,
        qdrant_client: KBQdrantClient,
        keyword_retriever: BM25,
        mongo_client: KBMongoClient,
        vector_search_params: Dict[str, Any],
        keyword_search_params: Dict[str, Any],
        top_k: int,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
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
        return vector_search(
            # retrieve_method="cosine_similarity",
            query_input=embedding_user_question,
            vector_db_client=qdrant_client,
            vector_collection_name=vector_search_params["chunk_data"]["qdrant_collection_name"],
            mongo_client=mongo_client,
            mongo_database_name=vector_search_params["chunk_data"]["mongo_database_name"],
            mongo_collection_name=vector_search_params["chunk_data"]["mongo_collection_name"],
        )

    def task_vector_search_2():
        return vector_search(
            query_input=embedding_user_question,
            vector_db_client=qdrant_client,
            vector_collection_name=vector_search_params["qa"]["qdrant_collection_name"],
            mongo_client=mongo_client,
            mongo_database_name=vector_search_params["qa"]["mongo_database_name"],
            mongo_collection_name=vector_search_params["qa"]["mongo_collection_name"],
        )

    def task_keyword_search_1():
        return bm25_search(
            # retrieve_method="bm25",
            query_input=" ".join(entity_list),
            bm25_client=keyword_retriever,
            index_dir=keyword_search_params["chunk_data"]["lucene_index_dir"],
            mongo_client=mongo_client,
            mongo_database_name=keyword_search_params["chunk_data"]["mongo_database_name"],
            mongo_collection_name=keyword_search_params["chunk_data"]["mongo_collection_name"],
        )

    def task_keyword_search_2():
        return bm25_search(
            query_input=" ".join(entity_list),
            bm25_client=keyword_retriever,
            index_dir=keyword_search_params["qa"]["lucene_index_dir"],
            mongo_client=mongo_client,
            mongo_database_name=keyword_search_params["qa"]["mongo_database_name"],
            mongo_collection_name=keyword_search_params["qa"]["mongo_collection_name"],
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
        hybrid_retriever = Rerank(api_key=cohere_config["api_key"])
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
        hybrid_retriever = Rerank(api_key=cohere_config["api_key"])
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


def retrieve_pipeline(
        user_query: str,
        entity_list: List[str],
        llm_client: OpenAILLM,
        qdrant_client: KBQdrantClient,
        keyword_retriever: BM25,
        mongo_client: KBMongoClient,
        vector_search_params: Dict[str, Any],
        keyword_search_params: Dict[str, Any],
        top_k: int,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
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
        return vector_search(
            # retrieve_method="cosine_similarity",
            query_input=embedding_user_question,
            vector_db_client=qdrant_client,
            vector_collection_name=vector_search_params["chunk_data"]["qdrant_collection_name"],
            mongo_client=mongo_client,
            mongo_database_name=vector_search_params["chunk_data"]["mongo_database_name"],
            mongo_collection_name=vector_search_params["chunk_data"]["mongo_collection_name"],
        )

    def task_vector_search_2():
        return vector_search(
            query_input=embedding_user_question,
            vector_db_client=qdrant_client,
            vector_collection_name=vector_search_params["qa"]["qdrant_collection_name"],
            mongo_client=mongo_client,
            mongo_database_name=vector_search_params["qa"]["mongo_database_name"],
            mongo_collection_name=vector_search_params["qa"]["mongo_collection_name"],
        )

    def task_keyword_search_1():
        return bm25_search(
            # retrieve_method="bm25",
            query_input=" ".join(entity_list),
            bm25_client=keyword_retriever,
            index_dir=keyword_search_params["chunk_data"]["lucene_index_dir"],
            mongo_client=mongo_client,
            mongo_database_name=keyword_search_params["chunk_data"]["mongo_database_name"],
            mongo_collection_name=keyword_search_params["chunk_data"]["mongo_collection_name"],
        )

    def task_keyword_search_2():
        return bm25_search(
            query_input=" ".join(entity_list),
            bm25_client=keyword_retriever,
            index_dir=keyword_search_params["qa"]["lucene_index_dir"],
            mongo_client=mongo_client,
            mongo_database_name=keyword_search_params["qa"]["mongo_database_name"],
            mongo_collection_name=keyword_search_params["qa"]["mongo_collection_name"],
        )

    vector_retrieve_payloads = task_vector_search_1()
    qa_vector_retrieve_payloads = task_vector_search_2()
    keyword_retrieve_payloads = task_keyword_search_1()
    qa_keyword_retrieve_payloads = task_keyword_search_2()

    ###############################################
    # Hybrid Search
    ###############################################
    cohere_config = app_config.get("cohere")

    def task_reranking_1():
        # Reranking chunk data
        chunks_candidates = vector_retrieve_payloads + keyword_retrieve_payloads

        # Init hybrid retriever
        hybrid_retriever = Rerank(api_key=cohere_config["api_key"])
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
        hybrid_retriever = Rerank(api_key=cohere_config["api_key"])
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

    reranking_item_list = task_reranking_1()
    reranking_item_list_qa = task_reranking_2()

    return vector_retrieve_payloads, qa_vector_retrieve_payloads, keyword_retrieve_payloads, qa_keyword_retrieve_payloads, reranking_item_list, reranking_item_list_qa
