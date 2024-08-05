from typing import List, Dict, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from aiknowledge.db import KBQdrantClient, KBMongoClient
from aiknowledge.llm import OpenAILLM
from aiknowledge.rag.retriever.bm25 import BM25
from aiknowledge.rag.retriever.rerank import Rerank
from aiknowledge.rag.retriever.retriever import retrieve_parallel
from aiknowledge.rag.query_analysis.query_analysis import QueryType


def retrieval_augmentation_pipeline(
        query_analysis_list: List[Dict],
        llm_client: OpenAILLM,
        mongo_client: KBMongoClient,
        qdrant_client: KBQdrantClient,
        keyword_retriever: BM25,
        hybrid_retriever: Rerank,
        vector_search_params: Dict[str, Any],
        keyword_search_params: Dict[str, Any],
        top_k: int,
) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    """
    Retrieve and rerank documents in parallel

    :param query_analysis_list: [{"query": "", "type": 0, "entity": []}, ...]
    :param llm_client:
    :param mongo_client:
    :param qdrant_client:
    :param keyword_retriever:
    :param hybrid_retriever:
    :param vector_search_params:
    :param keyword_search_params:
    :param top_k:
    :return: reranking payload list
    [
        # query 1
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
            ...
        ],
        # query 2
        [], ...
    ],
    [...]
    """

    reranking_payloads_list, qa_reranking_payloads_list = [], []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for query_no, query_analysis in enumerate(query_analysis_list):
            user_query_type, user_query, entity_list = query_analysis["type"], query_analysis["query"], query_analysis["entity"]

            # Skip casual query
            if user_query_type == QueryType.CASUAL:
                continue

            futures.append(
                executor.submit(retrieve_parallel,
                                query_no=query_no,
                                query=user_query,
                                entity_list=entity_list,
                                llm_client=llm_client,
                                mongo_client=mongo_client,
                                qdrant_client=qdrant_client,
                                keyword_retriever=keyword_retriever,
                                hybrid_retriever=hybrid_retriever,
                                vector_search_params=vector_search_params,
                                keyword_search_params=keyword_search_params,
                                top_k=top_k)
            )

        for future in as_completed(futures):
            # Considering that the processed result may not be in order, we need to store the query_no
            query_no, reranking_payloads, qa_reranking_payloads = future.result()

            # Insert the result into the corresponding position
            reranking_payloads_list.insert(query_no, reranking_payloads)
            qa_reranking_payloads_list.insert(query_no, qa_reranking_payloads)

    return reranking_payloads_list, qa_reranking_payloads_list
