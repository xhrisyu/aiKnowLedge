import json
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from aiknowledge.llm import OpenAILLM


class QueryType:
    CASUAL = 0
    ENTERPRISE = 1


def query_decomposition(
        query: str,
        llm_client: OpenAILLM
) -> List[Dict]:
    """
    Decompose the query into a structured format:

    [
        {"type": <query type, int>, "query": <sub-query content, str>},
        ...
    ]

    (type 0: casual, type 1: enterprise)

    """

    query_decomposition_response = llm_client.query_decomposition(query)
    query_decomposition_response_json = json.loads(query_decomposition_response.content)
    query_decomposition_list = list(query_decomposition_response_json.values())

    return query_decomposition_list


def entity_recognition(
        query: str,
        llm_client: OpenAILLM
) -> List[str]:
    """
    Entity recognition for the query:

    [
        "entity1", "entity2", ...
    ]

    """

    entity_recognition_response = llm_client.entity_recognition(query)
    entity_recognition_response_json = json.loads(entity_recognition_response.content)
    entity_list = entity_recognition_response_json.get("entity", [])

    return entity_list


def query_analysis_pipeline(
        query: str,
        llm_client: OpenAILLM
) -> List:
    """
    Query analysis pipeline:

    1. Query decomposition
    2. Entity recognition for each sub query

    return: [{"query": "", "type": 0, "entity": []}, ...]
    """

    query_analysis_list = []

    # Query decomposition
    query_decomposition_list = query_decomposition(query, llm_client)

    # Entity recognition for each sub query
    with ThreadPoolExecutor() as executor:
        futures = []
        future_to_index = {}  # 用于存储 Future 对象和它们的索引

        for idx, decomposition_item in enumerate(query_decomposition_list):
            cur_query, cur_query_type = decomposition_item.get("query", ""), decomposition_item.get("type", 0)
            future = executor.submit(entity_recognition, cur_query, llm_client)
            future_to_index[future] = idx  # 存储 Future 对象和它们的索引
            futures.append(future)
            query_analysis_list.append({"query": cur_query, "type": cur_query_type})

        for future in as_completed(futures):
            idx = future_to_index[future]  # 获取 Future 对象的索引
            entity_list = future.result()
            query_analysis_list[idx]["entity"] = entity_list

    return query_analysis_list

