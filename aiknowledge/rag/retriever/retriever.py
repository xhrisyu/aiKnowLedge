from typing import Any

from aiknowledge.db import KBQdrantClient, KBMongoClient
from aiknowledge.rag.retriever.bm25 import BM25Searcher
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
):
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

