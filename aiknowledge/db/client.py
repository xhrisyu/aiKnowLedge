from typing import Optional, List, Dict, Sequence, Union, Any, Tuple
from qdrant_client import QdrantClient, models
from qdrant_client.conversions import common_types as types
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import PointStruct, VectorParams, Distance

from pymongo import MongoClient


# ========================= Qdrant Client =========================
class QAQdrantClient(QdrantClient):
    def __init__(self, collection_name: str, embedding_dim: int = 1536, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Create a new collection if it does not exist
        if collection_name and collection_name not in self.collection_names:
            self.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Qdrant Collection <{collection_name}> Created")

        self._collection_name = collection_name
        self._embedding_dim = embedding_dim

    @property
    def collection_name(self) -> Optional[str]:
        return self._collection_name

    @property
    def collection_names(self) -> List[str]:
        return list(map(
            lambda collection: collection.name,
            self.get_collections().collections
        ))

    def checkout_collection(self, collection_name: str) -> None:
        """
        Set the current collection name
        :param collection_name:
        :return:
        """

        # Check if the collection exists
        if collection_name not in self.collection_names:
            self.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Not found collection <{collection_name}>. Recreated.")

        self._collection_name = collection_name

    def insert_vector(self, vec_id: str, vector: List[float], payload: Optional[dict] = None) -> int:
        """
        Insert the embedding vector
        :param vec_id: 's id in vecdb
        :param vector: embedding vector
        :param payload: appending data
        :return: the number of inserted vectors
        """
        try:
            self.upsert(
                collection_name=self._collection_name,
                points=[PointStruct(
                    id=vec_id,
                    vector=vector,
                    payload=payload
                )]
            )
            return 1
        except Exception as e:
            print(f"Qdrant insert vectors error: {e}")
            return 0

    def insert_vectors(self, vectors_data: List[Dict]) -> int:
        """
        Insert the embedding vectors
        :param vectors_data: format: [{"vec_id": 1, "vector": [], "payload": {}}, ...]
        :return: the number of inserted vectors
        """
        try:
            self.upsert(
                collection_name=self._collection_name,
                points=[PointStruct(
                    id=vec['vec_id'],
                    vector=vec['vector'],
                    payload=vec['payload']
                ) for vec in vectors_data]
            )
            return len(vectors_data)
        except Exception as e:
            print(f"Qdrant insert vectors error: {e}")
            return 0

    def remove_vectors_by_document_id(self, document_ids: List[str]) -> int:
        """
        Remove vectors by doc_id
        :param document_ids: list of doc_id
        :return: the number of removed vectors
        """
        try:
            self.delete(
                collection_name=self._collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            ) for document_id in document_ids
                        ],
                    )
                ),
            )
            return len(document_ids)
        except Exception as e:
            print(f"Qdrant remove vectors error: {e}")
            return 0

    def remove_vectors_by_id(self, vec_ids: str | list[str]) -> bool:
        """
        Remove vectors by vec_id
        :param vec_ids: vector id or list of vector id
        :return: the number of removed vectors
        """
        try:
            self.delete(
                collection_name=self._collection_name,
                points_selector=models.PointIdsList(
                    points=[vec_ids] if isinstance(vec_ids, str) else vec_ids
                ),
            )
            return True
        except Exception as e:
            print(f"Qdrant remove vectors error: {e}")
            return False

    def retrieve_similar_vectors_simply(
            self,
            query_vector: List[float],
            top_k: int = 3,
            sim_lower_bound: float = 0.0
    ) -> List[Dict]:
        """
        Retrieve similar vectors with payload

        :param query_vector: query vector
        :param top_k: number of similar vectors to retrieve
        :param sim_lower_bound: similarity lower bound
        :return: list of PointStruct information
        """

        """
        > info structure:
        [
            {
                "doc_id": <doc_id>,
                "score": <similarity, [0, 1]>
            },
            ...
        ]
        """
        points = self.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            score_threshold=sim_lower_bound,
        )
        retrieved_infos = []
        for point in points:
            info = {
                "chunk_id": point.id,
                "score": point.score,
            }
            retrieved_infos.append(info)
        return retrieved_infos

    def retrieve_similar_vectors(
            self,
            query_vector: List[float],
            top_k: int = 3,
            sim_lower_bound: float = 0.75
    ) -> List[Dict]:
        """
        Retrieve similar vectors
        :param query_vector: query vector
        :param top_k: number of similar vectors to retrieve
        :param sim_lower_bound: similarity lower bound
        :return: list of PointStruct information
        """
        # Retrieve similar points
        points = self.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            score_threshold=sim_lower_bound,
        )
        retrieved_infos = []
        for point in points:
            info = {
                "chunk_id": point.payload['chunk_id'],
                "document_name": point.payload['document_name'],
                "page_content": point.payload['page_content'],
                "score": point.score,
                "page": point.payload['page']
            }
            retrieved_infos.append(info)
        return retrieved_infos

    def retrieve_similar_vectors_with_adjacent_context(
            self,
            query_vector: List[float],
            top_k: int = 3,
            sim_lower_bound: float = 0.75,
            adjacent_len: int = 0
    ) -> List[Dict]:
        """
        Retrieve similar vectors with theirs adjacent context(by chunk_id)
        :param query_vector: query vector
        :param top_k: number of similar vectors to retrieve
        :param sim_lower_bound: similarity lower bound
        :param adjacent_len: adjacent length
        :return: list of retrieved payloads.
        [
            {
                "chunk_id": 1,
                "document_name": "xxx",
                "page_content": "xxx",
                "pre_page_content": "xxx",
                "next_page_content": "xxx",
                "score": 0.72,
                "page": 0
            },...
        ]
        """
        def qdrant_query_filter(_document_id: str, _chunk_id: int):
            return Filter(must=[
                FieldCondition(key="document_id", match=MatchValue(value=_document_id)),
                FieldCondition(key="chunk_id", match=MatchValue(value=_chunk_id))
            ])

        def merge_chunks(chunk1, chunk2):
            """
            Merge two chunks by removing the overlap
            """
            max_overlap = min(len(chunk1), len(chunk2))
            for i in range(max_overlap, 0, -1):
                # Start with chunk1's tail and chunk2's head
                if chunk1[-i:] == chunk2[:i]:
                    return chunk1 + chunk2[i:]
            return chunk1 + chunk2  # if no overlap, simply concatenate

        def remove_overlap(chunk1, chunk2):
            """
            Remove the overlap from chunk1
            """
            max_overlap = min(len(chunk1), len(chunk2))
            # Start with chunk1's tail and chunk2's head
            for i in range(max_overlap, 0, -1):
                if chunk1[-i:] == chunk2[:i]:
                    return chunk1[:-i]

            # Start with chunk1's head and chunk2's tail
            for i in range(max_overlap, 0, -1):
                if chunk1[:i] == chunk2[-i:]:
                    return chunk1[i:]

            return chunk1

        # Retrieve similar points
        points = self.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            score_threshold=sim_lower_bound,
        )

        # Retrieve searched vectors adjacent points by payload(source and id)
        retrieved_infos = []
        for point in points:
            cur_document_id = point.payload['document_id']
            cur_chunk_id = point.payload['chunk_id']

            # Get adjacent points
            final_pre_content, final_next_content = "", ""
            last_pre_new_len, last_next_new_len = 0, 0  # last round new part length. If no update, break
            if adjacent_len > 0:
                adjacent_idx = 1
                while True:
                    # Pre-context & Next-context search
                    pre_point = self.search(collection_name=self._collection_name,
                                            query_vector=query_vector,  # no use
                                            query_filter=qdrant_query_filter(cur_document_id, cur_chunk_id - adjacent_idx),
                                            limit=1,
                                            with_payload=True,
                                            with_vectors=False)
                    next_point = self.search(collection_name=self._collection_name,
                                             query_vector=query_vector,
                                             query_filter=qdrant_query_filter(cur_document_id, cur_chunk_id + adjacent_idx),
                                             limit=1,
                                             with_payload=True,
                                             with_vectors=False)

                    # Get adjacent points' page_content
                    pre_content = pre_point[0].payload['page_content'] if pre_point else ""
                    next_content = next_point[0].payload['page_content'] if next_point else ""

                    if adjacent_idx == 1:
                        final_pre_content = remove_overlap(pre_content, point.payload['page_content'])
                        final_next_content = remove_overlap(next_content, point.payload['page_content'])
                    else:
                        pre_new_len, next_new_len = 0, 0
                        if pre_content:
                            final_pre_content = merge_chunks(pre_content, final_pre_content)
                            pre_new_len = len(final_pre_content) - len(pre_content)
                        if next_content:
                            final_next_content = merge_chunks(final_next_content, next_content)
                            next_new_len = len(final_next_content) - len(next_content)
                        # if no update, then break
                        if (last_pre_new_len == 0 and pre_new_len == 0) or (last_next_new_len == 0 and next_new_len == 0):
                            break

                        last_pre_new_len = pre_new_len
                        last_next_new_len = next_new_len

                    # Break if pre_content and next_content are enough
                    if len(final_pre_content) >= adjacent_len and len(final_next_content) >= adjacent_len:
                        break

                    adjacent_idx += 1

            # final_pre_content = final_pre_content[-adjacent_len:] if len(final_pre_content) > adjacent_len else final_pre_content
            # final_next_content = final_next_content[:adjacent_len] if len(final_next_content) > adjacent_len else final_next_content
            # python 3.11 可自动判断长度截取
            final_pre_content = final_pre_content[-adjacent_len:]
            final_next_content = final_next_content[:adjacent_len]

            # Construct retrieved payload
            info = {
                "chunk_id": cur_chunk_id,
                "document_name": point.payload['document_name'],
                "page_content": point.payload['page_content'],
                "pre_page_content": final_pre_content,
                "next_page_content": final_next_content,
                "score": point.score,
                "page": point.payload['page'],
            }
            retrieved_infos.append(info)

        return retrieved_infos

    def drop_collection(self) -> None:
        # Delete collection
        self.delete_collection(self._collection_name)


class KBMongoClient(MongoClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_neighbour_chunk_content_by_chunk_id(
            self,
            database_name: str,
            collection_name: str,
            chunk_id: str,
    ) -> tuple[str | Any, int | Any, str | Any, str | Any, str | Any]:

        # Checkout to target collection
        collection = self[database_name][collection_name]

        # Get doc_id, doc_name and chunk_seq
        mongo_document = collection.find_one({"chunk_id": chunk_id})
        if mongo_document:
            doc_id = mongo_document.get("doc_id", "")
            doc_name = mongo_document.get("doc_name", "")
            chunk_seq = mongo_document.get("chunk_seq", "")
        else:
            return "", "", "", "", ""

        # Get previous, current and next chunk content
        pre_chunk = collection.find_one({"doc_id": doc_id, "chunk_seq": chunk_seq - 1})
        pre_content = pre_chunk.get("content", "") if pre_chunk else ""

        chunk = collection.find_one({"doc_id": doc_id, "chunk_seq": chunk_seq})
        content = chunk.get("content", "") if chunk else ""

        next_chunk = collection.find_one({"doc_id": doc_id, "chunk_seq": chunk_seq + 1})
        next_content = next_chunk.get("content", "") if next_chunk else ""

        return doc_name, chunk_seq, pre_content, content, next_content



