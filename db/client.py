from typing import Optional, List, Dict
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import PointStruct, VectorParams, Distance


# ========================= Qdrant Client =========================
class QAQdrantClient(QdrantClient):
    def __init__(self, collection_name: str, embedding_dim: int = 1536, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # Initialize super class

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
        # Set the current collection name
        self._collection_name = collection_name

    def insert_vector(self, vec_id: str, vector: List[float], payload: dict) -> int:
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

    def retrieve_similar_vectors(
            self,
            query_vector: List[float],
            top_k: int,
            sim_lower_bound: float = 0.8
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
                "score": point.score
            }
            retrieved_infos.append(info)
        return retrieved_infos

    def retrieve_similar_vectors_with_adjacent_context(
            self,
            query_vector: List[float],
            top_k: int = 3,
            sim_lower_bound: float = 0.8,
            adjacent_len: int = 0,
    ) -> List[Dict]:
        """
        Retrieve similar vectors with theirs adjacent context(by chunk_id)
        :param query_vector: query vector
        :param top_k: number of similar vectors to retrieve
        :param sim_lower_bound: similarity lower bound
        :param adjacent_len: adjacent length
        :return: list of retrieved payloads.
                Sample: [{"chunk_id": 1, "document_name": "xxx", "page_content": "xxx"}, ...]
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
        # Retrieve searched vectors adjacent points by payload(source and id)
        retrieved_infos = []
        for point in points:
            cur_document_id = point.payload['document_id']
            cur_document_name = point.payload['document_name']
            cur_chunk_id = point.payload['chunk_id']
            score = point.score

            # Pre-context search
            pre_context_filter_condition = Filter(
                must=[
                    FieldCondition(  # 匹配特定的 source 字段
                        key="metadata.document_id",
                        match=MatchValue(value=cur_document_id)
                    ),
                    FieldCondition(  # 匹配特定的 chunk_id 字段
                        key="metadata.chunk_id",
                        match=MatchValue(value=cur_chunk_id - 1)
                    )]
            )
            pre_point = self.search(
                collection_name=self._collection_name,
                query_vector=query_vector,  # not used
                query_filter=pre_context_filter_condition,
                limit=1,
                with_payload=True,
                with_vectors=False
            )

            # Next-context search
            next_context_filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(value=cur_document_id)
                    ),
                    FieldCondition(
                        key="metadata.chunk_id",
                        match=MatchValue(value=cur_chunk_id + 1)
                    )]
            )
            next_point = self.search(
                collection_name=self._collection_name,
                query_vector=query_vector,  # not used
                query_filter=next_context_filter_condition,
                limit=1,
                with_payload=True,
                with_vectors=False
            )

            # Merge adjacent points' page_content
            pre_content = pre_point[0].payload['page_content'][:-adjacent_len] if pre_point else ""
            next_content = next_point[0].payload['page_content'][adjacent_len:] if next_point else ""
            merge_page_content = pre_content + point.payload['page_content'] + next_content

            # Construct retrieved payload
            info = {
                "chunk_id": cur_chunk_id,
                "document_name": cur_document_name,
                "page_content": merge_page_content,
                "score": score
            }
            retrieved_infos.append(info)
        return retrieved_infos

    # def retrieve_similar_note_vec_ids(self, query_vec: List[float], limit: int = 5) -> list[UUID]:
    #     points = self.search(
    #         collection_name=self._collection_name,
    #         query_vector=query_vec,
    #         limit=limit
    #     )
    #
    #     # Vector IDs of similar notes
    #     vec_ids = list(map(
    #         lambda point: UUID(point.id),
    #         points
    #     ))
    #
    #     return vec_ids

    def drop_collection(self) -> None:
        # Delete collection
        self.delete_collection(self._collection_name)

        # Recreate the collection
        # self.recreate_collection(
        #     collection_name=self._collection_name,
        #     vectors_config=VectorParams(
        #         size=self._embedding_dim,
        #         distance=Distance.COSINE
        #     )
        # )
