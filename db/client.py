from typing import Optional, List, Dict, Tuple
from sshtunnel import SSHTunnelForwarder
import psycopg2
from redis import StrictRedis
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import ScoredPoint, Filter, FieldCondition, MatchValue
from qdrant_client.models import PointStruct, VectorParams, Distance


# ========================= Postgresql Client =========================
class PostgresqlClient:
    def __init__(self, ssh_config, database, user, password, host, port):
        if ssh_config is not None:
            with SSHTunnelForwarder(
                    (ssh_config['ssh_host'], ssh_config['ssh_port']),
                    ssh_username=ssh_config['ssh_username'],
                    ssh_password=ssh_config['ssh_password'],
                    remote_bind_address=(ssh_config['pg_host'], ssh_config['pg_port'])
            ) as tunnel:
                self.conn = psycopg2.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password
                )
                self.cursor = self.conn.cursor()
        else:
            self.conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
            )
            self.cursor = self.conn.cursor()

    def execute_query(self, query, params=None):
        self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def execute_insert(self, query, params):
        self.cursor.execute(query, params)
        self.conn.commit()

    def execute_insert_many(self, query, params: List[Tuple]):
        self.cursor.executemany(query, params)
        self.conn.commit()

    def execute_delete(self, query, params):
        self.cursor.execute(query, params)
        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()


# ========================= Redis Client =========================
class ChatBotRedisClient(StrictRedis):
    def __init__(self, host="localhost", port=6379, password=None, db=0):
        super().__init__(
            host=host,
            port=port,
            password=password,
            db=db,
        )
        self.key_chat_history = "qa:{}"

    def get_chat_history(self, user_id: str) -> List[Dict]:
        key = self.key_chat_history.format(user_id)
        chat_history = self.lrange(key, 0, -1)[::-1]  # 左侧入队（最新），倒序后时间顺序为最旧到最新
        chat_history_decoded = [item.decode("utf-8") for item in chat_history]
        # print(f"chat_history_decoded: {chat_history_decoded}")
        if not chat_history_decoded:
            return []

        openai_messages = []
        for idx, message in enumerate(chat_history_decoded):
            role = "user" if idx % 2 == 0 else "assistant"
            openai_messages.append({"role": role, "content": message})

        # print(f"chat history: {openai_messages}\n\n\n")
        return openai_messages


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

        # Collection
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
