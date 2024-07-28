from typing import Optional, Any
import cohere


class Rerank:
    RECIPROCAL_RANK_FUSION = "rrf"
    CROSS_ENCODER = "cross_encoder"
    COHERE_CROSS_ENCODER_MODEL_NAME = "rerank-multilingual-v3.0"

    # Initialization
    def __init__(self, api_key: str):
        self._co = cohere.Client(api_key=api_key)
        self._k = None
        self._contents = None
        self._rankings = None

    def setup_reranking(self, rankings: list[dict], k: int):
        """
        Setup the re-ranking object with the rankings and the number of chunks to return
        :param rankings:
        :param k:
        :return:

        > rankings structure:
        [
            {
                "chunk_id": "",
                "chunk_seq": 1,
                "content": "",
                "doc_name": "",
                "ranking_method": "",
                "score": 0.8,
            },
            ...
        ]
        """

        self._k = k
        self._rankings = rankings
        self._contents = self.extract_all_contents()

    # Get the ranking object
    def get_ranking(self) -> list[dict]:
        return self._rankings

    # Set the ranking object
    def set_ranking(self, ranking: list[dict]):
        self._rankings = ranking

    def extract_all_contents(self) -> list[tuple]:
        """
        Extract all the pairs of the chunk_id and content
        :return: [(chunk_id, content), ...]
        """
        unique_contents = set()
        for record in self._rankings:
            unique_contents.add((record['chunk_id'], record['content']))
        return list(unique_contents)

    # Extract unique ranking methods
    def get_ranking_method(self) -> set[Any]:
        unique_ranking_methods = set()
        for record in self._rankings:
            unique_ranking_methods.add(record['ranking_method'])
        return unique_ranking_methods

    # Function to extract records for a specific ranking method
    def extract_records_by_ranking_method(self, ranking_method: str) -> list[dict]:
        return [record for record in self._rankings if record['ranking_method'] == ranking_method]

    # Function to extract and sort records for a specific ranking method
    def extract_and_sort_records_by_ranking_method(self, ranking_method: str) -> list[dict]:
        return self.sort_by(self.extract_records_by_ranking_method(ranking_method), 'score')

    # Function to extract and sort records for a specific ranking method
    def sort_by(self, records: list[dict], col_name: str) -> list[dict]:
        return sorted(records, key=lambda x: x[col_name], reverse=True)

    # Function to get the ordinal number of a specific chunk_id in its rank method
    def get_ordinal_number(self, chunk_id: str, ranking_method: str) -> Optional[int]:
        # Extract and sort records for the specified ranking method
        sorted_records = self.extract_and_sort_records_by_ranking_method(ranking_method)

        # Find the ordinal number of the specified chunk_id
        for index, record in enumerate(sorted_records, start=1):
            if record['chunk_id'] == chunk_id:
                return index

        # Return None if the chunk_id is not found
        return None

    # Calculate Reciprocal Rank Fusion score for each chunk      
    def get_reciprocal_rank_fusion_score(self, chunk_id: str, c: int = 42) -> float:
        score = 0.0
        ranking_methods = self.get_ranking_method()
        for ranking_method in ranking_methods:
            chunk_order = self.get_ordinal_number(chunk_id, ranking_method)
            if chunk_order is not None:
                score += 1.0 / (c + chunk_order)
        return score

    # Calculate Reciprocal Rank Fusion scores for all the chunks and return the ids records
    def get_rank_reciprocal_rank_fusion_scores(self, c: int = 42) -> list[dict]:
        records = []
        for chunk_id, _ in self._contents:
            re_rerank_score = self.get_reciprocal_rank_fusion_score(chunk_id, c)

            records.append({
                "chunk_id": chunk_id,
                "ranking_method": self.RECIPROCAL_RANK_FUSION,
                "re_ranking_score": re_rerank_score,
            })

        return self.sort_by(records, "re_ranking_score")[: self._k]

    def get_cross_encoder_scores(self, query: str) -> list[dict]:
        """
        Get the cross encoder scores for all the chunks and return the ids records

        :param query: The query to search.
        :return: The relevant documents.

        structure:
        [
            {
                "chunk_id": <chunk_id>,
                "ranking_method": <ranking_method>,
                "re_ranking_score": <re_ranking_score>,
            },
            ...
        ]

        """
        results = self._co.rerank(
            model=self.COHERE_CROSS_ENCODER_MODEL_NAME,
            query=query,
            documents=[content for _, content in self._contents],
            top_n=self._k,
            return_documents=True
        )

        re_ranking_results = [(re_ranking_record.index, re_ranking_record.relevance_score) for re_ranking_record in
                              results.results]

        records = [{
            "chunk_id": self._contents[index][0],  # chunk_id is in the first element in the tuples of the contents
            "ranking_method": self.CROSS_ENCODER,
            "re_ranking_score": score
        } for (index, score) in re_ranking_results]

        return self.sort_by(records, "re_ranking_score")[: self._k]
