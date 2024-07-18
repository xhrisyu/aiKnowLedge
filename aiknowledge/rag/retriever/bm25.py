import os
from typing import Optional
import numpy as np
import re

from pyserini.search.lucene import LuceneSearcher


class BM25Searcher:
    """
    Notes
    -----
        The basic usage of `LuceneSearcher` follows the user guide:
        https://github.com/castorini/pyserini/blob/master/docs/usage-search.md
    """

    def __init__(
            self,
            index_dir: os.PathLike | str,
            language="zh"
    ) -> None:
        """
        A searcher that searches relevant documents from the provided document base.

        Parameters
        ----------
            index_dir (PathLike): Directory where the trained metadata of the searched is stored.
            documents (Series): The documents to search.
            language (str): Language.
        """

        # convert path to string
        # * this is necessary since `LuceneSearcher` only accepts string
        index_dir = str(index_dir)

        # load searcher from trained metadata in index dir
        self._searcher = LuceneSearcher(index_dir)

        # set up the language
        self._language = language
        self.set_language(language)

    def checkout_index(self, index_dir: os.PathLike | str):
        index_dir = str(index_dir)
        self._searcher = LuceneSearcher(index_dir)
        self.set_language(self._language)

    def set_language(self, language: str):
        """
        Set the language of the searcher.

        Parameters
        ----------
            language (str): Language, e.g., "zh", "en".
        """

        self._searcher.set_language(language)

    def search(self, query: str, top_k: int) -> Optional[list]:
        """
        Search relevant documents from the document base.

        :param query: The query to search.
        :param top_k: The number of documents to return.
        :return: The relevant documents.
        > structure:
        [
            {
                "chunk_id": <chunk_id>,
                "score": <score>,
            },
            ...
        ]
        """

        return self.convert_hits(self._searcher.search(query, k=top_k))

    @staticmethod
    def get_score_thres(scores: np.ndarray) -> float:

        score_thres = np.quantile(scores, 0.3)
        return score_thres

    @staticmethod
    def convert_hits(hits: list) -> Optional[list]:
        """
        Convert search result to dict.
        :param hits: search result
        :return: docid and score
        > hits structure:
        {
            <docid>: <score>,
            ...
        }
        > return structure:
        [
            {
                "chunk_id": <chunk_id>,
                "score": <score>,
            },
            ...
        ]
        """
        result = []
        if hits:
            return [{"chunk_id": hit.docid, "score": hit.score} for hit in hits]

        return result

    @staticmethod
    def parse_docid(docid: str) -> tuple:
        """
        Use regex to parse docid to `doc_id` and `chunk_id` in MongoDB.
        > Structure: doc_id<chunk_id>
        > Example: bfc69511-5e42-49c3-88fc-4ef0b6e39b36<0>

        :param docid: docid string
        :return: doc_id, chunk_id
        """
        match = re.match(r"(.+)<(\d+)>", docid)
        if match:
            doc_id, chunk_id = match.groups()
            return str(doc_id), int(chunk_id)
        else:
            return None, None
