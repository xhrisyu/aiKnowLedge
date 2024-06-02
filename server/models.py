from pydantic import BaseModel
from typing import List, Optional, Union


class KBInsertItem(BaseModel):
    file_name: str
    file_len: int
    file_extension: str
    chunk_size: int
    overlap_size: int
    separators: Optional[List[str]]
    create_time: str
    location: str
    in_vector_db: bool


class KBInsertParams(BaseModel):
    database_name: str
    collection_name: str
    data: KBInsertItem


class KBUpdateParams(BaseModel):
    database_name: str
    collection_name: str
    doc_id: str
    update_dict: dict  # {"update_field 1": "new_value 1", "update_field 2": "new_value 2", ...}


class KBRemoveParams(BaseModel):
    database_name: str
    collection_name: str
    doc_id: str


class VecInsertItem(BaseModel):
    doc_id: str
    file_path: str
    chunk_size: int
    overlap_size: int
    separators: Optional[List[str]]


class VecInsertParams(BaseModel):
    vecdb_collection_name: str
    data: VecInsertItem


class VecRemoveParams(BaseModel):
    vecdb_collection_name: str
    doc_ids: Union[List[str], str]


class VecSearchParams(BaseModel):
    vecdb_collection_name: str
    user_question: str
    top_k: int
    sim_threshold: float


class QAParams(BaseModel):
    userQuestion: str
    context: str
    temperature: float
    modelName: Optional[str]
    isStream: bool


class QuizGenerateParams(BaseModel):
    context: str
    quizNum: int
    quizType: str
    temperature: float
    modelName: Optional[str]
    isStream: bool

