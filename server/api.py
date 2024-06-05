import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import asyncio
from pymongo import MongoClient
from bson.objectid import ObjectId

from config import app_config
from db import QAQdrantClient
from llm import OpenAILLM
from utils.vec_processor import TextVectorProcessor

from server.api_paths import APIPaths
from server.models import *


@asynccontextmanager
async def lifespan(FastAPI_app: FastAPI):
    loop = asyncio.get_running_loop()
    mongo_config = app_config.get("mongo")
    qdrant_config = app_config.get("qdrant")
    openai_config = app_config.get("openai")

    FastAPI_app.state.mongo_client = await loop.run_in_executor(
        None,
        lambda: MongoClient(
            mongo_config['uri']
        ))
    FastAPI_app.state.qdrant_client = await loop.run_in_executor(
        None,
        lambda: QAQdrantClient(
            url=qdrant_config["url"],
            collection_name=qdrant_config['collection_name']["general"],
            embedding_dim=qdrant_config["embedding_dim"]
        ))
    FastAPI_app.state.llm_client = await loop.run_in_executor(
        None,
        lambda: OpenAILLM(
            api_key=openai_config["api_key"],
        ))

    try:
        yield
    finally:
        await loop.run_in_executor(None, FastAPI_app.state.mongo_client.close)
        await loop.run_in_executor(None, FastAPI_app.state.qdrant_client.close)


# ------------------------------------ API ------------------------------------
app = FastAPI(lifespan=lifespan)


# ------------------------------------ Mongo API ------------------------------------
@app.get(APIPaths.KB_GET)
async def get_kb_data(database_name, collection_name):
    try:
        mongo_client = app.state.mongo_client
        database = mongo_client[database_name]
        collection = database[collection_name]
        cursor = collection.find().allow_disk_use(True)
        data = list(cursor)
        if not data or len(data) == 0:
            return JSONResponse(content=[])

        # Form DataFrame
        # data = [{k: v.decode('utf-8') if isinstance(v, bytes) else v for k, v in uploaded_file.items()} for uploaded_file in cursor]  # Encode bytes to utf-8
        kb_dataframe = pd.DataFrame(data)
        kb_dataframe.fillna('', inplace=True)  # fill NaN with empty string
        kb_dataframe['_id'] = kb_dataframe['_id'].apply(lambda x: str(x))  # Convert `_id` object to string
        result = kb_dataframe.to_dict(orient='records')  # Convert DataFrame to dict
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(APIPaths.KB_INSERT)
async def insert_kb_data(params: KBInsertParams):
    try:
        mongo_client = app.state.mongo_client
        db_client = mongo_client[params.database_name]
        collection = db_client[params.collection_name]
        result = collection.insert_one(params.data.model_dump())
        return JSONResponse(content={"inserted_id": str(result.inserted_id)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(APIPaths.KB_UPDATE)
async def update_kb_data_by_id(params: KBUpdateParams):
    try:
        mongo_client = app.state.mongo_client
        db_client = mongo_client[params.database_name]
        collection = db_client[params.collection_name]
        result = collection.update_one({"_id": ObjectId(params.doc_id)}, {"$set": params.update_dict})
        return JSONResponse(content={"updated_count": result.modified_count})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(APIPaths.KB_REMOVE)
async def remove_kb_data_by_id(params: KBRemoveParams):
    try:
        mongo_client = app.state.mongo_client
        db_client = mongo_client[params.database_name]
        collection = db_client[params.collection_name]
        result = collection.delete_one({"_id": ObjectId(params.doc_id)})
        return JSONResponse(content={"deleted_count": result.deleted_count})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------ Vector API ------------------------------------
@app.post(APIPaths.VEC_INSERT)
async def insert_vec_data(params: VecInsertParams):
    try:
        # Preprocess text and convert to vectors
        vec_processor = TextVectorProcessor(
            document_id=params.data.doc_id,
            file_path=params.data.file_path,
            chunk_size=params.data.chunk_size,
            overlap_size=params.data.overlap_size,
            separators=params.data.separators,
            embedding_client=app.state.llm_client
        )
        vectors_data = vec_processor.process()

        # Insert vectors to VecDB
        vecdb_client = app.state.qdrant_client
        vecdb_client.checkout_collection(params.vecdb_collection_name)
        inserted_num = vecdb_client.insert_vectors(vectors_data)
        return JSONResponse(content={"inserted_num": inserted_num})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(APIPaths.VEC_REMOVE)
async def remove_vec_data(params: VecRemoveParams):
    try:
        vecdb_client = app.state.qdrant_client
        vecdb_client.checkout_collection(params.vecdb_collection_name)
        if isinstance(params.doc_ids, str):
            params.doc_ids = [params.doc_ids]
        vecdb_client.remove_vectors_by_document_id(params.doc_ids)
        return JSONResponse(content="successfully removed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(APIPaths.VEC_SEARCH)
async def search_vec_data(params: VecSearchParams):
    # Get Params
    user_question = params.user_question
    vecdb_collection_name = params.vecdb_collection_name
    top_k = params.top_k
    sim_threshold = params.sim_threshold
    # Get clients
    qdrant_client = app.state.qdrant_client
    llm_client = app.state.llm_client
    try:
        embedded_user_question = llm_client.get_text_embedding(text=user_question)
        qdrant_client.checkout_collection(vecdb_collection_name)
        retrieved_payloads = qdrant_client.retrieve_similar_vectors(
            query_vector=embedded_user_question,
            top_k=top_k,
            sim_lower_bound=sim_threshold,
        )  # [{"chunk_id": 1, "document_name": "xxx", "page_content": "xxx", "score": 0.72, "page": 0}, ...]
        return JSONResponse(content=retrieved_payloads)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(APIPaths.QUIZ_GENERATE)
async def generate_quiz(params: QuizGenerateParams):
    pass

# @app.post(APIPaths.QA)
# async def qa(params: QAParams):
#     # Get request parameters
#     user_question = params.userQuestion
#     context = params.context
#     temperature = params.temperature
#     model_name = params.modelName
#     is_stream = params.isStream
#     # Get client
#     llm_client = app.state.llm_client
#
#     try:
#         if is_stream:
#             async def llm_stream():
#                 async for chunk in llm_client.async_get_qa_response(
#                     context=context,
#                     user_question=user_question,
#                     temperature=temperature,
#                     model_name=model_name
#                 ):
#                     yield json.dumps({"content": chunk}) + "\n"
#             return StreamingResponse(llm_stream(), media_type="application/json")
#         else:
#             print(f"entering stream is false...model name is {model_name}")
#             response = llm_client.get_qa_response(
#                 context=context,
#                 user_question=user_question,
#                 temperature=temperature,
#                 model_name=model_name
#             )
#             return JSONResponse(content={"content": response})
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    fastapi_config = app_config.get("fastapi")
    uvicorn.run("api:app", host=fastapi_config["host"], port=fastapi_config["port"], reload=True)

# terminal command:
# uvicorn server.api:app --host 127.0.0.1 --port 8500  # 生产
# uvicorn server.api:app --host 127.0.0.1 --port 8500 --reload  # 测试
