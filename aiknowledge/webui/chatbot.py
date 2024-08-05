"""
Chatbot Page

$ Retrieving documents use qdrant client directly, not through FastAPI backend [http request is slow]
$ QA request use llm client directly, not through FastAPI backend [streaming output is hard to implement]
"""
import time
import streamlit as st
from streamlit_float import float_init, float_css_helper, float_parent

from aiknowledge.config import app_config
from aiknowledge.config.log_config import setup_logging
from aiknowledge.llm import OpenAILLM
from aiknowledge.db import KBQdrantClient, KBMongoClient
from aiknowledge.rag.retriever.bm25 import BM25
from aiknowledge.rag.retriever.rerank import Rerank
from aiknowledge.rag.query_analysis.query_analysis import query_analysis_pipeline, QueryType
from aiknowledge.webui.constants import LUCENE_INDEX_DIR_INTFLEX_AUDIT_CHUNK_DATA
from aiknowledge.rag.retrieval_augmentation import retrieval_augmentation_pipeline
from aiknowledge.rag.context_prompt import format_context_prompt, format_qa_history_prompt


logger = setup_logging()


vector_search_params = {
    "chunk_data": {
        "qdrant_collection_name": "intflex_audit",
        "mongo_database_name": "intflex_audit",
        "mongo_collection_name": "chunk_data"
    },
    "qa": {
        "qdrant_collection_name": "intflex_audit_qa",
        "mongo_database_name": "intflex_audit",
        "mongo_collection_name": "qa"
    }
}
keyword_search_params = {
    "chunk_data": {
        "lucene_index_dir": "./aiknowledge/uploaded_file/indexes/chunk_data",
        "mongo_database_name": "intflex_audit",
        "mongo_collection_name": "chunk_data"
    },
    "qa": {
        "lucene_index_dir": "./aiknowledge/uploaded_file/indexes/qa",
        "mongo_database_name": "intflex_audit",
        "mongo_collection_name": "qa"
    }
}


@st.cache_resource
def get_llm_client():
    openai_config = app_config.get("openai")
    return OpenAILLM(api_key=openai_config["api_key"])


@st.cache_resource
def get_qdrant_client():
    qdrant_config = app_config.get("qdrant")
    return KBQdrantClient(
        url=qdrant_config["url"],
        collection_name=qdrant_config['collection_name']["intflex_audit"],
        embedding_dim=qdrant_config["embedding_dim"]
    )


@st.cache_resource
def get_mongo_client():
    mongo_config = app_config.get("mongo")
    return KBMongoClient(mongo_config['uri'])


@st.cache_resource
def get_hybrid_retriever():
    hybrid_config = app_config.get("cohere")
    return Rerank(api_key=hybrid_config["api_key"])


@st.cache_resource
def get_bm25_retriever():
    return BM25(index_dir=LUCENE_INDEX_DIR_INTFLEX_AUDIT_CHUNK_DATA)


def chatbot_page():

    # Sidebar (Settings)
    with st.sidebar:
        with st.expander("⚙️ 检索设置", expanded=True):
            top_k = st.slider(label="Top K", min_value=1, max_value=10, value=6, step=1)
            # sim_threshold = st.slider(label="Similarity Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
            # additional_context_length = st.slider(label=":orange[Additional Context Length]", min_value=0, max_value=300, value=50, step=5)
        with st.expander("⚙️ 模型设置", expanded=True):
            chat_model_type = st.selectbox(label="模型选择", options=["gpt-4o", "gpt-4-turbo", ])
            temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
                                    help="0.0: 确定性, 1.0: 多样性")
            # is_stream = st.toggle(label="流式输出", value=True)
            # chat_history_len = st.number_input(label="Chat History Length", min_value=0, max_value=20, value=0, step=2, disabled=True)
            # token_and_time_cost_caption = st.toggle(label="显示耗时&用量", value=False)
            word_limit_mode = st.radio("回答模式", ["详细回答", "精炼回答"], index=0, horizontal=True)

    # Main Area: Chatbot & Retriever Panel
    col1, gap, col2 = st.columns([3, 0.01, 2])

    # Area 1(Left): Chatbot
    chatbot_container = col1.container(border=False, height=550)

    # Area 2(Right): Retriever Panel
    retriever_container = col2.container(border=False, height=700)

    # Init chat message
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "这里是则成雨林，您的知识库问答助手，想问些什么？"}]

    with chatbot_container:
        # Chat input
        with st.container(border=False):
            float_init(theme=True, include_unstable_primary=False)
            user_input = st.chat_input("请输入问题...")
            button_css = float_css_helper(width="2.2rem", bottom="3rem", transition=0)
            float_parent(css=button_css)

        # Display chat session message
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if not user_input:
            return

        llm_client = get_llm_client()
        qdrant_client = get_qdrant_client()
        mongo_client = get_mongo_client()

        # Add to messages session
        st.session_state.messages.append({"role": "user", "content": user_input})

        # User Message
        with st.chat_message("user"):
            st.markdown(user_input)

        logger.info(f"User Input: {user_input}")
        logger.info(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        ###############################################
        # Query Decomposition & Entity Recognition
        ###############################################
        with st.spinner("问题分析中..."):
            time1 = time.time()
            query_analysis_list = query_analysis_pipeline(user_input, llm_client)
            time2 = time.time()
            logger.info(f"Query analysis time: {time2 - time1}")

        ###############################################
        # Retrieve Documents for Each Query
        ###############################################
        with st.spinner("文档检索中..."):
            time1 = time.time()
            reranking_payloads_list, qa_reranking_payloads_list = retrieval_augmentation_pipeline(
                query_analysis_list=query_analysis_list,
                llm_client=llm_client,
                mongo_client=mongo_client,
                qdrant_client=qdrant_client,
                keyword_retriever=get_bm25_retriever(),
                hybrid_retriever=get_hybrid_retriever(),
                vector_search_params=vector_search_params,
                keyword_search_params=keyword_search_params,
                top_k=top_k
            )
            time2 = time.time()
            logger.info(f"Retrieve documents time: {time2 - time1}")
            retrieved_log_str = ""
            for no_query in range(len(query_analysis_list)):
                retrieved_log_str += f'\t> Query {no_query + 1} | {query_analysis_list[no_query]["query"]} | {" ".join(query_analysis_list[no_query]["entity"])}\n'
                for reranking_payload in reranking_payloads_list[no_query]:
                    retrieved_log_str += f'\t\t>> {reranking_payload["doc_name"]} - {reranking_payload["chunk_seq"]}\n'
            logger.info(f"Retrieved Documents:\n{retrieved_log_str}")

        ###############################################
        # Generate LLM Response
        ###############################################
        # AI Message
        with st.chat_message("assistant"):
            # Convert retrieval result to context prompt
            prompt_context = format_context_prompt(reranking_payloads_list)
            qa_prompt_context = format_qa_history_prompt(qa_reranking_payloads_list)

            with st.spinner("AI思考中..."):
                time1 = time.time()
                word_limit_num = 100 if word_limit_mode == "精炼回答" else None
                response_generator = llm_client.stream_chat_response(
                    user_question=user_input,
                    context=prompt_context,
                    qa_history=qa_prompt_context,
                    temperature=temperature,
                    model_name=chat_model_type,
                    word_limit=word_limit_num
                )
                ai_response = st.write_stream(response_generator)
                ai_response_token_usage = llm_client.stream_chat_token_usage
                time2 = time.time()
                logger.info(f"LLM response time: {time2 - time1}\n")
                ai_response_log_str = "\t> " + ai_response.replace("\n", "\n\t> ")
                logger.info(f"LLM response:\n{ai_response_log_str}")

            st.session_state.messages.append({"role": "assistant", "content": ai_response})

    logger.info("=" * 50)

    ###############################################
    # Display Retrieve Documents
    ###############################################
    with retriever_container:
        for no_query, query_analysis in enumerate(query_analysis_list):
            user_query_type, user_query, entity_list = query_analysis["type"], query_analysis["query"], query_analysis["entity"]
            st.markdown(f'**问题{no_query + 1}**: :orange[{user_query}]')
            if user_query_type == QueryType.CASUAL:
                st.markdown("**问题类型**: :orange[日常聊天✅]")
            elif user_query_type == QueryType.ENTERPRISE:
                st.markdown("**问题类型**: :orange[企业知识✅]")
            st.markdown(f'**关键词**: :orange[{entity_list}]')

            if user_query_type == 0:
                continue

            retrieved_document_expander = st.expander("文档检索结果", expanded=False)
            for no, doc in enumerate(reranking_payloads_list[no_query]):
                retrieved_document_expander.write(f':orange[**文档来源{no + 1}: {doc["doc_name"]}**]')
                retrieved_document_expander.write(doc["content"][0:200] + "......")
                retrieved_document_expander.divider()

            st.divider()
