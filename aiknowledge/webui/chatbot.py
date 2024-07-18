"""
Chatbot Page

$ Retrieving documents use qdrant client directly, not through FastAPI backend [http request is slow]
$ QA request use llm client directly, not through FastAPI backend [streaming output is hard to implement]
"""
import json
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from streamlit_float import float_init, float_css_helper, float_parent

from aiknowledge.config import app_config
from aiknowledge.llm import OpenAILLM
from aiknowledge.db import KBQdrantClient, KBMongoClient
from aiknowledge.rag.retriever.hybrid_search import ReRanking
from aiknowledge.rag.retriever.bm25 import BM25Searcher
from aiknowledge.rag.retriever.retriever import format_retrieve_payload, format_retrieve_payload_parallel, retrieve_payloads_and_format
from aiknowledge.webui.constants import (
    MONGO_DATABASE_INTFLEX_AUDIT, MONGO_COLLECTION_INTFLEX_AUDIT_QA, MONGO_COLLECTION_INTFLEX_AUDIT_CHUNK_DATA,
    QDRANT_COLLECTION_INTFLEX_AUDIT_CHUNK_DATA, LUCENE_INDEX_DIR_INTFLEX_AUDIT_CHUNK_DATA,
    LUCENE_INDEX_DIR_INTFLEX_AUDIT_QA, QDRANT_COLLECTION_INTFLEX_AUDIT_QA
)


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
def get_bm25_retriever():
    bm25_retriever = BM25Searcher(index_dir=LUCENE_INDEX_DIR_INTFLEX_AUDIT_CHUNK_DATA)
    bm25_retriever.set_language("zh")
    return bm25_retriever


@st.cache_resource
def get_hybrid_retriever():
    re_ranking_config = app_config.get("cohere")
    hybrid_search = ReRanking(api_key=re_ranking_config["api_key"])
    return hybrid_search


def chatbot_page():
    # Sidebar (Settings)
    with st.sidebar:
        with st.expander("⚙️ 检索设置", expanded=True):
            top_k = st.slider(label="Top K", min_value=1, max_value=10, value=6, step=1)
            # sim_threshold = st.slider(label="Similarity Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
            # additional_context_length = st.slider(label=":orange[Additional Context Length]", min_value=0, max_value=300, value=50, step=5)
        with st.expander("⚙️ 模型设置", expanded=True):
            chat_model_type = st.selectbox(label="模型选择", options=["gpt-4o", "gpt-4-turbo", ])
            temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1, help="0.0: 确定性, 1.0: 多样性")
            is_stream = st.toggle(label="流式输出", value=True)
            # chat_history_len = st.number_input(label="Chat History Length", min_value=0, max_value=20, value=0, step=2, disabled=True)

    token_and_time_cost_caption = False

    # Main Area: Chatbot & Retriever Panel
    col1, gap, col2 = st.columns([3, 0.01, 2])

    # Area 1(Left): Chatbot
    chatbot_container = col1.container(border=False, height=550)

    # Area 2(Right): Retriever Panel
    retriever_container = col2.container(border=False, height=700)

    with chatbot_container:

        # Chat input
        with st.container(border=False):
            float_init(theme=True, include_unstable_primary=False)
            user_input = st.chat_input("请输入问题...")
            button_css = float_css_helper(width="2.2rem", bottom="3rem", transition=0)
            float_parent(css=button_css)

        # Init chat message
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "这里是则成雨林，您的知识库问答助手，想问些什么？"}]

        # Display chat session message
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if user_input:
            llm_client = get_llm_client()
            qdrant_client = get_qdrant_client()
            mongo_client = get_mongo_client()

            keyword_retriever = get_bm25_retriever()

            # Add to messages session
            st.session_state.messages.append({"role": "user", "content": user_input})

            # User Message
            with st.chat_message("user"):
                st.markdown(user_input)

            # AI Message
            with st.chat_message("assistant"):
                query_analysis_list = []  # [(query_type: int, query: str, keywords: list), ...]
                st.session_state.qa_hybrid_retrieve_documents = []
                st.session_state.hybrid_retrieve_documents = []

                ###############################################
                # Query Decomposition & Entity Recognition
                ###############################################
                with st.spinner("问题分析中..."):
                    print("Start query decomposition...")
                    time1 = time.time()
                    # {"1": {"type": 0, "query": "xxx"}, ...}  0: casual, 1: enterprise
                    query_decomposition_response = llm_client.query_decomposition(user_input)
                    query_decomposition_response_json = json.loads(query_decomposition_response.content)
                    time2 = time.time()
                    print(f"Query decomposition time: {time2 - time1}")
                    if token_and_time_cost_caption:
                        chatbot_container.caption(f"query decomposition token usage: {query_decomposition_response.token_usage} | time cost: {time2 - time1}")

                    print("Start entity recognition...")
                    time1 = time.time()
                    entity_recognition_token_usage = 0
                    for query_decomposition in query_decomposition_response_json.values():
                        cur_query = query_decomposition.get("query", "")
                        cur_query_type = query_decomposition.get("type", 0)

                        # Entity Recognition for each query
                        entity_recognition_response = llm_client.entity_recognition(cur_query)  # {"entity": [...]}
                        entity_recognition_response_json = json.loads(entity_recognition_response.content)
                        query_analysis_list.append(
                            (cur_query_type, cur_query, entity_recognition_response_json.get("entity", []))
                        )
                        entity_recognition_token_usage += entity_recognition_response.token_usage

                    time2 = time.time()
                    print(f"Entity recognition time: {time2 - time1}")
                    if token_and_time_cost_caption:
                        chatbot_container.caption(f"entity recognition token usage: {entity_recognition_token_usage} | time cost: {time2 - time1}")

                ###############################################
                # Retrieve Documents for Each Query
                ###############################################
                with st.spinner("文档检索中..."):
                    for no_query, (user_query_type, user_query, entity_list) in enumerate(query_analysis_list):

                        if user_query_type == 0:  # Casual Chat
                            continue

                        ###############################################
                        # Vector Search
                        ###############################################
                        embedding_response = llm_client.get_text_embedding(text=user_query)  # Embedding
                        embedding_user_question, embedding_response_token_usage = embedding_response.content, embedding_response.token_usage

                        # Vector search in chunk data
                        vector_retrieve_payloads = retrieve_payloads_and_format(
                            retrieve_method="cosine_similarity",
                            query_input=embedding_user_question,
                            retrieve_client=qdrant_client,
                            retrieve_client_scope_name=QDRANT_COLLECTION_INTFLEX_AUDIT_CHUNK_DATA,
                            mongo_client=mongo_client,
                            mongo_database_name=MONGO_DATABASE_INTFLEX_AUDIT,
                            mongo_collection_name=MONGO_COLLECTION_INTFLEX_AUDIT_CHUNK_DATA,
                            top_k=top_k
                        )

                        # Vector search in QA
                        qa_vector_retrieve_payloads = retrieve_payloads_and_format(
                            retrieve_method="cosine_similarity",
                            query_input=embedding_user_question,
                            retrieve_client=qdrant_client,
                            retrieve_client_scope_name=QDRANT_COLLECTION_INTFLEX_AUDIT_QA,
                            mongo_client=mongo_client,
                            mongo_database_name=MONGO_DATABASE_INTFLEX_AUDIT,
                            mongo_collection_name=MONGO_COLLECTION_INTFLEX_AUDIT_QA,
                            top_k=top_k
                        )

                        ###############################################
                        # Keyword Search
                        ###############################################
                        # Keyword search in chunk data
                        keyword_retrieve_payloads = retrieve_payloads_and_format(
                            retrieve_method="bm25",
                            query_input=" ".join(entity_list),
                            retrieve_client=keyword_retriever,
                            retrieve_client_scope_name=LUCENE_INDEX_DIR_INTFLEX_AUDIT_CHUNK_DATA,
                            mongo_client=mongo_client,
                            mongo_database_name=MONGO_DATABASE_INTFLEX_AUDIT,
                            mongo_collection_name=MONGO_COLLECTION_INTFLEX_AUDIT_CHUNK_DATA,
                            top_k=top_k
                        )
                        # Keyword search in QA
                        qa_keyword_retrieve_payloads = retrieve_payloads_and_format(
                            retrieve_method="bm25",
                            query_input=" ".join(entity_list),
                            retrieve_client=keyword_retriever,
                            retrieve_client_scope_name=LUCENE_INDEX_DIR_INTFLEX_AUDIT_QA,
                            mongo_client=mongo_client,
                            mongo_database_name=MONGO_DATABASE_INTFLEX_AUDIT,
                            mongo_collection_name=MONGO_COLLECTION_INTFLEX_AUDIT_QA,
                            top_k=top_k
                        )

                        ###############################################
                        # Hybrid Search
                        hybrid_retriever = get_hybrid_retriever()
                        ###############################################

                        chunks_candidate = vector_retrieve_payloads + keyword_retrieve_payloads
                        hybrid_retriever.setup_reranking(rankings=chunks_candidate, k=top_k)
                        reranking_item_list = hybrid_retriever.get_cross_encoder_scores(query=user_query)
                        # Get original content from `chunks_candidate`
                        for reranking_item in reranking_item_list:
                            chunk_id = reranking_item["chunk_id"]
                            for doc in chunks_candidate:
                                if doc["chunk_id"] == chunk_id:
                                    reranking_item["doc_name"] = doc["doc_name"]
                                    reranking_item["chunk_seq"] = doc["chunk_seq"]
                                    reranking_item["pre_content"] = doc.get("pre_content", "")
                                    reranking_item["content"] = doc.get("content", "")
                                    reranking_item["next_content"] = doc.get("next_content", "")

                        qas_candidate = qa_vector_retrieve_payloads + qa_keyword_retrieve_payloads
                        hybrid_retriever.setup_reranking(rankings=qas_candidate, k=top_k)
                        reranking_item_list = hybrid_retriever.get_cross_encoder_scores(query=user_query)
                        # Get original content from `candidate_chunks`
                        for reranking_item in reranking_item_list:
                            chunk_id = reranking_item["chunk_id"]
                            for doc in qas_candidate:
                                if doc["chunk_id"] == chunk_id:
                                    reranking_item["doc_name"] = doc["doc_name"]
                                    reranking_item["chunk_seq"] = doc["chunk_seq"]
                                    reranking_item["pre_content"] = doc.get("pre_content", "")
                                    reranking_item["content"] = doc.get("content", "")
                                    reranking_item["next_content"] = doc.get("next_content", "")

                    ###############################################
                    # Display Query Analysis Result & Retrieve Result
                    ###############################################
                    with retriever_container:
                        st.markdown(f'**问题{no_query + 1}**: :orange[{user_query}]')
                        if user_query_type == 0:
                            st.markdown("**问题类型**: :orange[日常聊天✅]")
                        elif user_query_type == 1:
                            st.markdown("**问题类型**: :orange[企业知识✅]")
                        st.markdown(f'**关键词**: :orange[{entity_list}]')

                        if user_query_type == 1:

                            with st.expander("向量检索结果", expanded=False):
                                for no, doc in enumerate(vector_retrieve_payloads):
                                    st.markdown(f'**来源{no + 1}**: **{doc["doc_name"]}** ***(chunk_seq:{doc["chunk_seq"]})***')
                                    st.markdown(f':red[相似度={doc["score"]}]')
                                    st.markdown(f':orange[{doc["pre_content"]}]:green{doc["content"]}:orange[{doc["next_content"]}]')
                                    st.divider()

                            with st.expander("关键词检索结果", expanded=False):
                                for no, doc in enumerate(keyword_retrieve_payloads):
                                    st.markdown(f'**来源{no + 1}**: **{doc["doc_name"]}** ***(chunk_seq:{doc["chunk_seq"]})***')
                                    st.markdown(f':red[相似度={doc["score"]}]')
                                    st.markdown(f':orange[{doc["pre_content"]}]:green{doc["content"]}:orange[{doc["next_content"]}]')
                                    st.divider()

                            with st.expander("混合搜索结果", expanded=False):
                                for no, doc in enumerate(st.session_state.hybrid_retrieve_documents[no_query]):
                                    st.markdown(f'**来源{no + 1}**: **{doc["doc_name"]}** ***(chunk_seq:{doc["chunk_seq"]})***')
                                    st.markdown(f':red[评分={doc["re_ranking_score"]}]')
                                    st.markdown(f':orange[{doc["pre_content"]}]:green{doc["content"]} :orange[{doc["next_content"]}]')
                                    st.divider()

                            st.divider()

                ###############################################
                # Generate LLM Response
                ###############################################
                # Formatted context for prompt
                prompt_context = ""
                prompt_context_chunk_ids = []  # !!!! multiple doc retrieve remove duplicate
                for no_query in range(len(st.session_state.hybrid_retrieve_documents)):
                    for doc in st.session_state.hybrid_retrieve_documents[no_query]:
                        chunk_id = doc["chunk_id"]
                        if chunk_id in prompt_context_chunk_ids:
                            continue
                        else:
                            prompt_context_chunk_ids.append(chunk_id)

                        doc_name = doc["doc_name"]
                        content = doc.get("content", "")
                        pre_content = doc.get("pre_content", "")
                        next_content = doc.get("next_content", "")
                        prompt_context += f"[文本来源]: 《{doc_name}》\n[正文]:{pre_content}{content}{next_content}\n<DIVIDER>"

                with st.spinner("AI思考中..."):
                    # Get chat history from st.session.state
                    # chat_history = st.session_state.messages
                    # if len(chat_history) > chat_history_len:
                    #     chat_history = chat_history[-chat_history_len:]
                    chat_history = []
                    print("Start to generate llm response...")
                    time1 = time.time()
                    if not is_stream:
                        ai_response = llm_client.get_chat_response(
                            user_question=user_input,
                            context=prompt_context,
                            chat_history=chat_history,
                            temperature=temperature,
                            model_name=chat_model_type
                        )
                        ai_response_token_usage = ai_response.token_usage
                        st.markdown(ai_response.content)
                    else:
                        response_generator = llm_client.stream_chat_response(
                            user_question=user_input,
                            context=prompt_context,
                            chat_history=chat_history,
                            temperature=temperature,
                            model_name=chat_model_type
                        )
                        ai_response = st.write_stream(response_generator)
                        ai_response_token_usage = llm_client.stream_chat_token_usage
                    time2 = time.time()

                    print(f"Generate AI response time: {time2 - time1}\n")
                    if token_and_time_cost_caption:
                        chatbot_container.caption(f"llm response token usage: {ai_response_token_usage} | time cost: {time2 - time1}")

                    if no_query != len(query_analysis_list) - 1:
                        st.markdown("----------")

                st.session_state.messages.append({"role": "assistant", "content": ai_response})

    # Vertical divider
    # with gap:
    #     st.markdown("""
    #     <style>
    #     .column-left {
    #         background-color: #000000;  /* 设置左列的背景颜色 */
    #         padding: 1px;  /* 添加一些内边距 */
    #         height: 550px;  /* 设置高度，以便背景色覆盖整个列 */
    #     }
    #     </style>
    #     """, unsafe_allow_html=True)
    #     st.markdown('<div class="column-left">', unsafe_allow_html=True)
    #     st.markdown('</div>', unsafe_allow_html=True)
