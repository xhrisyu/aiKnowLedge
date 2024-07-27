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
from aiknowledge.rag.retriever.bm25 import BM25Searcher
from aiknowledge.rag.retriever.retriever import retrieve_pipeline
from aiknowledge.webui.constants import LUCENE_INDEX_DIR_INTFLEX_AUDIT_CHUNK_DATA


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

        # AI Message
        query_analysis_list = []  # [(query_type: int, query: str, keywords: list), ...]
        vector_retrieve_payloads_list, qa_vector_retrieve_payloads_list = [], []
        keyword_retrieve_payloads_list, qa_keyword_retrieve_payloads_list = [], []
        reranking_payloads_list, qa_reranking_payloads_list = [], []

        with st.chat_message("assistant"):

            ###############################################
            # Query Decomposition & Entity Recognition
            ###############################################
            with st.spinner("问题分析中..."):
                time1 = time.time()
                query_decomposition_response = llm_client.query_decomposition(user_input)
                query_decomposition_response_json = json.loads(query_decomposition_response.content)
                # query_decomposition_response_json: {"1": {"type": 0, "query": "xxx"}, ...}
                # 0: casual, 1: enterprise
                time2 = time.time()
                print(f"Query decomposition time: {time2 - time1}")

                time1 = time.time()
                entity_recognition_token_usage = 0
                for query_decomposition in query_decomposition_response_json.values():
                    cur_query, cur_query_type = query_decomposition.get("query", ""), query_decomposition.get("type", 0)

                    # Entity Recognition for each query
                    entity_recognition_response = llm_client.entity_recognition(cur_query)  # {"entity": [...]}
                    entity_recognition_response_json = json.loads(entity_recognition_response.content)
                    query_analysis_list.append(
                        (cur_query_type, cur_query, entity_recognition_response_json.get("entity", []))
                    )
                    entity_recognition_token_usage += entity_recognition_response.token_usage

                time2 = time.time()
                print(f"Entity recognition time: {time2 - time1}")

            ###############################################
            # Retrieve Documents for Each Query
            ###############################################
            with st.spinner("文档检索中..."):
                time1 = time.time()
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
                        "lucene_index_dir": "./aiknowledge/uploaded_file/indexes/chunk_data_index",
                        "mongo_database_name": "intflex_audit",
                        "mongo_collection_name": "chunk_data"
                    },
                    "qa": {
                        "lucene_index_dir": "./aiknowledge/uploaded_file/indexes/qa_index",
                        "mongo_database_name": "intflex_audit",
                        "mongo_collection_name": "qa"
                    }
                }

                # Retrieve documents for each query
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for user_query_type, user_query, entity_list in query_analysis_list:
                        if user_query_type == 0:
                            continue

                        futures.append(
                            executor.submit(retrieve_pipeline,
                                            user_query=user_query,
                                            entity_list=entity_list,
                                            llm_client=llm_client,
                                            qdrant_client=qdrant_client,
                                            keyword_retriever=BM25Searcher(index_dir=LUCENE_INDEX_DIR_INTFLEX_AUDIT_CHUNK_DATA),
                                            mongo_client=mongo_client,
                                            vector_search_params=vector_search_params,
                                            keyword_search_params=keyword_search_params,
                                            top_k=top_k
                                            )
                        )

                    for future in as_completed(futures):
                        result = future.result()
                        vector_retrieve_payloads, qa_vector_retrieve_payloads, \
                            keyword_retrieve_payloads, qa_keyword_retrieve_payloads, \
                            reranking_payloads, qa_reranking_payloads = result

                        vector_retrieve_payloads_list.append(vector_retrieve_payloads)
                        qa_vector_retrieve_payloads_list.append(qa_vector_retrieve_payloads)
                        keyword_retrieve_payloads_list.append(keyword_retrieve_payloads)
                        qa_keyword_retrieve_payloads_list.append(qa_keyword_retrieve_payloads)
                        reranking_payloads_list.append(reranking_payloads)
                        qa_reranking_payloads_list.append(qa_reranking_payloads)

                time2 = time.time()
                print(f"Retrieve documents time: {time2 - time1}")

            ###############################################
            # Display Retrieve Documents
            ###############################################
            with retriever_container:
                for no_query, (user_query_type, user_query, entity_list) in enumerate(query_analysis_list):
                    st.markdown(f'**问题{no_query + 1}**: :orange[{user_query}]')
                    if user_query_type == 0:
                        st.markdown("**问题类型**: :orange[日常聊天✅]")
                    elif user_query_type == 1:
                        st.markdown("**问题类型**: :orange[企业知识✅]")
                    st.markdown(f'**关键词**: :orange[{entity_list}]')

                    if user_query_type == 0:
                        continue

                    expander1 = st.expander("向量检索结果", expanded=False)
                    for no, doc in enumerate(vector_retrieve_payloads_list[no_query]):
                        expander1.markdown(f':orange[**文档来源{no + 1}: {doc["doc_name"]}**]')
                        # expander1.markdown(f':red[相似度={doc["score"]}]')
                        # expander1.markdown(f':orange[{doc["pre_content"]}] {doc["content"]} :orange[{doc["next_content"]}]')
                        # expander1.markdown(f'{doc["pre_content"]}{doc["content"]}{doc["next_content"]}')
                        expander1.markdown(doc["content"])
                        expander1.divider()

                    expander2 = st.expander("关键词检索结果", expanded=False)
                    for no, doc in enumerate(keyword_retrieve_payloads_list[no_query]):
                        expander2.markdown(f':orange[**文档来源{no + 1}: {doc["doc_name"]}**]')
                        expander2.markdown(doc["content"])
                        expander2.divider()

                    expander3 = st.expander("混合搜索结果", expanded=False)
                    for no, doc in enumerate(reranking_payloads_list[no_query]):
                        expander3.markdown(f':orange[**文档来源{no + 1}: {doc["doc_name"]}**]')
                        expander3.markdown(doc["content"])
                        expander3.divider()

            ###############################################
            # Generate LLM Response
            # ! Merge multiple query's retrieved documents into one prompt
            ###############################################
            # Formatted retrieved documents for prompt
            prompt_context = ""
            prompt_context_chunk_ids = []  # !!!! multiple doc retrieve remove duplicate
            for no_query in range(len(reranking_payloads_list)):
                for doc in reranking_payloads_list[no_query]:
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

            # Formatted retrieved qa documents for prompt
            qa_prompt_context = ""
            qa_prompt_context_chunk_ids = []
            for no_query in range(len(qa_reranking_payloads_list)):
                for doc in qa_reranking_payloads_list[no_query]:
                    chunk_id = doc["chunk_id"]
                    if chunk_id in qa_prompt_context_chunk_ids:
                        continue
                    else:
                        qa_prompt_context_chunk_ids.append(chunk_id)

                    content = doc.get("content", "")
                    qa_prompt_context += f"[参考历史问答]:\n{content}\n"

            with st.spinner("AI思考中..."):
                # Get chat history from QA chunk
                time1 = time.time()
                response_generator = llm_client.stream_chat_response(
                    user_question=user_input,
                    context=prompt_context,
                    qa_history=qa_prompt_context,
                    temperature=temperature,
                    model_name=chat_model_type
                )
                ai_response = st.write_stream(response_generator)
                ai_response_token_usage = llm_client.stream_chat_token_usage
                time2 = time.time()
                print(f"Generate AI response time: {time2 - time1}\n")

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
