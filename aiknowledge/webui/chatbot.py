"""
Chatbot Page

$ Retrieving documents use qdrant client directly, not through FastAPI backend [http request is slow]
$ QA request use llm client directly, not through FastAPI backend [streaming output is hard to implement]
"""
import json
import time
import streamlit as st
from streamlit_float import float_init, float_css_helper, float_parent

from aiknowledge.config import app_config
from aiknowledge.llm import OpenAILLM
from aiknowledge.db import QAQdrantClient, KBMongoClient
from aiknowledge.rag.retriever.hybrid_search import ReRanking
from aiknowledge.webui.constants import QDRANT_COLLECTION_INTFLEX
from aiknowledge.utils.tools import remove_overlap
from aiknowledge.rag.retriever.bm25_search import BM25Searcher


@st.cache_resource
def get_llm_client():
    openai_config = app_config.get("openai")
    return OpenAILLM(api_key=openai_config["api_key"])


@st.cache_resource
def get_qdrant_client():
    qdrant_config = app_config.get("qdrant")
    return QAQdrantClient(
        url=qdrant_config["url"],
        collection_name=qdrant_config['collection_name']["intflex_audit"],
        embedding_dim=qdrant_config["embedding_dim"]
    )


@st.cache_resource
def get_mongo_client():
    mongo_config = app_config.get("mongo")
    return KBMongoClient(mongo_config['uri'])


@st.cache_resource
def get_bm25_searcher():
    bm25_searcher = BM25Searcher(index_dir="./aiknowledge/uploaded_file/indexes/lucene-index")
    bm25_searcher.set_language("zh")
    return bm25_searcher


@st.cache_resource
def get_hybrid_searcher():
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
            keyword_searcher = get_bm25_searcher()

            # Add to messages session
            st.session_state.messages.append({"role": "user", "content": user_input})

            # User Message
            with st.chat_message("user"):
                st.markdown(user_input)

            # AI Message
            with st.chat_message("assistant"):
                query_analysis_list = []  # [(query_type: int, query: str, keywords: list), ...]
                st.session_state.vector_searched_documents = []
                st.session_state.keyword_searched_documents = []
                st.session_state.hybrid_searched_documents = []
                """ > searched_documents structure
                [
                    {
                        "score": 0.5,
                        "chunk_id": "",
                        "doc_name": "",
                        "chunk_seq": 1,
                        "pre_content": "",
                        "content": "",
                        "next_content": "",
                    },
                    ...
                ]
                """

                ###############################################
                # Query Decomposition & Entity recognition
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

                for no_query, (user_query_type, user_query, entity_list) in enumerate(query_analysis_list):
                    ###############################################
                    # Retrieve Documents
                    ###############################################

                    with st.spinner("文档检索中..."):
                        ###############################################
                        # Vector Search
                        ###############################################
                        time1 = time.time()
                        qdrant_client.checkout_collection(QDRANT_COLLECTION_INTFLEX)
                        embedding_response = llm_client.get_text_embedding(text=user_query)  # Embedding
                        embedding_user_question = embedding_response.content
                        embedding_response_token_usage = embedding_response.token_usage
                        time2 = time.time()
                        print(f"Embedding time cost: {time2 - time1}")
                        if token_and_time_cost_caption:
                            chatbot_container.caption(f"embedding token usage: {embedding_response_token_usage} | time cost: {time2 - time1}")

                        time1 = time.time()
                        vector_search_payloads = qdrant_client.retrieve_similar_vectors_simply(
                            query_vector=embedding_user_question,
                            top_k=5,
                        )

                        # Get chunk content from MongoDB (base on chunk_id and doc_name)
                        for vector_search_payload in vector_search_payloads:
                            doc_name, chunk_seq, pre_content, content, next_content = mongo_client.get_neighbour_chunk_content_by_chunk_id(
                                chunk_id=vector_search_payload["chunk_id"],
                                database_name="intflex_audit",
                                collection_name="chunk_data"
                            )
                            vector_search_payload["doc_name"] = doc_name
                            vector_search_payload["chunk_seq"] = chunk_seq
                            vector_search_payload["ranking_method"] = "cosine_similarity"
                            vector_search_payload["pre_content"] = remove_overlap(pre_content, content)
                            vector_search_payload["content"] = content
                            vector_search_payload["next_content"] = remove_overlap(next_content, content)

                        st.session_state.vector_searched_documents.append(vector_search_payloads)
                        # Sort retrieved documents result by score [from Vector Search]
                        st.session_state.vector_searched_documents[no_query] = sorted(st.session_state.vector_searched_documents[no_query], key=lambda x: x["score"], reverse=True)
                        st.session_state.vector_searched_documents[no_query] = st.session_state.vector_searched_documents[no_query][:top_k]
                        time2 = time.time()
                        print(f"Vector search time cost: {time2 - time1}")

                        ###############################################
                        # Keyword Search
                        ###############################################
                        print("Start to keyword search based on BM25...")
                        time1 = time.time()
                        keyword_str = " ".join(entity_list)
                        keyword_search_results = keyword_searcher.search(
                            query=keyword_str,
                            top_k=5
                        )

                        # Get original content from MongoDB (base on chunk_id and doc_name)
                        for keyword_search_result in keyword_search_results:
                            doc_name, chunk_seq, pre_content, content, next_content = mongo_client.get_neighbour_chunk_content_by_chunk_id(
                                chunk_id=keyword_search_result["chunk_id"],
                                database_name="intflex_audit",
                                collection_name="chunk_data"
                            )
                            keyword_search_result["doc_name"] = doc_name
                            keyword_search_result["chunk_seq"] = chunk_seq
                            keyword_search_result["ranking_method"] = "bm25"
                            keyword_search_result["pre_content"] = remove_overlap(pre_content, content)
                            keyword_search_result["content"] = content
                            keyword_search_result["next_content"] = remove_overlap(next_content, content)

                        st.session_state.keyword_searched_documents.append(keyword_search_results)

                        # Sort retrieved documents result by score [from Vector Search]
                        st.session_state.keyword_searched_documents[no_query] = sorted(st.session_state.keyword_searched_documents[no_query], key=lambda x: x["score"], reverse=True)
                        st.session_state.keyword_searched_documents[no_query] = st.session_state.keyword_searched_documents[no_query][:top_k]
                        time2 = time.time()
                        print(f"Keyword search time cost: {time2 - time1}")

                        ###############################################
                        # Hybrid Search
                        ###############################################
                        # Remove the duplicate documents from vector_searched_documents and keyword_searched_documents
                        # candidate_documents = st.session_state.vector_searched_documents[no_query] + st.session_state.keyword_searched_documents[no_query]
                        candidate_documents = st.session_state.keyword_searched_documents[no_query]
                        for vector_searched_candidate_docs in st.session_state.vector_searched_documents[no_query]:
                            cur_chunk_id = vector_searched_candidate_docs["chunk_id"]
                            if not any(doc["chunk_id"] == cur_chunk_id for doc in candidate_documents):
                                candidate_documents.append(vector_searched_candidate_docs)

                        hybrid_searcher = get_hybrid_searcher()
                        hybrid_searcher.setup_reranking(rankings=candidate_documents, k=top_k)
                        st.session_state.hybrid_searched_documents.append(hybrid_searcher.get_cross_encoder_scores(
                            query=user_query))

                        # Get original content from candidate_documents
                        for ranking_item in st.session_state.hybrid_searched_documents[no_query]:
                            chunk_id = ranking_item["chunk_id"]
                            for doc in candidate_documents:
                                if doc["chunk_id"] == chunk_id:
                                    ranking_item["doc_name"] = doc["doc_name"]
                                    ranking_item["chunk_seq"] = doc["chunk_seq"]
                                    ranking_item["pre_content"] = doc.get("pre_content", "")
                                    ranking_item["content"] = doc.get("content", "")
                                    ranking_item["next_content"] = doc.get("next_content", "")

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

                        with st.expander("向量检索结果", expanded=False):
                            for no, doc in enumerate(st.session_state.vector_searched_documents[no_query]):
                                content = doc.get("content", "").replace("\n", "")
                                pre_content = doc.get("pre_content", "").replace("\n", "")
                                next_content = doc.get("next_content", "").replace("\n", "")
                                st.markdown(f'**来源{no + 1}**: **{doc["doc_name"]}** ***(chunk_seq:{doc["chunk_seq"]})***')
                                st.markdown(f':red[相似度={doc["score"]}]')
                                st.markdown(f':orange[{pre_content}]:green{content} :orange[{next_content}]')
                                st.divider()

                        with st.expander("关键词检索结果", expanded=False):
                            for no, doc in enumerate(st.session_state.keyword_searched_documents[no_query]):
                                content = doc.get("content", "").replace("\n", "")
                                pre_content = doc.get("pre_content", "").replace("\n", "")
                                next_content = doc.get("next_content", "").replace("\n", "")
                                st.markdown(f'**来源{no + 1}**: **{doc["doc_name"]}** ***(chunk_seq:{doc["chunk_seq"]})***')
                                st.markdown(f':red[相似度={doc["score"]}]')
                                st.markdown(f':orange[{pre_content}]:green{content}:orange[{next_content}]')
                                st.divider()

                        with st.expander("混合搜索结果", expanded=False):
                            for no, doc in enumerate(st.session_state.hybrid_searched_documents[no_query]):
                                content = doc.get("content", "").replace("\n", "")
                                pre_content = doc.get("pre_content", "").replace("\n", "")
                                next_content = doc.get("next_content", "").replace("\n", "")
                                st.markdown(f'**来源{no + 1}**: **{doc["doc_name"]}** ***(chunk_seq:{doc["chunk_seq"]})***')
                                st.markdown(f':red[评分={doc["re_ranking_score"]}]')
                                st.markdown(f':orange[{pre_content}]:green{content} :orange[{next_content}]')
                                st.divider()

                        st.divider()

                ###############################################
                # Generate LLM Response
                ###############################################
                # Formatted context for prompt
                prompt_context = ""
                prompt_context_chunk_ids = []  # !!!! multiple doc retrieve remove duplicate
                for no_query in range(len(st.session_state.hybrid_searched_documents)):
                    for doc in st.session_state.hybrid_searched_documents[no_query]:
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
