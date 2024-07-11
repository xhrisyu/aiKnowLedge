"""
Chatbot Page

$ Retrieving documents use qdrant client directly, not through FastAPI backend [http request is slow]
$ QA request use llm client directly, not through FastAPI backend [streaming output is hard to implement]
"""
import json
import os
import time
import streamlit as st
from pymongo import MongoClient
from streamlit_float import float_init, float_css_helper, float_parent

from aiknowledge.config import app_config
from aiknowledge.llm import OpenAILLM
from aiknowledge.db import QAQdrantClient
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
    return MongoClient(mongo_config['uri'])


@st.cache_resource
def get_bm25searcher():
    searcher = BM25Searcher(index_dir="./aiknowledge/uploaded_file/indexes/lucene-index")
    searcher.set_language("zh")
    return searcher


def chatbot_page():
    # Sidebar (Settings)
    with st.sidebar:
        with st.expander("⚙️ Retriever", expanded=True):
            top_k = st.slider(label="Top K", min_value=1, max_value=10, value=3, step=1)
            sim_threshold = st.slider(label="Similarity Threshold", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
            additional_context_length = st.slider(label=":orange[Additional Context Length]", min_value=0, max_value=300, value=50, step=5)
        with st.expander("⚙️ Generator", expanded=True):
            chat_model_type = st.selectbox(label="Chat Model", options=["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o", ])
            temperature = st.slider(label="Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
            is_stream = st.toggle(label="Stream", value=True)
            chat_history_len = st.number_input(label="Chat History Length", min_value=0, max_value=20, value=6, step=2)

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
            st.session_state.messages = [{"role": "assistant", "content": "这里是则成雨林，您的知识库问答助手，想问些什么？"}]

        # Display chat session message
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if user_input:
            llm_client = get_llm_client()
            qdrant_client = get_qdrant_client()
            mongo_client = get_mongo_client()
            mongo_collection = mongo_client["intflex_audit"]["chunk_data"]
            searcher = get_bm25searcher()

            # User Message
            with st.chat_message("user"):
                st.markdown(user_input)

            # AI Message
            with st.chat_message("assistant"):
                st.session_state.user_intention = {}  # save llm intention recognition result
                st.session_state.user_recognized_question = []  # save llm recognized questions
                st.session_state.vector_searched_documents = []
                st.session_state.keyword_searched_documents = []
                st.session_state.context = ""  # formatted context for AI response

                # Intention recognition
                # with st.spinner("问题意图识别中..."):
                #     print("Start to recognize user intention...")
                #     time1 = time.time()
                #
                #     intention_response = llm_client.intention_recognition(user_input)  # "{"1": {"keywords": [], "coherent_sentence": "xxx"}, "2": {}}"
                #     intention_response_str = intention_response.content
                #     intention_response_token_usage = intention_response.token_usage
                #     st.session_state.user_intention = json.loads(intention_response_str)
                #     time2 = time.time()
                #     print(f"Intention recognition time: {time2 - time1}\n")
                #     chatbot_container.caption(f"Intention Recognition Token Usage: {intention_response_token_usage} | time cost: {time2 - time1}")
                #
                #     with retriever_container:
                #         st.markdown("#### 问题意图识别结果")
                #         for no, intention in st.session_state.user_intention.items():
                #             keywords = intention.get("keywords", [])
                #             coherent_sentence = intention.get("coherent_sentence", "")
                #             if coherent_sentence:
                #                 st.session_state.user_recognized_question.append(coherent_sentence)
                #             st.markdown(f"**分析{no}**")
                #             keywords_str = ", ".join(keywords)
                #             st.markdown(f":orange[**{coherent_sentence}** ***(关键词: {keywords_str})***]")
                #         st.divider()

                # Retrieve Documents

                with st.spinner("文档检索中..."):
                    time1 = time.time()
                    qdrant_client.checkout_collection(QDRANT_COLLECTION_INTFLEX)
                    # for user_question in st.session_state.user_recognized_question:
                    #     embedding_response = llm_client.get_text_embedding(text=user_question)
                    #     embedding_user_question = embedding_response.content
                    #     embedding_response_token_usage = embedding_response.token_usage
                    #     retrieved_payloads = qdrant_client.retrieve_similar_vectors_simply(
                    #         query_vector=embedding_user_question,
                    #         top_k=top_k,
                    #         sim_lower_bound=sim_threshold,
                    #     )
                    embedding_response = llm_client.get_text_embedding(text=user_input)
                    embedding_user_question = embedding_response.content
                    embedding_response_token_usage = embedding_response.token_usage
                    time2 = time.time()
                    print(f"Embedding time cost: {time2 - time1}")
                    chatbot_container.caption(f"Embedding Token Usage: {embedding_response_token_usage} | time cost: {time2 - time1}")

                    # Vector Search
                    print("Start to vector search...")
                    time1 = time.time()
                    vector_search_payloads = qdrant_client.retrieve_similar_vectors_simply(
                        query_vector=embedding_user_question,
                        top_k=top_k,
                        sim_lower_bound=sim_threshold,
                    )

                    # Get original content from MongoDB (base on chunk_id and doc_name)
                    for vector_search_payload in vector_search_payloads:
                        doc_id = vector_search_payload["doc_id"]
                        chunk_id = vector_search_payload["chunk_id"]

                        pre_chunk = mongo_collection.find_one({"doc_id": doc_id, "chunk_id": chunk_id - 1})
                        pre_content = pre_chunk.get("content", "") if pre_chunk else ""

                        chunk = mongo_collection.find_one({"doc_id": doc_id, "chunk_id": chunk_id})
                        content = chunk.get("content", "") if chunk else ""

                        next_chunk = mongo_collection.find_one({"doc_id": doc_id, "chunk_id": chunk_id + 1})
                        next_content = next_chunk.get("content", "") if next_chunk else ""

                        vector_search_payload["content"] = content
                        vector_search_payload["pre_content"] = remove_overlap(pre_content, content)
                        vector_search_payload["next_content"] = remove_overlap(next_content, content)

                    st.session_state.vector_searched_documents.extend(vector_search_payloads)

                    # Sort retrieved documents result by score [from Vector Search]
                    st.session_state.vector_searched_documents = st.session_state.vector_searched_documents[:top_k]
                    st.session_state.vector_searched_documents = sorted(st.session_state.vector_searched_documents, key=lambda x: x["score"], reverse=True)
                    for doc in st.session_state.vector_searched_documents:
                        doc_name = doc["doc_name"]
                        content = doc.get("content", "")
                        pre_content = doc.get("pre_content", "")
                        next_content = doc.get("next_content", "")
                        st.session_state.context += f"[文本来源]: {doc_name}\n[正文]:{pre_content}{content}{next_content}\n<DIVIDER>"

                    time2 = time.time()
                    print(f"Vector search time cost: {time2 - time1}")

                    # Keyword Search
                    print("Start to keyword search based on BM25...")

                    time1 = time.time()
                    keyword_search_results = searcher.search(user_input, top_k)
                    # Get original content from MongoDB (base on chunk_id and doc_name)
                    for keyword_search_result in keyword_search_results:
                        doc_id = keyword_search_result["doc_id"]
                        chunk_id = keyword_search_result["chunk_id"]
                        doc_name = mongo_collection.find_one({"doc_id": doc_id, "chunk_id": chunk_id}).get("doc_name", "")

                        pre_chunk = mongo_collection.find_one({"doc_id": doc_id, "chunk_id": chunk_id - 1})
                        pre_content = pre_chunk.get("content", "") if pre_chunk else ""

                        chunk = mongo_collection.find_one({"doc_id": doc_id, "chunk_id": chunk_id})
                        content = chunk.get("content", "") if chunk else ""

                        next_chunk = mongo_collection.find_one({"doc_id": doc_id, "chunk_id": chunk_id + 1})
                        next_content = next_chunk.get("content", "") if next_chunk else ""

                        keyword_search_result["doc_name"] = doc_name
                        keyword_search_result["content"] = content
                        keyword_search_result["pre_content"] = remove_overlap(pre_content, content)
                        keyword_search_result["next_content"] = remove_overlap(next_content, content)

                    st.session_state.keyword_searched_documents.extend(keyword_search_results)

                    # Display result
                    with retriever_container:
                        st.markdown("#### BM25检索结果")
                        for no, doc in enumerate(st.session_state.keyword_searched_documents):
                            chunk_id = doc["chunk_id"]
                            doc_name = doc["doc_name"]
                            content = doc.get("content", "").replace("\n", "")
                            pre_content = doc.get("pre_content", "").replace("\n", "")
                            next_content = doc.get("next_content", "").replace("\n", "")
                            score = doc["score"]
                            # Display
                            st.markdown(f"**来源{no + 1}**: **{doc_name}** ***(chunk_id:{chunk_id})***")
                            st.markdown(f":orange[{pre_content}]:green{content}:orange[{next_content}]")
                            st.markdown(f":red[相似度={score}]")
                            st.divider()

                        st.markdown("#### 向量检索结果")
                        for no, doc in enumerate(st.session_state.vector_searched_documents):
                            chunk_id = doc["chunk_id"]
                            doc_name = doc["doc_name"]
                            content = doc.get("content", "").replace("\n", "")
                            pre_content = doc.get("pre_content", "").replace("\n", "")
                            next_content = doc.get("next_content", "").replace("\n", "")
                            score = doc["score"]
                            # Display
                            st.markdown(f"**来源{no + 1}**: **{doc_name}** ***(chunk_id:{chunk_id})***")
                            st.markdown(f":orange[{pre_content}]:green{content} :orange[{next_content}]")
                            st.markdown(f":red[相似度={score}]")
                            st.divider()

                        time2 = time.time()
                        print(f"Keyword search time cost: {time2 - time1}")

                # Generate AI Response

                with st.spinner("AI思考中..."):
                    # Get chat history from st.session.state
                    chat_history = st.session_state.messages
                    if len(chat_history) > chat_history_len:
                        chat_history = chat_history[-chat_history_len:]

                    print("Start to generate AI response...")
                    time1 = time.time()
                    if not is_stream:
                        ai_response = llm_client.get_chat_response(
                            user_question=user_input,
                            context=st.session_state.context,
                            chat_history=chat_history,  # [{"role": "assistant", "content": "xxx"}, {"role": "user", "content": "xxx"}...]
                            temperature=temperature,
                            model_name=chat_model_type
                        )
                        ai_response_token_usage = ai_response.token_usage
                        st.markdown(ai_response.content)
                    else:
                        response_generator = llm_client.stream_chat_response(
                            user_question=user_input,
                            context=st.session_state.context,
                            chat_history=chat_history,
                            temperature=temperature,
                            model_name=chat_model_type
                        )
                        ai_response = st.write_stream(response_generator)
                        ai_response_token_usage = llm_client.stream_chat_token_usage
                    time2 = time.time()
                    print(f"Generate AI response time: {time2 - time1}\n")
                    chatbot_container.caption(f"Chat Response Token Usage: {ai_response_token_usage} | time cost: {time2 - time1}")

            # Add to messages session
            st.session_state.messages.append({"role": "user", "content": user_input})
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

