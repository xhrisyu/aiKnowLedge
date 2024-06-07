"""
Chatbot Page

$ Retrieving documents use qdrant client directly, not through FastAPI server [http request is slow]
$ QA request use llm client directly, not through FastAPI server [streaming output is hard to implement]
"""
import json
import time
import streamlit as st
from streamlit_float import float_init, float_css_helper, float_parent

from config import app_config
from views.constants import *
from llm import OpenAILLM
from db import QAQdrantClient


@st.cache_resource
def get_llm_client():
    openai_config = app_config.get("openai")
    return OpenAILLM(api_key=openai_config["api_key"])


@st.cache_resource
def get_qdrant_client():
    qdrant_config = app_config.get("qdrant")
    return QAQdrantClient(
        url=qdrant_config["url"],
        collection_name=qdrant_config['collection_name']["general"],
        embedding_dim=qdrant_config["embedding_dim"]
    )


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
            st.session_state.messages = [{"role": "assistant", "content": "这里是Ledge，你的知识库问答助手，想问些什么？"}]

        # Display chat session message
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if user_input:
            llm_client = get_llm_client()
            qdrant_client = get_qdrant_client()

            # User Message
            with st.chat_message("user"):
                st.markdown(user_input)

            # AI Message
            with st.chat_message("assistant"):
                st.session_state.user_intention = {}  # save llm intention recognition result
                st.session_state.user_recognized_question = []  # save llm recognized questions
                st.session_state.retrieved_docs = []  # save retrieved document from vecdb
                st.session_state.context = ""  # formatted context for AI response

                # Intention recognition
                with st.spinner("问题意图识别中..."):
                    print("Start to recognize user intention...")
                    time1 = time.time()
                    intention_response = llm_client.intention_recognition(user_input)  # "{"1": {"keywords": [], "coherent_sentence": "xxx"}, "2": {}}"
                    intention_response_str = intention_response.content
                    intention_response_token_usage = intention_response.token_usage
                    st.session_state.user_intention = json.loads(intention_response_str)
                    time2 = time.time()
                    print(f"Intention recognition time: {time2 - time1}\n")
                    chatbot_container.caption(f"Intention Recognition Token Usage: {intention_response_token_usage} | time cost: {time2 - time1}")

                    with retriever_container:
                        st.markdown("#### 问题意图识别结果")
                        for no, intention in st.session_state.user_intention.items():
                            keywords = intention.get("keywords", [])
                            coherent_sentence = intention.get("coherent_sentence", "")
                            if coherent_sentence:
                                st.session_state.user_recognized_question.append(coherent_sentence)
                            st.markdown(f"**分析{no}**")
                            keywords_str = ", ".join(keywords)
                            st.markdown(f":orange[**{coherent_sentence}** ***(关键词: {keywords_str})***]")
                        st.divider()

                # Retrieve Documents
                with st.spinner("文档检索中..."):
                    print("Start to retrieve documents...")
                    time1 = time.time()
                    # Iterate each recognized question
                    qdrant_client.checkout_collection(QDRANT_COLLECTION_DEFAULT_NAME)
                    for user_question in st.session_state.user_recognized_question:
                        embedding_response = llm_client.get_text_embedding(text=user_question)
                        embedding_user_question = embedding_response.content
                        embedding_response_token_usage = embedding_response.token_usage
                        retrieved_payloads = qdrant_client.retrieve_similar_vectors_with_adjacent_context(
                            query_vector=embedding_user_question,
                            top_k=top_k,
                            sim_lower_bound=sim_threshold,
                            adjacent_len=additional_context_length
                        )
                        st.session_state.retrieved_docs.extend(retrieved_payloads)
                    time2 = time.time()
                    print(f"Embedding time cost: {time2 - time1}")
                    chatbot_container.caption(f"Embedding Token Usage: {embedding_response_token_usage} | time cost: {time2 - time1}")

                    # Sort retrieved documents result by score
                    time1 = time.time()
                    st.session_state.retrieved_docs = st.session_state.retrieved_docs[:top_k]
                    st.session_state.retrieved_docs = sorted(st.session_state.retrieved_docs, key=lambda x: x["score"], reverse=True)
                    for doc in st.session_state.retrieved_docs:
                        document_name = doc['document_name']
                        page_content = doc.get('page_content', "")
                        pre_page_content = doc.get('pre_page_content', "")
                        next_page_content = doc.get('next_page_content', "")
                        st.session_state.context += f"[文本来源]: {document_name}\n[正文]:{pre_page_content}{page_content}{next_page_content}\n<DIVIDER>"
                    time2 = time.time()
                    print(f"Retrieve documents time: {time2 - time1}")

                    with retriever_container:
                        st.markdown("#### 文档检索结果")
                        for no, doc in enumerate(st.session_state.retrieved_docs):
                            chunk_id = doc["chunk_id"]
                            page = doc["page"]
                            document_name = doc["document_name"]
                            page_content = doc.get("page_content", "")
                            pre_page_content = doc.get('pre_page_content', "")
                            next_page_content = doc.get('next_page_content', "")
                            score = doc["score"]
                            # Display
                            st.markdown(f"**来源{no + 1}**: **{document_name}** ***(page:{page}, chunk_id:{chunk_id})***")
                            st.markdown(f":orange[{pre_page_content}]{page_content}:orange[{next_page_content}]")
                            st.markdown(f":red[相似度={score}]")
                            st.divider()

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



