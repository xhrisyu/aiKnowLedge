"""
Chatbot Page

$ Retrieving documents use qdrant client directly, not through FastAPI server [http request is slow]
$ QA request use llm client directly, not through FastAPI server [streaming output is hard to implement]
"""
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
            sim_threshold = st.slider(label="Similarity Threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
            additional_context_length = st.slider(label="Additional Context Length", min_value=0, max_value=500, value=50, step=10)
        with st.expander("⚙️ Generator", expanded=True):
            chat_model_type = st.selectbox(label="Chat Model", options=["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o", ])
            temperature = st.slider(label="Temperature", min_value=0.0, max_value=2.0, value=0.8, step=0.1)
            is_stream = st.toggle(label="Stream", value=False)

    # Main Area: Chatbot & Retriever Panel
    col1, gap, col2 = st.columns([3, 0.01, 2])

    # Area 1(Left): Chatbot
    with col1.container(border=False, height=550):
        # Chat input
        with st.container(border=False):
            float_init(theme=True, include_unstable_primary=False)
            user_input = st.chat_input("请输入问题...")
            button_css = float_css_helper(width="2.2rem", bottom="3rem", transition=0)
            float_parent(css=button_css)

        # Init chat message
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "这里是Ledge，有什么可以帮您？"}]

        # Display chat session message
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if user_input:
            # User Message
            with st.chat_message("user"):
                st.markdown(user_input)

            # AI Message
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    llm_client = get_llm_client()
                    qdrant_client = get_qdrant_client()

                    # Intention recognition


                    # Retrieve Documents
                    print("Start to retrieve documents...")
                    time1 = time.time()
                    embedded_user_question = llm_client.get_text_embedding(text=user_input)
                    qdrant_client.checkout_collection(QDRANT_COLLECTION_DEFAULT_NAME)
                    retrieved_payloads = qdrant_client.retrieve_similar_vectors(
                        query_vector=embedded_user_question,
                        top_k=top_k,
                        sim_lower_bound=sim_threshold,
                    )  # [{"chunk_id": 1, "document_name": "xxx", "page_content": "xxx", "score": 0.72}, ...]
                    st.session_state.retrieved_docs = retrieved_payloads
                    context = ""
                    for doc in st.session_state.retrieved_docs:
                        document_name = doc['document_name']
                        page_content = doc['page_content']
                        context += f"相关文本来源: {document_name}\n相关文本内容:\n{page_content}\n\n"
                    time2 = time.time()
                    print(f"Retrieve documents time: {time2 - time1}")

                    # Get chat history from st.session.state (exclude the first message and newly added user message)
                    # chat_history = [message for message in st.session_state.messages][1:-1]
                    # if not chat_history:
                    #     print("chat history is empty")
                    #     chat_history_text = ""
                    # else:
                    #     if len(chat_history) > 6:
                    #         chat_history = chat_history[-6:]
                    #     chat_history_text = get_chat_history_text(chat_history)
                    # chat_history_text += f"用户: {user_input}\n"

                    # Generate AI Response
                    print("Start to generate AI response...")
                    if not is_stream:
                        ai_response = llm_client.get_chat_response(
                            user_question=user_input,
                            context=context,
                            chat_history="",
                            temperature=temperature,
                            model_name=chat_model_type
                        )
                        st.markdown(ai_response)
                    else:
                        response_generator = llm_client.stream_chat_response(
                            user_question=user_input,
                            context=context,
                            chat_history="",
                            temperature=temperature,
                            model_name=chat_model_type
                        )
                        ai_response = st.write_stream(response_generator)
                    time3 = time.time()
                    print(f"Generate AI response time: {time3 - time2}")

            # Add to messages session
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "ai", "content": ai_response})

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

    # Area 2(Right): Retriever Panel
    with col2.container(border=True, height=700):
        st.write("召回文档片段")
        if "retrieved_docs" in st.session_state:
            for no, doc in enumerate(st.session_state.retrieved_docs):
                chunk_id = doc["chunk_id"]
                document_name = doc["document_name"]
                page_content = doc["page_content"]
                score = doc["score"]
                st.subheader(f"召回文档 {no + 1}")
                st.write(f"文档名称: {document_name}")
                st.write(f"片段号: {chunk_id}")
                st.write(f"文档片段: {page_content}")
                st.write(f"匹配得分: {score}")
                st.divider()

