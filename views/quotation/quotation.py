import streamlit as st


def quiz_generator_page():
    with st.sidebar:
        with st.expander("⚙️ Question", expanded=True):
            quiz_type = st.selectbox(label="Quiz Type", options=["单选题", "多选题", "填空题", "问答题"], index=0,
                                     disabled=True)
            quiz_num = st.number_input(label="Quiz Number", min_value=1, max_value=5, value=1, step=1)
        with st.expander("⚙️ Generator", expanded=True):
            model_type = st.selectbox(label="Generate Model", options=["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o", ])
            temperature = st.slider(label="Temperature", min_value=0.0, max_value=2.0, value=0.6, step=0.1)
            # is_stream = st.toggle(label="Stream", value=False)
