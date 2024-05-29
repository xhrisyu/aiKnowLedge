"""
题目生成界面
1. 输入文本段落
2. 点击生成题目
3. 展示题目、选项和答案（选择题）
4. 选择保存或丢弃
"""
import streamlit as st
import time
import pandas as pd


def quiz_generator_page():
    text = st.text_area(label="文本", height=200, placeholder="请输入文本段落")
    generate_btn = st.button("生成题目")

    if generate_btn:
        with st.spinner("正在生成题目"):
            time.sleep(1.0)
            st.session_state["generate_status"] = True

    if "generate_status" in st.session_state and st.session_state["generate_status"]:
        question = "这是一个问题, 请回答, 选项 A, B, C, D, 选择一个回答"
        options = {
            "A": "选项A阿斯顿发送到发送到发送地方",
            "B": "选项Bi哦破IPO皮哦皮肤破的四方坡阿斯顿发i",
            "C": "选项C阿斯顿发送地表现出v看不见火车离开v吧",
            "D": "选项D列五头破ID发送破IPv富家地方v",
        }
        answer = "A"

        one_quiz_title = st.text(f"问题: {question}")
        one_quiz = st.radio(
            label=f"\\huge 问题: {question}",
            options=[f"{key}. {value}" for key, value in options.items()],
            index=list(options.keys()).index(answer),
            disabled=True
        )



