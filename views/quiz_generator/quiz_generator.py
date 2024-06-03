"""
题目生成界面
1. 输入文本段落, 输入题目数量、题目类型
2. 输入生成模型的参数[temperature, modelName]
2. 点击生成题目
3. 展示题目、选项和答案（选择题）
4. 选择保存或丢弃
"""
import streamlit as st
import time
import pandas as pd

from config import app_config
from llm.generator import QuizGenerator


def quiz_generator_page():
    with st.sidebar:
        with st.expander("⚙️ Question", expanded=True):
            quiz_type = st.selectbox(label="Quiz Type", options=["单选题", "多选题", "填空题", "问答题"], index=0, disabled=True)
            quiz_num = st.number_input(label="Quiz Number", min_value=1, max_value=5, value=1, step=1)
        with st.expander("⚙️ Generator", expanded=True):
            model_type = st.selectbox(label="Generate Model", options=["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o", ])
            temperature = st.slider(label="Temperature", min_value=0.0, max_value=2.0, value=0.8, step=0.1)
            # is_stream = st.toggle(label="Stream", value=False)

    for_test_context = """7.1.6 组织的知识
组织应确定必要的知识，以运行过程，并获得合格产品和服务。
这些知识应予以保持，并能在所需的范围内得到。
为应对不断变化的需求和发展趋势，组织应审视现有的知识，确定如何获取或接触更多必要的知识和知识更新。
注1:组织的知识是组织特有的知识，通常从其经验中获得，是为实现组织目标所使用和共享的信息。
注2:组织的知识可基于：
a)内部来源(如知识产权、从经验获得的知识、从失败和成功项目汲取的经验和教训、获取和分享未成文的知识和经验，以及过程、产品和服务的改进结果);
b)外部来源(如标准、学术交流、专业会议、从顾客或外部供方收集的知识)。"""

    context = st.text_area(label="文本", value=for_test_context, placeholder="请输入文本段落", height=300)

    if st.button("生成题目"):
        generator = QuizGenerator(
            api_key=app_config.get("openai")["api_key"],
            context=context,
            num=quiz_num,
            model_name=model_type,
            temperature=temperature
        )
        with st.spinner("正在生成知识点和题目..."):
            knowledge_points = generator.generate_knowledge_point()  # ['知识点1', '知识点2', ...]
            questions = generator.generate_question(knowledge_point=knowledge_points)

            precise_num = min(len(knowledge_points), len(questions))
            for i in range(precise_num):
                cur_knowledge_point = knowledge_points[i]
                cur_question = questions[i]
                with st.container(border=True):
                    st.subheader(f"知识点{i + 1}")
                    st.markdown(f"**{cur_knowledge_point}**")
                    st.markdown(f"***问题***")
                    st.radio(
                        label=f"{cur_question['question']}",
                        options=[f"{key}. {value}" for key, value in cur_question['options'].items()],
                        index=list(cur_question['options'].keys()).index(cur_question['answer']),
                    )


