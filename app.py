import os

import streamlit as st
from streamlit_option_menu import option_menu
import hmac

from views.chatbot import chatbot
from views.knowledge_base import kb_management
from views.quiz_generator import quiz_generator

VERSION = "1.0.0"


def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


# Page Setting
st.set_page_config(
    page_title="aiKnowLedge",
    initial_sidebar_state="expanded",
    page_icon="🤖",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/yyyzzx7/aiKnowLedge.git',
        'Report a bug': "https://github.com/yyyzzx7/aiKnowLedge/issues",
        'About': f"""欢迎使用 aiKnowLedge {VERSION}！"""
    }
)

with st.sidebar:
    if os.path.exists(os.path.join("img", "aiknow_logo_transparent.png")):
        st.image(os.path.join("img", "aiknow_logo_transparent.png"))
        st.caption(
            f"""<p align="right">当前版本：{VERSION}</p>""",
            unsafe_allow_html=True,
        )

    selected_item = option_menu(
        menu_title="",
        # options=["主页", "问答", "上传", "管理"],
        options=["主页", "问答助手", "知识管理", "习题生成"],
        icons=["house", "robot", "cloud-upload", "clipboard-data", "box"],
        menu_icon="cast",
        default_index=0,
    )

if not check_password():
    st.stop()

if selected_item == "主页":
    st.title("Welcome to aiKnowLedge🤓")

    # Load local README.md file
    en_readme, cn_readme = "", ""
    en_readme_path = os.path.join("README_EN.md")
    if os.path.exists(en_readme_path):
        with open(en_readme_path, "r", encoding="utf-8") as f:
            en_readme = f.read()

    cn_readme_path = os.path.join("README_CN.md")
    if os.path.exists(cn_readme_path):
        with open(cn_readme_path, "r", encoding="utf-8") as f:
            cn_readme = f.read()

    tab1, tab2 = st.tabs(["English", "中文"])
    if en_readme:
        tab1.markdown(en_readme)
    if cn_readme:
        tab2.markdown(cn_readme)

    # st.write("""
    # # Under developing...🤓
    # """)
    # st.caption('This is a string that explains something above.')
    # st.caption('A caption with _italics_ :blue[colors] and emojis :sunglasses:')
    # with st.echo():
    #     st.write('This code will be printed')

if selected_item == "问答助手":
    chatbot.chatbot_page()

if selected_item == "知识管理":
    kb_management.kb_management_page()

if selected_item == "习题生成":
    quiz_generator.quiz_generator_page()
