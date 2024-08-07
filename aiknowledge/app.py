import os
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import hmac

from aiknowledge.webui.chatbot import chatbot_page


VERSION = "1.0.0"


def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        cols = st.columns([1, 4, 1])
        with cols[1].form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't knowledge_base the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


# Page Setting
st.set_page_config(
    page_title="则成雨林",
    initial_sidebar_state="expanded",
    page_icon="🤖",
    layout="wide",
    menu_items={
        # 'Get Help': 'https://github.com/yyyzzx7/aiKnowLedge.git',
        # 'Report a bug': "https://github.com/yyyzzx7/aiKnowLedge/issues",
        'About': f"""欢迎使用 aiKnowLedge {VERSION}！"""
    }
)

# Sidebar Setting
with st.sidebar:
    if os.path.exists(os.path.join("webui/src/logo", "aiknow_logo_transparent_2.png")):
        st.image(os.path.join("webui/src/logo", "aiknow_logo_transparent_2.png"))
        st.caption(
            f"""<p align="right">Version: {VERSION}</p>""",
            unsafe_allow_html=True,
        )

    selected_item = option_menu(
        menu_title="",
        # options=["主页", "问答助手", "知识管理", "习题生成"],
        # icons=["house", "robot", "cloud-upload", "clipboard-data", "box"],
        options=["问答助手", "文件列表"],
        icons=["robot", "cloud-upload"],
        menu_icon="cast",
        default_index=0,
    )

if not check_password():
    st.stop()

# if selected_item == "主页":
#     st.title("Welcome to aiKnowLedge🤓")
#
#     # Load local README.md file
#     en_readme, cn_readme = "", ""
#
#     cn_readme_path = os.path.join("README.md")
#     if os.path.exists(cn_readme_path):
#         with open(cn_readme_path, "r", encoding="utf-8") as f:
#             cn_readme = f.read()
#
#     en_readme_path = os.path.join("README_EN.md")
#     if os.path.exists(en_readme_path):
#         with open(en_readme_path, "r", encoding="utf-8") as f:
#             en_readme = f.read()
#
#     tab1, tab2 = st.tabs(["中文", "English"])
#     if cn_readme:
#         tab1.markdown(cn_readme)
#     if en_readme:
#         tab2.markdown(en_readme)

if selected_item == "问答助手":
    chatbot_page()

if selected_item == "文件列表":
    file_list_path = os.path.join("aiknowledge/webui/src/", "file_list.txt")
    file_list = []
    with open(file_list_path, "r", encoding="utf-8") as f:
        for line in f:
            file_list.append(line.strip())
    st.table(pd.DataFrame(file_list, columns=["文件列表"]))


# if selected_item == "使用手册":
#     test_instruction_path = os.path.join("aiknowledge/webui/src/test_instruction", "test_instruction.md")
#     if os.path.exists(test_instruction_path):
#         with open(test_instruction_path, "r", encoding="utf-8") as f:
#             test_instruction = f.read()
#         st.markdown(test_instruction)
#     print(test_instruction)

# if selected_item == "知识管理":
#     kb_management_page()
#
# if selected_item == "习题生成":
#     quiz_generator_page()
