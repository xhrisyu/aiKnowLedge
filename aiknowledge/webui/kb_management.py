import os
import time
from typing import Dict, Tuple, Literal, List
import pandas as pd
from datetime import datetime
import requests
import streamlit as st
from streamlit_tags import st_tags
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder

from aiknowledge.utils.tools import file_exist, convert_numpy_types, get_file_extension
from aiknowledge.backend.api_paths import APIPaths
from aiknowledge.webui.constants import MONGO_DATABASE_NAME, MONGO_COLLECTION_DEFAULT_NAME, QDRANT_COLLECTION_DEFAULT_NAME, SUPPORTED_EXTS

TF_cell_renderer = JsCode("""function(params) {if(params.value==true){return '✓'}else{return '×'}}""")
list_cell_renderer = JsCode("""
function(params) {
    if (Array.isArray(params.value)) {
        return params.value.join(' | ');
    } else {
        return '';
    }
}""")


def config_aggrid(
        df: pd.DataFrame,
        columns: Dict[Tuple[str, str], Dict],
        selection_mode: Literal["single", "multiple", "disabled"] = "single",
        use_checkbox: bool = False,
        pre_selected_rows: List[int] = None
) -> GridOptionsBuilder:
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column("No", width=40)
    for (col, header), kw in columns.items():
        gb.configure_column(col, header, wrapHeaderText=True, **kw)
    gb.configure_selection(
        selection_mode=selection_mode,
        use_checkbox=use_checkbox,
        pre_selected_rows=pre_selected_rows,
    )
    gb.configure_pagination(
        enabled=True,
        paginationAutoPageSize=False,
        paginationPageSize=10
    )
    return gb


def kb_management_page():
    # Get store data from mongo DB
    response = requests.get(
        url=APIPaths.get_full_path(APIPaths.KB_GET),
        params={"database_name": MONGO_DATABASE_NAME, "collection_name": MONGO_COLLECTION_DEFAULT_NAME}
    )
    if response.status_code == 200:
        data = response.json()
    else:
        st.error(f"获取知识数据库错误. 错误码: {response.status_code}")
        return

    # Display the KB data
    kb_dataframe = pd.DataFrame(data)
    if not len(kb_dataframe):
        st.info(f"知识库 `{MONGO_COLLECTION_DEFAULT_NAME}` 中暂无文件")
    else:
        st.write(f"知识库 `{MONGO_COLLECTION_DEFAULT_NAME}` 中已有文件:")
        # process display data
        kb_dataframe_display = kb_dataframe.drop(columns=['_id'])
        kb_dataframe_display_column_order = [
            "file_name", "file_extension", "chunk_size", "overlap_size", "separators", "in_vector_db", "location", "create_time"
        ]
        kb_dataframe_display = kb_dataframe_display[kb_dataframe_display_column_order]
        gb = config_aggrid(
            df=kb_dataframe_display,
            columns={
                ("file_name", "文档名称"): {},
                ("file_extension", "文件类型"): {},
                ("chunk_size", "分块长度"): {},
                ("overlap_size", "重合长度"): {},
                ("separators", "分隔符"): dict(cellRenderer=list_cell_renderer, headerName="分隔符"),
                ("in_vector_db", "是否在向量库"): dict(cellRenderer=TF_cell_renderer, headerName="是否在向量库"),
                ("create_time", "创建时间"): {},
                ("location", "文件位置"): {},
            },
            selection_mode="single",
            use_checkbox=True,
            pre_selected_rows=st.session_state.get("selected_rows", [0])
        )
        kb_grid = AgGrid(
            kb_dataframe_display,
            gb.build(),
            columns_auto_size_mode="FIT_CONTENTS",
            theme="alpine",
            custom_css={"#gridToolBar": {"display": "none"}},
            allow_unsafe_jscode=True,
            enable_enterprise_modules=False,
        )
        selected_rows = kb_grid.get("selected_rows", [])  # 'single' mode returns only one row

        cols = st.columns(4)
        if selected_rows is None or selected_rows.empty:  # Placeholder for not selected rows
            cols[0].download_button("下载选中文档", data="", file_name="", disabled=True, use_container_width=True, key="temp_download")
            cols[1].button("添加至向量库", disabled=True, use_container_width=True, key="temp_add_to_vec")
            cols[2].button("从向量库删除", disabled=True, use_container_width=True, key="temp_remove_from_vec")
            cols[3].button("从知识库中删除", type="primary", disabled=True, use_container_width=True, key="temp_remove_from_kb")
        if selected_rows is not None and not selected_rows.empty:
            row, row_id = selected_rows.iloc[0], int(selected_rows.index[0])
            doc_id = kb_dataframe.iloc[row_id]["_id"]  # find `_id` in kb_dataframe by selected row index
            in_local_disk = file_exist(row["location"])  # check if file exists in local disk

            # Column 1: Download selected file
            if not in_local_disk:
                cols[0].download_button(
                    label="下载选中文档",
                    data="",
                    file_name=row["file_name"],
                    use_container_width=True,
                    disabled=True
                )
            else:
                with open(row["location"], "rb") as fp:
                    cols[0].download_button(
                        label="下载选中文档",
                        data=fp,
                        file_name=row["file_name"],
                        use_container_width=True,
                        disabled=False
                    )

            # Column 2: Add selected file to Vector DB
            if cols[1].button(
                    label="添加至向量库",
                    disabled=row['in_vector_db'],
                    use_container_width=True,
            ):
                params_data = {
                    "doc_id": str(doc_id),
                    "file_path": convert_numpy_types(row["location"]),
                    "chunk_size": convert_numpy_types(row["chunk_size"]),
                    "overlap_size": convert_numpy_types(row["overlap_size"]),
                    "separators": convert_numpy_types(row["separators"]),  # escaped[转义] characters, '\\n' etc.
                }
                with st.spinner('正在添加到向量数据库中...'):
                    print(f"params data fro VEC_INSERT:\n{params_data}")
                    # Add file content to Vector DB (Qdrant)
                    response1 = requests.post(
                        url=APIPaths.get_full_path(APIPaths.VEC_INSERT),
                        json={
                            "vecdb_collection_name": QDRANT_COLLECTION_DEFAULT_NAME,
                            "data": params_data
                        }
                    )
                    # Update `in_vector_db` field in MongoDB
                    if response1.status_code == 200:
                        response2 = requests.post(
                            url=APIPaths.get_full_path(APIPaths.KB_UPDATE),
                            json={
                                "database_name": MONGO_DATABASE_NAME,
                                "collection_name": MONGO_COLLECTION_DEFAULT_NAME,
                                "doc_id": doc_id,
                                "update_dict": {"in_vector_db": True}
                            }
                        )
                        if response2.status_code == 200:
                            st.success("文件已添加至向量库", icon='🎉')
                            time.sleep(1.0)
                            st.rerun()
                        else:
                            st.error(f"更新文件在知识库中的状态时发生错误. 状态码: {response2.status_code}")
                            return
                    else:
                        st.error(f"添加文件至向量库时发生错误. 状态码: {response1.status_code}")
                        return

            # Column 3: Remove selected file from Vector DB (not remove from KB)
            if cols[2].button(
                    label="从向量库删除",
                    disabled=not row['in_vector_db'],
                    use_container_width=True,
            ):
                with st.spinner('正在从向量数据库中删除...'):
                    response1 = requests.post(
                        url=APIPaths.get_full_path(APIPaths.VEC_REMOVE),
                        json={"vecdb_collection_name": QDRANT_COLLECTION_DEFAULT_NAME, "doc_ids": str(doc_id)}
                    )
                    if response1.status_code == 200:
                        response2 = requests.post(
                            url=APIPaths.get_full_path(APIPaths.KB_UPDATE),
                            json={
                                "database_name": MONGO_DATABASE_NAME,
                                "collection_name": MONGO_COLLECTION_DEFAULT_NAME,
                                "doc_id": doc_id,
                                "update_dict": {"in_vector_db": False}
                            }
                        )
                        if response2.status_code == 200:
                            st.success("文件已从向量库中删除", icon='🎉')
                            time.sleep(1.0)
                            st.rerun()
                        else:
                            st.error(f"更新文件在知识库中的状态时发生错误. 状态码: {response2.status_code}")
                            return
                    else:
                        st.error(f"从向量库中删除文件时发生错误. 状态码: {response1.status_code}")
                        return

            # Column 4: Remove selected file both from KB and Vector DB
            if cols[3].button(
                    label="从知识库中删除",
                    type="primary",
                    use_container_width=True,
            ):
                with st.spinner('正在从知识库中删除...'):
                    # Remove file from Vector DB (Qdrant)
                    response1 = requests.post(
                        url=APIPaths.get_full_path(APIPaths.VEC_REMOVE),
                        json={"vecdb_collection_name": QDRANT_COLLECTION_DEFAULT_NAME, "doc_ids": str(doc_id)}
                    )
                    # Remove file from MongoDB
                    response2 = requests.post(
                        url=APIPaths.get_full_path(APIPaths.KB_REMOVE),
                        json={
                            "database_name": MONGO_DATABASE_NAME,
                            "collection_name": MONGO_COLLECTION_DEFAULT_NAME,
                            "doc_id": doc_id
                        }
                    )
                    # Remove file from local disk
                    remove_local_file = False
                    if in_local_disk:
                        # Check other inserted rag file have the same file location
                        if len(kb_dataframe[kb_dataframe["location"] == row["location"]]) == 1:
                            os.remove(row["location"])
                            remove_local_file = True

                    if response1.status_code == 200 and response2.status_code == 200 and remove_local_file:
                        st.success("文件已从知识库中删除", icon='🎉')
                    else:
                        st.error(f"从知识库中删除文件时发生错误. 状态码: {response1.status_code}")
                    st.rerun()

    """
    Multiple file upload - under developing
    """
    # Single file upload
    with st.form("my-form", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            label="上传知识文件",
            type=SUPPORTED_EXTS,
            accept_multiple_files=False,
        )
        # Document Preprocess parameters
        with st.expander("文本处理参数", expanded=True):
            cols = st.columns([0.4, 0.4, 0.2])
            chunk_size = cols[0].slider("分块长度", min_value=0, max_value=1000, value=200, step=10)
            overlap_size = cols[1].slider("分块重合长度", min_value=0, max_value=500, value=20, step=5)
            separators = st_tags(
                label='分隔符',
                text='按`回车`添加',
                value=['\\n\\n', '\\n', '。'],
                suggestions=[],
                maxtags=6,
                key='1')

        # Add uploaded file to KnowledgeBase
        submitted = st.form_submit_button("添加文件到知识库")
        if submitted:
            # Firstly check if file is uploaded
            if uploaded_file is None:
                st.error("请先上传文件")
                return
            try:
                # Read file and get metadata
                uploaded_file_bytes = uploaded_file.getvalue()  # read file as bytes
                uploaded_file_name = uploaded_file.name
                file_extension = get_file_extension(uploaded_file_name, upper_case=True, with_dot=False)

                # Create local directories
                local_doc_dir = os.path.join(os.getcwd(), "uploaded_file")  # os.getcwd()当前工程绝对路径
                local_uploaded_file_dir = os.path.join(local_doc_dir, "rag")
                if not os.path.exists(local_doc_dir):
                    os.makedirs(local_doc_dir)
                if not os.path.exists(local_uploaded_file_dir):
                    os.makedirs(local_uploaded_file_dir)
            except Exception as e:
                st.error(f"文件处理时发生错误: {e}")
                return

            # Add file metadata to MongoDB
            params_data = {
                "file_name": uploaded_file.name,
                "file_extension": file_extension,
                "chunk_size": chunk_size,
                "overlap_size": overlap_size,
                "separators": separators,
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "location": f"{local_uploaded_file_dir}/{uploaded_file.name}",
                "in_vector_db": False
            }
            response = requests.post(
                url=APIPaths.get_full_path(APIPaths.KB_INSERT),
                json={
                    "database_name": MONGO_DATABASE_NAME,
                    "collection_name": MONGO_COLLECTION_DEFAULT_NAME,
                    "data": params_data
                }
            )  # json不是对应fastapi接口函数参数名, 而是request body的key

            if response.status_code == 200:
                # Save file to local disk
                with open(f"{local_uploaded_file_dir}/{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file_bytes)
                st.success("文件已添加到知识库", icon='🎉')
                time.sleep(1.0)
                st.rerun()  # Reload the page after adding
            else:
                st.error(f"添加文件到知识库时发生错误. 状态码: {response.status_code}")
                return  # not use st.rerun() here, to hang up the error message
