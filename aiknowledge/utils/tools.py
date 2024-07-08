import os
import re
from typing import List, Dict
import numpy as np


def get_metadata_source_filename(source_name: str) -> str:
    match = re.search(r'/([^/]+)\s*\.', source_name)
    if match:
        source_name = match.group(1)
    return source_name


def convert_chat_message_to_str(chat_history: List[Dict]) -> str:
    chat_history_text = ""
    for message in chat_history:
        if message['role'] == "user":
            chat_history_text += f"用户: {message['content']}\n"
        elif message['role'] == "ai" or message['role'] == "assistant":
            chat_history_text += f"AI: {message['content']}\n"

    return chat_history_text


def get_company_document_code(document_name: str) -> str:
    """
    Get doc code from intflex doc name
    """
    # num_part = document_name.split("-")[-1][:3]
    # document_code = "-".join(document_name.split("-")[:-1]) + "-" + num_part
    # return document_code

    document_name = os.path.splitext(document_name)[0]  # remove extension
    pattern = r'(([A-Z0-9]{1,2}-){1,3}\d{3})'
    match = re.search(pattern, document_name)
    if match:
        return match.group(1)
    else:
        return document_name


def get_file_path_list(root_path: str, file_suffix: str) -> list:
    # Check file suffix
    if not file_suffix.startswith("."):
        file_suffix = "." + file_suffix

    file_path_list = []

    # Get absolute path from root_path (eliminate the mistake of using relative path)
    root_path = os.path.abspath(root_path)

    # Sub folders
    sub_paths = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    for sub_path in sub_paths:
        file_names = [file_name for file_name in os.listdir(f"{root_path}/{sub_path}") if file_name.endswith(file_suffix)]
        for file_name in file_names:
            file_path_list.append(f"{root_path}/{sub_path}/{file_name}")

    # Non-sub folders
    file_names = [file_name for file_name in os.listdir(f"{root_path}") if file_name.endswith(file_suffix)]
    for file_name in file_names:
        file_path_list.append(f"{root_path}/{file_name}")

    # Sort file path list by file name
    file_path_list.sort()
    return file_path_list


# ------------------- Document Utils -------------------
def split_document(document: str, split_length: int, overlap_length: int):
    """
    Split article into multiple parts (last part ranged in [split_length, 2 * split_length])
    :param document: original article length
    :param split_length: each split length
    :param overlap_length: overlap length
    :return: list of split article
    """
    split_chunks = []
    doc_len = len(document)
    idx, pos = 1, 0
    while pos < doc_len:
        content = document[pos:pos + split_length]  # if pos+split_length is greater than doc_len, then return the rest
        split_chunks.append({"id": idx, "content": content})
        pos += split_length - overlap_length
        idx += 1

    # check the last chunk, if the length is less than 50% of the split_length, then merge it with the previous chunk
    if len(split_chunks) > 1 and len(split_chunks[-1]["content"]) < split_length * 0.5:
        split_chunks[-2]["content"] += split_chunks[-1]["content"]
        split_chunks.pop()

    return split_chunks


def split_document_by_separator(document: str, split_length: int, overlap_length: int, separator: str = '\n'):
    split_chunks = []
    doc_len = len(document)
    idx, pos = 1, 0

    pre_split_chunks = document.split(separator)
    pre_split_chunks_i = 0
    pre_split_chunks_len = len(pre_split_chunks)
    print(f"pre_split_chunks_len: {pre_split_chunks_len}")

    while pos < doc_len:
        content = ""
        while pre_split_chunks_i < pre_split_chunks_len and len(content) < split_length:
            content += pre_split_chunks[pre_split_chunks_i] + separator
            pre_split_chunks_i += 1

        if pre_split_chunks_i >= pre_split_chunks_len:
            break

        if abs(split_length - len(content)) > abs(len(content) + len(pre_split_chunks[pre_split_chunks_i]) - split_length):
            content += pre_split_chunks[pre_split_chunks_i] + separator
            pre_split_chunks_i += 1

        print(f"pos: {pos}, len(content): {len(content)}")
        split_chunks.append({"id": idx, "content": content})
        pos += len(content) - overlap_length
        idx += 1

    return split_chunks


# =================== File Utils ===================
def file_exist(file_path: str) -> bool:
    return os.path.exists(file_path)


def convert_escaped_chars_to_original_chars(escaped_list: List[str]) -> List[str]:
    """
    将包含转义字符的字符串列表转换为包含实际字符的字符串列表。
    :param escaped_list: 包含转义字符的字符串列表
    :return: 包含实际字符的字符串列表
    """
    original_list = ['\\n\\n', '\\n', '。', '\\t', '.', 'a']
    return [s.encode().decode('unicode_escape') for s in original_list]


def convert_numpy_types(obj):
    """
    将 numpy 类型转换为 Python 原生类型
    :param obj:
    :return:
    """
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def get_file_name(file_path: str, with_extension: bool = True) -> str:
    """
    Get the file name from the file path.
    :param file_path: The file path.
    :param with_extension: Whether to include the file extension.
    :return: The file name.
    """

    # Exp: "/root/folder_name/CQI-8_分层过程审核指南中文版知识点.txt" -> "CQI-8_分层过程审核指南中文版知识点.txt"

    if with_extension:
        return os.path.basename(file_path)
    else:
        return os.path.splitext(os.path.basename(file_path))[0]


def get_file_extension(file_path: str, with_dot: bool, upper_case: bool) -> str:
    """
    Get the file extension from the file path.
    :param file_path: file path
    :param with_dot: if True, return the extension with dot
    :param upper_case: if True, return the upper case extension
    :return: file extension
    """

    # Exp: "/root/folder_name/CQI-8_分层过程审核指南中文版知识点.txt" -> ".txt" | "txt"

    if with_dot:
        file_extension = os.path.splitext(os.path.basename(file_path))[1]
    else:
        file_extension = os.path.splitext(os.path.basename(file_path))[1][1:]

    return file_extension.upper() if upper_case else file_extension


def get_all_file_path_from_folder(folder_path: str, is_absolute_path: bool = True) -> list[str]:

    all_file_paths = []
    for _root, _dirs, _files in os.walk(folder_path):
        if is_absolute_path:
            _root = os.path.abspath(_root)
        else:
            _root = os.path.relpath(_root)
        for _file in _files:
            all_file_paths.append(str(os.path.join(_root, _file)))  # string type

    return all_file_paths

