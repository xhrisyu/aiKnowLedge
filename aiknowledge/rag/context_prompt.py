from typing import List, Dict


def format_context_prompt(retrieval_augmentation_result: List[List[Dict]]) -> str:
    """
    Convert retrieval augmentation result to context prompt.
    The retrieval augmentation results have multiple sub-query results.

    retrieval_augmentation_result structure:
    [
        # query 1
        [
            {
                "score": 0.1,
                "chunk_id": "",
                "doc_name": "",
                "chunk_seq": 1,
                "pre_content": "",
                "content": "",
                "next_content": "",
            },
            ...
        ],
        # query 2
        [], ...
    ]

    1. remove duplicated chunk based on `chunk_id`
    2. format the context prompt
    """

    context_prompt = ""
    chunk_id_set = set()
    for no_query in range(len(retrieval_augmentation_result)):
        for retrieved_payload in retrieval_augmentation_result[no_query]:
            chunk_id = retrieved_payload["chunk_id"]
            doc_name = retrieved_payload["doc_name"]
            if chunk_id in chunk_id_set:
                continue
            chunk_id_set.add(chunk_id)
            content = retrieved_payload["pre_content"] + retrieved_payload["content"] + retrieved_payload["next_content"]
            content = content.replace("\n\n", "\n")
            context_prompt += f"文本来源:《{doc_name}》\n正文:{content}\n<DIVIDER>\n"

    return context_prompt


def format_qa_history_prompt(retrieval_augmentation_result: List[List[Dict]]) -> str:
    """
    retrieval_augmentation_result structure:
    [
        {
            "score": 0.1,
            "chunk_id": "",
            "doc_name": "",
            "chunk_seq": 1,
            "pre_content": "",
            "content": "",
            "next_content": "",
        },
        ...
    ]

    1. remove duplicated chunk based on `chunk_id`
    2. format the context prompt
    """

    qa_history_prompt = ""
    chunk_id_set = set()
    for no_query in range(len(retrieval_augmentation_result)):
        for retrieved_payload in retrieval_augmentation_result[no_query]:
            chunk_id = retrieved_payload["chunk_id"]
            if chunk_id in chunk_id_set:
                continue
            chunk_id_set.add(chunk_id)
            content = retrieved_payload["content"].replace("\n\n", "\n")
            qa_history_prompt += f"参考历史问答:\n{content}\n"

    return qa_history_prompt
