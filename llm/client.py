import base64
import time
from typing import Optional, List, Generator

import requests
import re
from . import prompt
from openai import OpenAI


class OpenAILLM:
    MODEL = "gpt-4o"
    CHAT_MODEL = "gpt-4-0125-preview"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    TRANSLATE_MODEL = "gpt-3.5-turbo-instruct"
    CLASSIFY_MODEL = "gpt-4-0125-preview"
    GENERATE_CONTENT_MODEL = "gpt-4-1106-preview"

    def __init__(self, api_key):
        self.embedding = OpenAI(api_key=api_key, max_retries=3).embeddings
        self.chat_completion = OpenAI(api_key=api_key, max_retries=3).chat.completions

    def get_text_embedding(self, text: str, embedding_model: Optional[str] = None) -> List[float]:
        if not embedding_model:
            response = self.embedding.create(model=self.EMBEDDING_MODEL, input=text)
        else:
            response = self.embedding.create(model=embedding_model, input=text)
        return response.data[0].embedding

    def get_chat_response(
            self,
            user_question: str,
            context: Optional[str],
            chat_history: Optional[str],
            temperature: Optional[float] = 0.8,
            model_name: Optional[str] = None
    ) -> Optional[str]:
        # messages = [
        #     {"role": "system", "content": prompt.QA["system"]},
        #     {"role": "user", "content": prompt.QA["user"].format(
        #         CONTENT=related_text,
        #         QUESTION=user_question,
        #         CHAT_HISTORY=chat_history if chat_history else ""
        #     )}
        # ]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{context}\n\nQ: {user_question}\nA:"}
        ]
        response = self.chat_completion.create(
            model=self.CHAT_MODEL if not model_name else model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=2048,
            stream=False,
        )
        # Get token cost
        completion_tokens = response.usage.completion_tokens
        prompt_tokens = response.usage.prompt_tokens
        total_tokens = response.usage.total_tokens
        # print(f"[[企业问答消耗的token数量: {total_tokens}]]")
        return response.choices[0].message.content

    def stream_chat_response(
            self,
            user_question: str,
            context: Optional[str],
            chat_history: Optional[str],
            temperature: Optional[float] = 0.8,
            model_name: Optional[str] = None
    ) -> Generator[str, None, None]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{context}\n\nQ: {user_question}\nA:"}
        ]
        response_stream = self.chat_completion.create(
            model=self.CHAT_MODEL if not model_name else model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=2048,
            stream=True
        )
        for chunk in response_stream:
            if chunk.choices[0].delta.content is not None:
                chunk_message = chunk.choices[0].delta.content
                yield chunk_message
            time.sleep(0.02)  # 模拟延迟

    def intention_recognition(self, user_question: str, model_name: Optional[str] = None) -> str:
        messages = [
            {"role": "user", "content": user_question}
        ]
        response = self.chat_completion.create(
            model=self.CLASSIFY_MODEL if not model_name else model_name,
            messages=messages,
            temperature=0.8,
            max_tokens=2048,
            n=1
        )
        return response.choices[0].message.content

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_table_content(self, table_pic_path: str, api_key: str) -> str:
        """
        Get table pic content from table pic path
        """
        base64_image = self.encode_image(table_pic_path)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt.get_table_content_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000
        }

        response = requests.post("https://api.client.com/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()
        description = response_json["choices"][0]["message"]["content"]
        return description

    def get_image_content(self, image_path: str, api_key: str) -> str:
        base64_image = self.encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt.get_image_content_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000
        }

        response = requests.post("https://api.client.com/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()
        description = response_json["choices"][0]["message"]["content"]
        return description

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # --------- Generate Knowledge and Question Part ---------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def generate_knowledge(self, text: str, n_knowledge: int = 1) -> str:
        messages = [{"role": "user",
                     "content": prompt.GENERATE_KNOWLEDGE.format(NUM_KNOWLEDGE_POINT=n_knowledge, ARTICLE=text)}]
        response_gpt = self.chat_completion.create(
            model=self.GENERATE_CONTENT_MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=2048,
            n=n_knowledge
        )
        return response_gpt.choices[0].message.content

    def generate_question(self, knowledge: str, n_question: int = 1) -> list[dict]:
        def parse_raw_questions_to_json(_raw_questions: str) -> list[dict]:
            """
            Parse ChatGPT generated string content, and convert each generated question --to--> json format
            :param _raw_questions: string of ChatGPT response
            :return: list of question in json format
            """
            """expected format:
            {
                "question": "分层过程审核在制造行业中的作用是什么？",
                "options": {
                    "A": "减少员工培训成本",
                    "B": "确保产品质量",
                    "C": "增加生产线速度",
                    "D": "提高员工休息时间"
                },
                "answer": "B:确保产品质量"
            }
            """
            processed_questions = []
            for _raw_question in _raw_questions.split('<DIVIDED>'):
                _raw_question = _raw_question.split('\n')
                json_question = {}
                for _section in _raw_question:
                    _section = _section.replace(" ", "")  # remove space and special characters
                    # section = re.sub(r'[^\w\s.,?;:。，？；：|\-~～]', '', section)
                    if _section == "": continue
                    print(_section)

                    if _section.startswith("question"):
                        question_content = _section.split("question:")[1]
                        json_question["question"] = question_content

                    elif _section.startswith("options"):
                        json_options = {}
                        for _option in _section.split("options:")[1].split("|"):
                            if not _option: continue
                            option_tag, option_content = _option.split(":")[0], _option.split(":")[1]
                            json_options[option_tag] = option_content
                        json_question["options"] = json_options

                    elif _section.startswith("answer"):
                        answer_content = _section.split("answer:")[1]
                        # if key options not in json_question, then skip this question
                        if "options" not in json_question:
                            continue

                        if len(answer_content) > 0 and answer_content[0] in ["A", "B", "C", "D"]:
                            _option = answer_content[0]
                            # check if all the options(A, B, C, D) is in the options key
                            json_options_keys = list(json_question["options"].keys())
                            if len(json_options_keys) == 4 and "A" in json_options_keys and "B" in json_options_keys and "C" in json_options_keys and "D" in json_options_keys:
                                answer_content = _option + ":" + json_question["options"][_option]
                                json_question["answer"] = answer_content

                if len(json_question) == 3:
                    processed_questions.append(json_question)

            return processed_questions

        messages = [{"role": "user", "content": prompt.GENERATE_QUESTION.format(KNOWLEDGE=knowledge)}]
        response_gpt = self.chat_completion.create(
            model=self.GENERATE_CONTENT_MODEL,
            messages=messages,
            temperature=1.0,
            max_tokens=2048,
            n=1
        )
        raw_questions = response_gpt.choices[0].message.content
        return parse_raw_questions_to_json(raw_questions)

    def validate_question(self, knowledge: str, question: dict):
        messages = [
            {"role": "system", "content": prompt.VALIDATE_QUESTION["system"]},
            *prompt.VALIDATE_QUESTION['few_shots'],
            {"role": "user",
             "content": prompt.VALIDATE_QUESTION["user"].format(KNOWLEDGE=knowledge, QUESTION=question["question"],
                                                                OPTIONS=question["options"], ANSWER=question["answer"])}
        ]
        print(f"messages: {messages}\n{'-' * 100}\n")

        response_gpt = self.chat_completion.create(
            model=self.GENERATE_CONTENT_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            n=1,
        )
        raw_check_result = response_gpt.choices[0].message.content
        print(f"raw_check_result:\n{raw_check_result}\n{'-' * 100}\n")
        check_result = re.search(r'output:\s*(.*)', raw_check_result).group(1).strip()
        print(f"check_result:\n{check_result}\n{'-' * 100}\n")
        return raw_check_result


class QianfanLLM:
    def __init__(self):
        self.name = "qianfan"

    def get_name(self):
        return self.name
