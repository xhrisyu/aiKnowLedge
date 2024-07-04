import base64
from typing import Optional, List, Generator, Dict, Any
import requests

from .prompt import *
from openai import OpenAI


class LLMAPIResponse:
    def __init__(self, content: Any, token_usage: int):
        self.content = content
        self.token_usage = token_usage


class OpenAILLM:
    MODEL = "gpt-4o"
    CHAT_MODEL = "gpt-4-turbo"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    INTENTION_RECOGNITION_MODEL = "gpt-4o"

    def __init__(self, api_key):
        self.embedding = OpenAI(api_key=api_key, max_retries=3).embeddings
        self.chat_completion = OpenAI(api_key=api_key, max_retries=3).chat.completions
        self._stream_chat_token_usage = 0

    @property
    def stream_chat_token_usage(self):
        return self._stream_chat_token_usage

    def get_text_embedding(self, text: str, embedding_model: Optional[str] = None) -> LLMAPIResponse:
        if not embedding_model:
            response = self.embedding.create(model=self.EMBEDDING_MODEL, input=text)
        else:
            response = self.embedding.create(model=embedding_model, input=text)

        token_usage = response.usage.total_tokens
        return LLMAPIResponse(response.data[0].embedding, token_usage)

    def get_chat_response(
            self,
            user_question: str,
            context: Optional[str],
            chat_history: List[Dict],
            temperature: Optional[float] = 0.8,
            model_name: Optional[str] = None
    ) -> LLMAPIResponse:
        messages = [
            {"role": "system", "content": KNOWLEDGE_QA_PROMPT},
            *chat_history,
            {"role": "user", "content": f"{context}\n用户问题：{user_question}\n"}
        ]
        response = self.chat_completion.create(
            model=self.CHAT_MODEL if not model_name else model_name,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        # Get token cost
        completion_tokens = response.usage.completion_tokens
        prompt_tokens = response.usage.prompt_tokens
        token_usage = response.usage.total_tokens
        return LLMAPIResponse(response.choices[0].message.content, token_usage)

    def stream_chat_response(
            self,
            user_question: str,
            context: Optional[str],
            chat_history: List[Dict],
            temperature: Optional[float] = 0.8,
            model_name: Optional[str] = None
    ) -> Generator[str, None, Optional[int]]:  # Generator[YieldType, SendType, ReturnType]
        messages = [
            {"role": "system", "content": KNOWLEDGE_QA_PROMPT},
            *chat_history,
            {"role": "user", "content": f"{context}\n用户问题：{user_question}\n"}
        ]
        response_stream = self.chat_completion.create(
            model=self.CHAT_MODEL if not model_name else model_name,
            messages=messages,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True}
        )
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
            if chunk.usage is not None:
                self._stream_chat_token_usage += chunk.usage.total_tokens  # only the last chunk has the total token count

    def intention_recognition(self, user_question: str, model_name: Optional[str] = None) -> LLMAPIResponse:
        messages = [
            {"role": "system", "content": USER_INTENTION_RECOGNITION_PROMPT},
            {"role": "user", "content": user_question}
        ]
        response = self.chat_completion.create(
            model=self.INTENTION_RECOGNITION_MODEL if not model_name else model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.5
        )
        token_usage = response.usage.total_tokens
        return LLMAPIResponse(response.choices[0].message.content, token_usage)

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
                            "text": PARSE_TABLE_CONTENT_PROMPT
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
                            "text": PARSE_IMAGE_CONTENT_PROMPT
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

