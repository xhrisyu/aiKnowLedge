from typing import Optional, Generator, Any
from openai import OpenAI

from .prompt import KNOWLEDGE_QA_PROMPT, ENTITY_RECOGNITION_PROMPT, QUERY_DECOMPOSITION_PROMPT


class LLMAPIResponse:
    def __init__(self, content: Any, token_usage: int):
        self.content = content
        self.token_usage = token_usage


class OpenAILLM:
    CHAT_MODEL = "gpt-4-turbo"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    INTENTION_RECOGNITION_MODEL = "gpt-4o"

    def __init__(self, api_key):
        self._embedding = OpenAI(api_key=api_key, max_retries=3).embeddings
        self._chat_completion = OpenAI(api_key=api_key, max_retries=3).chat.completions
        self._stream_chat_token_usage = 0

    @property
    def stream_chat_token_usage(self):
        return self._stream_chat_token_usage

    def get_text_embedding(self, text: str, embedding_model: Optional[str] = None) -> LLMAPIResponse:
        if not embedding_model:
            response = self._embedding.create(model=self.EMBEDDING_MODEL, input=text)
        else:
            response = self._embedding.create(model=embedding_model, input=text)

        token_usage = response.usage.total_tokens
        return LLMAPIResponse(response.data[0].embedding, token_usage)

    def get_chat_response(
            self,
            user_question: str,
            context: Optional[str],
            chat_history: list[dict],
            temperature: Optional[float] = 0.6,
            model_name: Optional[str] = None
    ) -> LLMAPIResponse:
        messages = [
            {"role": "system", "content": KNOWLEDGE_QA_PROMPT},
            *chat_history,
            {"role": "user", "content": f"{context}\n用户问题：{user_question}\n"}
        ]
        response = self._chat_completion.create(
            model=self.CHAT_MODEL if not model_name else model_name,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        # Get token cost
        # completion_tokens = response.usage.completion_tokens
        # prompt_tokens = response.usage.prompt_tokens
        token_usage = response.usage.total_tokens
        return LLMAPIResponse(response.choices[0].message.content, token_usage)

    def stream_chat_response(
            self,
            user_question: str,
            context: Optional[str],
            qa_history: list[dict] | str,
            temperature: Optional[float] = 0.2,
            model_name: Optional[str] = None
    ) -> Generator[str, None, Optional[int]]:  # Generator[YieldType, SendType, ReturnType]
        messages = [
            # {"role": "system", "content": },
            {"role": "user", "content": f"{KNOWLEDGE_QA_PROMPT.format(CONTEXT=context, QA_HISTORY=qa_history)}\n用户问题：{user_question}"}
        ]
        response_stream = self._chat_completion.create(
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

    def query_decomposition(self, user_question: str, model_name: Optional[str] = None) -> LLMAPIResponse:
        messages = [
            {"role": "user", "content": QUERY_DECOMPOSITION_PROMPT + user_question}
        ]
        response = self._chat_completion.create(
            model=self.INTENTION_RECOGNITION_MODEL if not model_name else model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        token_usage = response.usage.total_tokens
        return LLMAPIResponse(response.choices[0].message.content, token_usage)

    def entity_recognition(self, user_question: str, model_name: Optional[str] = None) -> LLMAPIResponse:
        messages = [
            {"role": "user", "content": ENTITY_RECOGNITION_PROMPT + user_question}
        ]
        response = self._chat_completion.create(
            model=self.INTENTION_RECOGNITION_MODEL if not model_name else model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        token_usage = response.usage.total_tokens
        return LLMAPIResponse(response.choices[0].message.content, token_usage)
