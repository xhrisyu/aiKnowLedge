import json
import re
from typing import List, Dict, Optional
from openai import OpenAI

from .prompt import GENERATE_KNOWLEDGE_POINT_PROMPT, GENERATE_QUESTION_PROMPT, VALIDATE_QUESTION_PROMPT

MAX_RETRIES = 3


class QuizGenerator:
    knowledge_points: List[str]
    questions: List[Dict]

    def __init__(self, api_key: str, context: str, num: int, model_name: str, temperature: float):
        self.chat_completion = OpenAI(api_key=api_key, max_retries=3).chat.completions
        self.context = context
        self.num = num
        self.model_name = model_name
        self.temperature = temperature

    def generate_knowledge_point(self, context: Optional[str] = None) -> list[str]:
        messages = [
            {"role": "system", "content": GENERATE_KNOWLEDGE_POINT_PROMPT.format(NUM_KNOWLEDGE_POINT=self.num)},
        ]
        if not context:
            messages.append({"role": "user", "content": f"Article:\n{self.context}"})
        else:
            messages.append({"role": "user", "content": f"Article:\n{context}"})

        gpt_response = self.chat_completion.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self.temperature
        )
        response_str = gpt_response.choices[0].message.content
        response_json = json.loads(response_str)  # {'1': '知识点1', '2': '知识点2', ...}
        self.knowledge_points = [value if value != "" else key for key, value in response_json.items()]
        return self.knowledge_points  # ['知识点1', '知识点2', ...]

    def generate_question(self, knowledge_point: list[str]) -> list[dict]:
        generated_questions = []
        for item in knowledge_point:
            messages = [
                {"role": "system", "content": GENERATE_QUESTION_PROMPT},
                {"role": "user", "content": f"The Knowledge point:\n{item}"}
            ]
            gpt_response = self.chat_completion.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self.temperature,
            )
            response_str = gpt_response.choices[0].message.content
            json_response = json.loads(response_str)
            generated_questions.append(json_response)
        self.questions = generated_questions
        return self.questions

    def validate_question(self, question: list[dict]):
        messages = [
            {"role": "system", "content": VALIDATE_QUESTION_PROMPT},
            {"role": "user", "content": f"{question}"}
        ]
        gpt_response = self.chat_completion.create(
            model=self.model_name,
            messages=messages,
            temperature=0.5,
        )
        raw_check_result = gpt_response.choices[0].message.content
        check_result = re.search(r'output:\s*(.*)', raw_check_result).group(1).strip()
        return raw_check_result
