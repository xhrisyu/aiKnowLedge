import json
import os
import re
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from tqdm import tqdm
import time
import pandas as pd
from openai import OpenAI

from .client import OpenAILLM
from .prompt import Prompt
from utils.tools import get_file_name_no_extension

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

    def generate_knowledge_point(self) -> List[str]:
        messages = [
            {"role": "system", "content": Prompt.generate_knowledge_point(num_knowledge_point=self.num)},
            {"role": "user", "content": f"Article:\n{self.context}"}
        ]
        gpt_response = self.chat_completion.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self.temperature
        )
        response_str = gpt_response.choices[0].message.content
        response_json = json.loads(response_str)  # {'1': '知识点1', '2': '知识点2', ...}
        self.knowledge_points = [value if value != "" else key for key, value in response_json.items()]
        return self.knowledge_points

    def generate_question(self, knowledge_point: List[str]) -> List[Dict]:
        generated_questions = []
        for item in knowledge_point:
            messages = [
                {"role": "system", "content": Prompt.generate_json_question()},
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

    def validate_question(self, question: List[Dict]):
        messages = [
            {"role": "system", "content": Prompt.validate_question()},
            {"role": "user", "content": f"{question}"}
        ]
        print(f"messages: {messages}\n{'-' * 100}\n")
        gpt_response = self.chat_completion.create(
            model=self.model_name,
            messages=messages,
            temperature=0.5,
        )
        raw_check_result = gpt_response.choices[0].message.content
        print(f"raw_check_result:\n{raw_check_result}\n{'-' * 100}\n")
        check_result = re.search(r'output:\s*(.*)', raw_check_result).group(1).strip()
        print(f"check_result:\n{check_result}\n{'-' * 100}\n")
        return raw_check_result

    @staticmethod
    def generate(llm_model: OpenAILLM, source_folder_path: str, output_folder_path: str):
        # Check the folder path
        if not os.path.exists(source_folder_path):
            raise ValueError(f"Source File Folder [{source_folder_path}] not exists")

        # Get all the source files from this folder
        source_file_abs_paths = []
        for _root, _dirs, _files in os.walk(source_folder_path):
            _root = os.path.abspath(_root)  # get absolute path
            for _file in _files:
                if _file.endswith(".txt"):
                    source_file_abs_paths.append(os.path.join(_root, _file))

        # Check the output path
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        output_path = output_folder_path

        # Process each file
        for _file_name in source_file_abs_paths:
            print(f"Processing: {_file_name}")

            # Split by Langchain Method
            txt_data = TextLoader(_file_name).load()
            split_txt_data = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                separators=["\n\n", "。", ".", "\n"]
            ).split_documents(txt_data)

            # Generate knowledge & question for each chunk in the current file
            kq_pairs = []
            for _chunk in tqdm(split_txt_data, desc="Generating Knowledge & Question"):
                # Generate Knowledge
                retries = 0
                knowledge_text = ""
                while retries < MAX_RETRIES:
                    try:
                        knowledge_text = llm_model.generate_knowledge_point(text=_chunk.page_content,
                                                                            num_point=1).strip()
                        break
                    except Exception as e:
                        print(f"Network Error for `generate knowledge`: {e}\nRetrying...")
                        retries += 1
                        time.sleep(1)  # add delay

                if knowledge_text == "":
                    continue

                # Generate Question
                retries = 0
                question_json = {}
                while retries < MAX_RETRIES:
                    try:
                        question_json = llm_model.generate_question(knowledge=knowledge_text, n_question=1)
                        break
                    except Exception as e:
                        print(f"Network Error for `generate question`: {e}\nRetrying...")
                        retries += 1
                        time.sleep(1)  # add delay

                # Save this knowledge & question pair
                kq_pairs.append([knowledge_text, question_json])

            # Convert to DataFrame, and Save generated knowledge to local file
            output_df = pd.DataFrame(data=kq_pairs, columns=["Knowledge", "Question"])
            output_path = f'{output_path}/{get_file_name_no_extension(_file_name)}_kq.csv'
            output_df.to_csv(output_path, index=True, encoding="utf-8")

    @staticmethod
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

# if __name__ == "__main__":
#     source_folder_path = "/Users/yu/Projects/AIGC/intflex_qa/doc_temp"
#     output_folder_path = "/Users/yu/Projects/AIGC/intflex_qa/output_temp"
#     load_dotenv(".env")
#     llm_model = OpenAILLM()
#     kq_generator = KnowledgeQuestionGenerator(source_folder_path=source_folder_path, output_folder_path=output_folder_path)
#     kq_generator.generate(llm_model)
