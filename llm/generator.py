import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from tqdm import tqdm
import time
import pandas as pd

from llm import OpenAILLM
from utils.tools import get_file_name_no_extension

MAX_RETRIES = 3


class KnowledgeQuestionGenerator:
    def __init__(self, source_folder_path: str, output_folder_path: str) -> None:
        # Check the folder path
        if not os.path.exists(source_folder_path):
            raise ValueError(f"Source File Folder [{source_folder_path}] not exists")

        # Get all the source files from this folder
        self.source_file_abs_paths = []
        for _root, _dirs, _files in os.walk(source_folder_path):
            _root = os.path.abspath(_root)  # get absolute path
            for _file in _files:
                if _file.endswith(".txt"):
                    self.source_file_abs_paths.append(os.path.join(_root, _file))

        # Check the output path
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        self.output_path = output_folder_path

    def generate(self, llm_model: OpenAILLM):
        # Process each file
        for _file_name in self.source_file_abs_paths:
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
                        knowledge_text = llm_model.generate_knowledge(text=_chunk.page_content, n_knowledge=1).strip()
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
            output_path = f'{self.output_path}/{get_file_name_no_extension(_file_name)}_kq.csv'
            output_df.to_csv(output_path, index=True, encoding="utf-8")

# if __name__ == "__main__":
#     source_folder_path = "/Users/yu/Projects/AIGC/intflex_qa/doc_temp"
#     output_folder_path = "/Users/yu/Projects/AIGC/intflex_qa/output_temp"
#     load_dotenv(".env")
#     llm_model = OpenAILLM()
#     kq_generator = KnowledgeQuestionGenerator(source_folder_path=source_folder_path, output_folder_path=output_folder_path)
#     kq_generator.generate(llm_model)
