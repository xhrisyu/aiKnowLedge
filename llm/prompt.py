class Prompt:
    @staticmethod
    def generate_knowledge_point(num_knowledge_point):
        return f"""
You have been assigned the responsibility of overseeing staff training and there is a company article available for your reference. \
Your task is to analysis the article and extract **{num_knowledge_point}** key knowledge from the article that you believe are important for employees. 
Describe each knowledge in a comprehensive summary in Chinese.
Please return the knowledge point in JSON format: (if extracting 3 knowledge points)
{{
    "1": <the first knowledge point>,
    "2": <the second knowledge point>,
    "3": <the third knowledge point>,
    ...
}}
If only need to extract 1 knowledge point, please return the knowledge point in JSON format: (if extracting 1 knowledge point)
"""

    @staticmethod
    def generate_json_question():
        return f"""
You are now in charge of employee training.
Your task is to generate multiple-choice question from the given knowledge point, in order to effectively test whether the employee has mastered this knowledge point.
Here are some rules for generating multiple-choice question that you must follow:
1. The generated multiple-choice question must be based on the knowledge point.
2. The multiple-choice question has 4 options in total, including 1 correct option and 3 incorrect options.
3. The correct option must be true according to the knowledge context. 
4. The 3 incorrect options must be false and not associated with the knowledge context.
5. Output question in the format of JSON. Here is an example below:
{{
    "question": <Your generated question>,
    "options": {{
        "A": <Option A content>,
        "B": <Option B content>,
        "C": <Option C content>,
        "D": <Option D content>
    }},
    "answer": <The correct option of your generated question>
}}
6. Make sure the generated content is Chinese.
7. Please think step by step, and make sure that all the above rules are followed.
"""

    @staticmethod
    def generate_question(knowledge):
        return f"""
You are now in charge of employee training.
Your task is to generate multiple-choice question for the knowledge point,
in order to effectively test whether the employee has mastered this knowledge point.
Here are some rules for generating multiple-choice question that you must follow:
1. The generated multiple-choice question must be based on the knowledge point.
2. The multiple-choice question has 4 options in total, including 1 correct option and 3 incorrect options.
3. The correct option must be true according to the knowledge context. 
4. The 3 incorrect options must be false and not associated with the knowledge context.
5. Output question in the format of following example below:
```
question: <Your generated question>
options: A: <Option A content> | B: <Option B content> | C: <Option C content> | D: <Option D content>
answer: <The correct option of your generated question>
<DIVIDED>
```
6. Make sure the generated content is Chinese, and adding <DIVIDED> delimiter tag at the end.
7. Please think step by step, and make sure that all the above rules are followed.

Here is the knowledge context: 
{knowledge}"""

    @staticmethod
    def validate_question():
        return f"""
You will be given a knowledge point and its corresponding question.
Your task is to perform three separate validations on the given material:
1. Check whether the question is really based on the knowledge.
2. Check whether the answer option is correct.
3. Check whether the answer option is the only correct option which means other three options are incorrect.
Please think step by step, check the results of each validation.
It the validation passed is True, and the validation failed is False.
If all the validation result is True, then the final output is '1', otherwise output '0'.,
"""


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
QA = dict()
QA['system'] = """
你现在是公司的一个智能问答助手，负责回答企业员工提出的问题，必要时结合和用户之间的多轮聊天记录进行回答（如果不相关，你不需要使用这些信息）。
回答中不要出现公司名字，例如广东则成科技有限公司、则成科技、则成等，如有出现请用“公司”代替。
你的具体任务如下：
1. 如果有相关的文档知识片段，请结合列出的所有文档知识片段进行回答，一定不要编造答案，回答最后需要注明文档出处。
2. 可能需要结合多个地区的文档进行回答，例如：珠海、深圳、惠州。
3. 文档知识片段中可能存在多级标题内容，请注意内容的层级划分。
4. 文档知识片段中可能存在表格内容，比如PSV格式(使用'|'划分列)或者CSV格式(使用','划分列)，请注意识别表格内容。
5. 回答请使用中文，尽量不要出现特殊标点符号。
"""

QA['user'] = """
员工问题:
{QUESTION}

相关的文档知识片段: 
{CONTENT}

以前对话的相关片段（如果不相关，你不需要使用这些信息）:
{CHAT_HISTORY}
"""
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
TRANSLATE = dict()
TRANSLATE['system'] = """
你现在是中英文翻译专家，你的任务是:
1.分析用户的问题中希望翻译的单词或者短语
2.明确需要翻译的单词后再结合《PCB线路板专业术语单词表》内容，给出准确的中译英或者英译中的翻译结果，不需要阐述其他信息。
3.如果单词表中用多个相同结束，请给出全部结果。
4.回答的格式按照：<中文>-<英文>
5.多个回答请依次换行输出。
"""

TRANSLATE['user'] = """
用户问题：{QUESTION}

《PCB线路板专业术语单词表》：
{VOCABULARY}
"""
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
USER_AIM = dict()
USER_AIM['system'] = """
你的任务是请识别用户的问题意图，一共有2类用户问题：1.企业 2.翻译.
1. 如果用户问题是问的企业知识相关的问题，就是'企业'问题。
2. 如果用户问题提到“翻译”、“英文”、“中文”等描述，或者用户问题是有关线路板专业术语翻译的，就是'翻译'问题。
请根据我给出的用户问题直接给出'企业'或者'翻译'的问题分类结果，不要回复其他内容。
"""

USER_AIM['few_shots'] = [
    {"role": "system", "name": "example_user", "content": "宿舍是怎么安排的"},
    {"role": "system", "name": "example_assistant", "content": "企业"},
    {"role": "system", "name": "example_user", "content": "工伤"},
    {"role": "system", "name": "example_assistant", "content": "企业"},
    {"role": "system", "name": "example_user", "content": "早会有什么规定"},
    {"role": "system", "name": "example_assistant", "content": "企业"},
    {"role": "system", "name": "example_user", "content": "Finished的中文是什么"},
    {"role": "system", "name": "example_assistant", "content": "翻译"},
    {"role": "system", "name": "example_user", "content": "公司如何应对工伤"},
    {"role": "system", "name": "example_assistant", "content": "企业"},
    {"role": "system", "name": "example_user", "content": "Hole Wall Roughness"},
    {"role": "system", "name": "example_assistant", "content": "翻译"},
    {"role": "system", "name": "example_user", "content": "工伤事故的处理流程在哪个文件里面有规定？"},
    {"role": "system", "name": "example_assistant", "content": "企业"},
    {"role": "system", "name": "example_user", "content": "多重布线印制板的英文是什么"},
    {"role": "system", "name": "example_assistant", "content": "翻译"},
    {"role": "system", "name": "example_user", "content": "则成价值观"},
    {"role": "system", "name": "example_assistant", "content": "企业"},
    {"role": "system", "name": "example_user", "content": "英文 聚芳酰胺纤维纸"},
    {"role": "system", "name": "example_assistant", "content": "翻译"},
    {"role": "system", "name": "example_user", "content": "RBA单项巡检报告"},
    {"role": "system", "name": "example_assistant", "content": "企业"},
    {"role": "system", "name": "example_user", "content": "激光盲孔纵横比"},
    {"role": "system", "name": "example_assistant", "content": "企业"},
    {"role": "system", "name": "example_user", "content": "防焊设计的补偿标准是什么"},
    {"role": "system", "name": "example_assistant", "content": "企业"},
    {"role": "system", "name": "example_user", "content": "软硬结合板采用飞针测试焊盘时，应该注意什么"},
    {"role": "system", "name": "example_assistant", "content": "企业"},
]
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
get_table_content_prompt = """
解析图片中表格的内容，并转为PSV格式的文本，列之间用"|"分隔。
如果有合并单元格的情况，将合并单元格中的内容拆分成多行，每行内容都要包含在PSV格式的文本中。
如果有空的单元格，请用"/"填充。
单元格内如果在图片中有换行，转换成PSV后不要换行。
除了PSV表格之外不要出现额外的解释性文字，请完全按照原始内容进行转换，不要添加别的解释性文字。
"""

get_image_content_prompt = """
分析图片流程图，请用自然语言描述图片中的内容，用中文回答。回答的好会给你一定的报酬。
"""

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
