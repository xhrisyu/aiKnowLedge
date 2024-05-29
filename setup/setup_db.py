import configparser
import json
import psycopg2
import yaml
import pandas as pd
import os
from sshtunnel import SSHTunnelForwarder


config = yaml.safe_load(open("../config/config.yaml", "r"))
ssh_config = yaml.safe_load(open("../config/config.yaml", "r"))['ssh']


def import_knowledge_question_into_db():
    """
    Import generated knowledge and questions into database
    """

    # SSH服务器的地址和端口
    ssh_host = '120.198.21.93'
    ssh_port = 8022
    ssh_username = 'px'
    ssh_password = 'abcd@1234'
    # PostgreSQL数据库的地址和端口（在SSH隧道内部的地址和端口）
    db_host = 'localhost'
    db_port = 5432
    db_name = 'wecom'
    db_username = 'postgres'
    db_password = 'E4^KJ2rPbpn$PA^GR2'
    # 建立SSH隧道
    with SSHTunnelForwarder(
            (ssh_host, ssh_port),
            ssh_username=ssh_username,
            ssh_password=ssh_password,
            remote_bind_address=(db_host, db_port)
    ) as tunnel:
        try:
            conn = psycopg2.connect(
                database=db_name,
                user=db_username,
                password=db_password,
                host=tunnel.local_bind_host,
                port=tunnel.local_bind_port
            )
            cursor = conn.cursor()
        except Exception as e:
            print(f"Postgresql Client 初始化失败. Exception: {e}")
            return

        """读取SQL文件"""
        SQL_CONFIG = configparser.ConfigParser()
        SQL_CONFIG.read('sql.py')

        sql_template_knowledge = SQL_CONFIG.get('insert', 'insert_knowledge', raw=True)
        sql_template_question = SQL_CONFIG.get('insert', 'insert_question', raw=True)

        """写入knowledge和question"""
        root_path = "../doc/question"
        # file_names = [file_name for file_name in os.listdir(f"{root_path}") if file_name.endswith(".csv")]
        file_names = ['CQI-8_分层过程审核指南中文版知识点知识点-题目.csv']
        for file_name in file_names:
            knowledge_id, knowledge_tuple_list = 1, []
            question_id, question_tuple_list = 1, []

            # Load csv file
            df = pd.read_csv(f"{root_path}/{file_name}", header=None)
            for index, row in df.iterrows():
                print(f"Processing {file_name} {index}...")
                doc_code, knowledge, question = row[0], row[1], row[2]
                print(f"doc_code={doc_code}, knowledge={knowledge}, question={question}")
                # if row has empty value, skip it
                if not doc_code or not knowledge or not question:
                    continue

                if not pd.isna(knowledge):
                    knowledge_tuple_list.append((doc_code, knowledge_id, knowledge))
                    knowledge_id += 1

                if not pd.isna(question):
                    question = eval(question)
                    question_tuple_list.append((doc_code, question_id, question[0]['question'], 0, json.dumps(question[0]['options']), [question[0]['answer']]))
                    question_id += 1

            cursor.executemany(sql_template_knowledge, knowledge_tuple_list)
            conn.commit()
            cursor.executemany(sql_template_question, question_tuple_list)
            conn.commit()

        cursor.close()
        conn.close()


if __name__ == "__main__":
    import_knowledge_question_into_db()
