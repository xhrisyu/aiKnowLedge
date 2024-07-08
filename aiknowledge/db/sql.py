insert_knowledge = """INSERT INTO knowledges (doc_code, knowledge_id, knowledge_text) VALUES (%s, %s, %s)"""

insert_question = """
INSERT INTO questions (doc_code, question_id, question_text, question_type, options, correct_options) 
VALUES (%s, %s, %s, %s, %s::jsonb, %s)
"""