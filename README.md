# aiKnowLedge

## Manage your knowledge base

1. View your knowledge base
2. Add new file to your knowledge base
    - customize parameters [chunk size, overlap size, separators]
3. Delete file from your knowledge base
4. Add file to vector database(for QA chatbot retrieval)
5. Delete file from vector database

## QA Chatbot

1. Answer your question base on your knowledge base(imported to vector database)
   - customize parameters [model_name, temperature, stream, top_k, similarity threshold]
   - model_name: gpt-3.5-turbo, gpt-4-turbo, gpt-4o
   - temperature: LLM temperature
   - stream: Streaming output
   - top_k: Top k chunk retrieved from vector database
   - similarity threshold: Similarity lower bound for QA retrieval
2. Display the retrieved chunk from vector database, including page_id and chunk_id

## Knowledge Point & Question Generating Interface

Upload a paragraph to generate questions and answers


## Run locally

```shell
conda env create -n aiknowledge python=3.11.4
pip install -r requirements.txt
```

Terminal 1, frontend(streamlit):
```shell
streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Terminal 2, backend(fastapi):
```shell
uvicorn server.api:app --host 127.0.0.1 --port 8500
```

## Docker deployment

```shell
docker build -t aiknowledge .
```

```shell
docker compose up -d
```
