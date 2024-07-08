# aiKnowLedge

aiKnowLedge is a project for managing a knowledge base and a QA chatbot, designed to help users efficiently manage and utilize their knowledge base.

## Technology Stack

- **[Streamlit](https://streamlit.io/)**: Web UI
- **[FastAPI](https://fastapi.tiangolo.com/)**: Backend server
- **[Qdrant](https://qdrant.tech/)**: Vector Database for storing and retrieving knowledge files
- **[MongoDB](https://www.mongodb.com/)**: Database for storing knowledge files' metadata


## Features

### QA Chatbot

1. **Answer Questions Based on Knowledge Base (Imported to Vector Database)**:
   - Supports customizable parameters [model name, temperature, stream, top k, similarity threshold, additional context length].
   - Model Name: LLM model name (e.g., gpt-3.5-turbo, gpt-4-turbo, gpt-4o).
   - Temperature: LLM temperature setting, controlling the creativity of the answers.
   - Stream: Real-time streaming output of answers.
   - Top k: Top k chunks retrieved from the vector database.
   - Similarity Threshold: Lower bound for similarity in QA retrieval.
   - Additional Context Length: Extension word length of the retrieved document chunk.

2. **Display User Question Intention and Retrieved Document Chunks**:
   - Display the analysis of user question's keywords and intentions.
   - Display top k retrieved chunks, including original text(additional context), page_id, chunk_id, and similarity score.
   - Provide detailed retrieval results to help users understand and verify the source of answers.

### Knowledge Base Management

1. **View Knowledge Base**:
   - Browse and manage existing knowledge files.
   
2. **Add New File to Knowledge Base**:
   - Supports customizable parameters [chunk size, overlap size, separators].
   - Flexible file chunking and overlap settings for efficient organization and retrieval.
   
3. **Delete File from Knowledge Base**:
   - Easily remove unnecessary files, keeping the knowledge base clean and organized.
   
4. **Add File to Vector Database (for QA Chatbot Retrieval)**:
   - Index knowledge files into the vector database for efficient QA retrieval.
   
5. **Delete File from Vector Database**:
   - Remove no longer needed indexed files, optimizing vector database storage and retrieval performance.

### Knowledge Point & Question Generating

Upload a paragraph to generate questions and answers.

## Run Locally

1. Create and activate a Conda environment:
   ```shell
   conda env create -n aiknowledge python=3.11.4
   conda activate aiknowledge
   ```
   
2. Install dependencies:
   ```shell
   pip install -r requirements.txt
   ```

3. Start frontend (Streamlit):
   ```shell
   streamlit run app.py --backend.address=127.0.0.1 --backend.port=8501
   ```

4. Start backend (FastAPI):
   ```shell
   uvicorn backend.api:app --host 127.0.0.1 --port 8500
   ```

## Docker deployment
1. Build Docker image:
   ```shell
   docker build -t aiknowledge .
   ```

2. Start Docker containers:
   ```shell
   docker compose up -d
   ```

## Contact Us

If you have any questions or suggestions, feel free to contact us: [aiknow2023@gmail.com](mailto:aiknow2023@gmail.com)
