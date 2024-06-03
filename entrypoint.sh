#!/bin/bash

# Start the FastAPI application
uvicorn server.api:app --host 0.0.0.0 --port 8500 &

# Start the Streamlit application
streamlit run app.py --server.address=0.0.0.0 --server.port=8501