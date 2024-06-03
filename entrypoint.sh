#!/bin/bash

streamlit run app.py --server.address=0.0.0.0 --server.port=8501 &
uvicorn server.api:app --host 127.0.0.1 --port 8500
