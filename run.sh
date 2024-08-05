#!/bin/bash

#uvicorn aiknowledge.backend.api:app --host 127.0.0.1 --port 8500 &
#exec streamlit run aiknowledge/app.py --backend.address=0.0.0.0 --backend.port=8501 &
#wait

streamlit run aiknowledge/app.py --server.maxMessageSize 512 --server.maxUploadSize 512 --server.address=0.0.0.0 --server.port=8501 &
wait
