#!/bin/bash

#uvicorn aiknowledge.backend.api:app --host 127.0.0.1 --port 8500 &
#exec streamlit run aiknowledge/app.py --backend.address=0.0.0.0 --backend.port=8501 &
#wait

#uvicorn aiknowledge.backend.api:app --host 127.0.0.1 --port 8500 &
#exec streamlit run aiknowledge/webui.py --server.address=127.0.0.1 --server.port=8501 &
#wait

streamlit run aiknowledge/app.py --server.address=0.0.0.0 --server.port=8501 &
wait