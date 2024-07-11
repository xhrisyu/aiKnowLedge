## Use an official Python runtime as a parent image
#ARG PYTHON_VERSION=3.11.4
#FROM python:${PYTHON_VERSION}-slim as base
#
## Set environment variables to prevent Python from generating .pyc files
#ENV PYTHONDONTWRITEBYTECODE=1
#
## Set environment variable to prevent Python from buffering stdout and stderr
#ENV PYTHONUNBUFFERED=1
#
## Set the working directory in the container
#WORKDIR /aiknowledge
#
## Install system dependencies
#RUN apt-get update && apt-get install -y \
#    pkg-config \
#    libhdf5-dev \
#    build-essential \
#    && apt-get clean \
#    && rm -rf /var/lib/apt/lists/*
#
## Upgrade pip
#RUN pip install --upgrade pip --progress-bar off
#
## Copy only the requirements.txt initially to leverage Docker cache
#COPY requirements.txt ./
#RUN pip install --no-cache-dir -r requirements.txt --progress-bar off
#
## Copy the rest of the application
#COPY run.sh /aiknowledge/run.sh
#COPY . .
#
## Expose the port the Streamlit app runs on
#EXPOSE 8501
#
## Make the entrypoint script executable, Run the entrypoint script
#RUN chmod +x /aiknowledge/run.sh
#CMD ["bash", "/aiknowledge/run.sh"]

# Use an official Python runtime as a parent image
ARG PYTHON_VERSION=3.11.4
FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables to prevent Python from generating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Set environment variable to prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Set the PYTHONPATH environment variable
ENV PYTHONPATH="/aiknowledge:$PYTHONPATH"

# Set the working directory in the container
WORKDIR /aiknowledge

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip --progress-bar off

# Copy only the requirements.txt initially to leverage Docker cache
COPY requirements.txt /aiknowledge/requirements.txt
RUN pip install --no-cache-dir -r /aiknowledge/requirements.txt --progress-bar off

# Copy the rest of the application
COPY run.sh /aiknowledge/run.sh
COPY aiknowledge /aiknowledge/aiknowledge

# Expose the port the Streamlit app runs on
EXPOSE 8501

# Make the entrypoint script executable, Run the entrypoint script
RUN chmod +x /aiknowledge/run.sh
CMD ["bash", "/aiknowledge/run.sh"]


# Run the application
#CMD uvicorn backend.api:app --host 127.0.0.1 --port 8500
#CMD streamlit run app.py --backend.address=0.0.0.0 --backend.port=8501
#CMD ["streamlit", "run", "app.py", "--backend.port=8501", "--backend.address=0.0.0.0"]
#CMD ["sh", "-c", "streamlit run app.py --backend.address=0.0.0.0 --backend.port=8501 & uvicorn backend.api:app --host 127.0.0.1 --port 8500"]
