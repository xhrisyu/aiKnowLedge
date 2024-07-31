# Use an official Python runtime as a parent image
ARG PYTHON_VERSION=3.10.14
#ARG PYTHON_VERSION=3.11.4
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
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install OpenJDK 21
RUN wget https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.tar.gz

RUN mkdir -p /usr/lib/jvm \
    && tar -xzf jdk-21_linux-x64_bin.tar.gz -C /usr/lib/jvm \
    && rm jdk-21_linux-x64_bin.tar.gz \
    && ln -s /usr/lib/jvm/jdk-21 /usr/lib/jvm/java-21-openjdk-amd64

# Set JAVA_HOME and JVM_PATH environment variables
#ENV JAVA_HOME="/usr/lib/jvm/java-21-openjdk-amd64"
#ENV JVM_PATH="/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so"
ENV JAVA_HOME="/usr/lib/jvm/jdk-21.0.4"
ENV JVM_PATH="/usr/lib/jvm/jdk-21.0.4/lib/server/libjvm.so"
ENV PATH="$JAVA_HOME/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip --progress-bar off

# Copy only the requirements.txt initially to leverage Docker cache
COPY requirements.txt /aiknowledge/requirements.txt
RUN pip install --no-cache-dir -r /aiknowledge/requirements.txt --progress-bar off

# Copy the rest of the application
COPY run.sh /aiknowledge/run.sh
COPY aiknowledge /aiknowledge/aiknowledge
COPY .streamlit /aiknowledge/.streamlit

# Expose the port the Streamlit app runs on
EXPOSE 8501

# Make the entrypoint script executable, Run the entrypoint script
RUN chmod +x /aiknowledge/run.sh
CMD ["bash", "/aiknowledge/run.sh"]
