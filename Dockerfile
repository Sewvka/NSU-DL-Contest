FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install python, pip, and build tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and set high timeout to prevent ReadTimeoutError
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies with increased timeout and retries
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --default-timeout=1000 \
    --retries 5 \
    -r requirements.txt

# Copy project files
WORKDIR /app
COPY . .

# Set permissions for the script
RUN chmod +x run_pipeline.sh

# Default command: run the whole pipeline
CMD ["./run_pipeline.sh"]
