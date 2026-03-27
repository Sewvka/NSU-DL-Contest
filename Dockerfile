FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Fix SSL and installation issues
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Upgrade certificates and pip
RUN update-ca-certificates && \
    pip install --no-cache-dir --upgrade pip

# Install dependencies with bypassed SSL checks
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --default-timeout=1000 \
    --retries 5 \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Copy project files
WORKDIR /app
COPY . .

# Set permissions for the script
RUN chmod +x run_pipeline.sh

# Default command: run the whole pipeline
CMD ["./run_pipeline.sh"]
