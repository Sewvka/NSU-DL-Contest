FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Use bash for better control
SHELL ["/bin/bash", "-c"]

# Minimal system setup
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Configure pip for maximum stability
# 1. High timeout
# 2. No cache (prevents some SSL issues)
# 3. Disable IPv6 (crucial for some networks)
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_NO_CACHE_DIR=1

RUN pip install --upgrade pip

# Install dependencies one by one to pinpoint issues
COPY requirements.txt .
RUN pip install certifi && \
    pip install numpy==2.4.3 && \
    pip install pandas==3.0.1 && \
    pip install polars==1.39.3 && \
    pip install catboost==1.2.10 && \
    pip install -r requirements.txt

WORKDIR /app
COPY . .
RUN chmod +x run_pipeline.sh

CMD ["./run_pipeline.sh"]
