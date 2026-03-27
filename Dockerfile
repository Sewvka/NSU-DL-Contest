FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install python, pip, and curl
RUN apt-get update && apt-get install -y python3 python3-pip python3-dev build-essential && \
    ln -s /usr/bin/python3 /usr/bin/python

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
WORKDIR /app
COPY . .

# Set permissions for the script
RUN chmod +x run_pipeline.sh

# Default command: run the whole pipeline
CMD ["./run_pipeline.sh"]
