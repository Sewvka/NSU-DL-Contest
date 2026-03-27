FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Используем локальные пакеты, интернет не нужен
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Копируем список и заранее скачанные пакеты
COPY requirements.txt .
COPY packages/ ./packages/

# Установка СТРОГО из локальной папки (--no-index запрещает лезть в сеть)
RUN pip install --no-index --find-links=./packages -r requirements.txt

# Копируем остальной код
COPY . .
RUN chmod +x run_pipeline.sh

CMD ["./run_pipeline.sh"]
