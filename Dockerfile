FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Устанавливаем python 3.10
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Копируем пакеты
COPY packages/ /app/packages/

# Устанавливаем всё содержимое папки напрямую (это надежнее, чем по списку)
# Мы просто говорим pip: "поставь всё, что найдешь в этой папке"
RUN pip install --no-cache-dir --no-index --find-links=/app/packages /app/packages/*.whl

# Теперь копируем остальной код
COPY . .
RUN chmod +x run_pipeline.sh

CMD ["./run_pipeline.sh"]
