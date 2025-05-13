# Sử dụng hình ảnh Python chính thức
FROM python:3.12-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các dependencies hệ thống
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    netcat-openbsd \
    && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Sao chép các file cấu hình Poetry
COPY pyproject.toml ./

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

# Cài đặt Poetry
RUN pip install poetry

RUN poetry env use python3.12

# Cài đặt dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

RUN poetry lock
RUN poetry update

COPY . .

# Đặt biến môi trường
ENV PYTHONUNBUFFERED=1

# RUN groupadd -r appuser && useradd -r -g appuser appuser
# RUN chown -R appuser:appuser .
# USER appuser

# HEALTHCHECK --interval=5m --timeout=3s CMD curl -f http://localhost:8000/ || exit 1

CMD ["/bin/bash", "-c", "ollama serve & until nc -z localhost 11434; do echo 'Waiting for Ollama to start...'; sleep 5; done; ollama pull nomic-embed-text && ollama pull qwen3:4b && poetry run python3 main.py"]
