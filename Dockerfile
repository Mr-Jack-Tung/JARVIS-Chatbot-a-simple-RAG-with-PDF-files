# Stage 1: Install PyPI dependencies only
FROM --platform=linux/amd64 python:3.12-slim AS deps
WORKDIR /tmp/deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel cython uv

# Copy only lockfiles, no local package
COPY pylock.toml uv.lock ./

# Install PyPI dependencies only
RUN uv pip install --requirements pylock.toml --preview-features pylock --system

# Stage 2: Final image
FROM --platform=linux/amd64 python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libopenblas-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy venv/dependencies from deps stage
COPY --from=deps /usr/local /usr/local

# Copy application code
COPY . .

# Install local package editable
RUN pip install --no-cache-dir -e .

CMD ["python3", "main.py"]
