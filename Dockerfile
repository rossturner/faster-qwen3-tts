FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive HF_HOME=/hf-cache PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3-pip python3.12-venv curl libsndfile1 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY faster_qwen3_tts ./faster_qwen3_tts
RUN python3.12 -m pip install --break-system-packages \
        torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130 && \
    python3.12 -m pip install --break-system-packages ".[server]"
EXPOSE 8092
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8092/health || exit 1
CMD ["faster-qwen3-tts", "serve-http", "--host", "0.0.0.0", "--port", "8092"]
