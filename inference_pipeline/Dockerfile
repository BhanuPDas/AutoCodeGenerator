FROM python:3.12.8
WORKDIR /app

# -------------------------
#  Install system dependencies
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
#  Install PyTorch with CUDA support
#  (For NVIDIA GPUs inside Docker)
# -------------------------
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade huggingface_hub
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN python -c 'from huggingface_hub import login; login(token=os.getenv("HF_TOKEN"))'

COPY . .

EXPOSE 5050

CMD ["uvicorn", "inference_13b:app", "--host", "0.0.0.0", "--port", "5050"]
