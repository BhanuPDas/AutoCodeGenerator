#FROM python:3.12.8
FROM python:3.11
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN python -c 'from huggingface_hub import login; import os; login(token=os.getenv("HF_TOKEN"))'
COPY . .
EXPOSE 5050
#CMD ["uvicorn", "inference_34b:app", "--host", "0.0.0.0", "--port", "5050"]
CMD ["uvicorn", "inference_34b_gptq:app", "--host", "0.0.0.0", "--port", "5050"]
