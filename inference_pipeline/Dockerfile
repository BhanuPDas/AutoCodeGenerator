#FROM python:3.12.8
FROM python:3.11
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*
#RUN pip install --no-cache-dir setuptools==70.0.0 wheel packaging
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -v gptqmodel --no-build-isolation
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN python -c 'from huggingface_hub import login; import os; login(token=os.getenv("HF_TOKEN"))'
COPY . .
EXPOSE 5050
#CMD ["uvicorn", "inference_34b:app", "--host", "0.0.0.0", "--port", "5050"]
CMD ["uvicorn", "inference_34b_gptq:app", "--host", "0.0.0.0", "--port", "5050"]
