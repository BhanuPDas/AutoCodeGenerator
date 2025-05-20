from transformers import AutoTokenizer
import transformers
import torch
from fastapi import FastAPI, HTTPException
from prompt_request_model import PromptRequest
from prompt_response_model import PromptResponse
import time

app = FastAPI()
model = "meta-llama/CodeLlama-13b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
generator = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

@app.post("/generate")
def generate_code(request: PromptRequest):
    try:
        start = time.perf_counter()
        formatted_prompt = f"[INST] {request.prompt.strip()} [/INST]"
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
        input_length = input_ids.shape[-1]
        max_context_length = 4096
        remaining_length = max_context_length - input_length
        output_length = min(800, int(remaining_length * 0.9))
        sequences = generator(
            formatted_prompt,
            do_sample=True,
            top_k=10,
            temperature=0.1,
            top_p=0.95,
             num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=input_length + output_length,
        )
        end = time.perf_counter()
        execution_time = (end - start) * 1000
        return PromptResponse(result=sequences[0]["generated_text"],inference_time=round(execution_time, 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

