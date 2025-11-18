from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi import FastAPI, HTTPException
from prompt_request_model import PromptRequest
from prompt_response_model import PromptResponse
import time

app = FastAPI()
model = "TheBloke/CodeLlama-13B-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model)
llm = AutoModelForCausalLM.from_pretrained(
    model,
    torch_dtype=torch.float16,
    device_map="auto"
)


@app.post("/generate")
def generate_code(request: PromptRequest):
    try:
        start = time.perf_counter()
        formatted_prompt = f"[INST] {request.prompt.strip()} [/INST]"
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(llm.device)
        input_length = inputs["input_ids"].shape[-1]
        max_context_length = 4096
        remaining_length = max_context_length - input_length
        max_new_tokens = max(200, int(remaining_length * 0.9))

        outputs = llm.generate(
            **inputs,
            do_sample=False,  # Change to True for Sampling runs
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            # temperature=0.1,
            # top_p=0.9,
            # top_k=50,
        )
        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        end = time.perf_counter()
        execution_time = (end - start) * 1000
        num_tokens = len(gen_ids)

        # token per ms
        tpms = num_tokens / execution_time

        return PromptResponse(
            result=result,
            inference_time=round(execution_time, 1),
            token_throughput=round(tpms, 1),
            num_tokens=round(num_tokens, 1)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
