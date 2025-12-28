from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi import FastAPI
from prompt_request_model import PromptRequest
from prompt_response_model import PromptResponse
import time
import gpu_stats
import threading

app = FastAPI()
model = "meta-llama/CodeLlama-13b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
llm = AutoModelForCausalLM.from_pretrained(
    model,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)


@app.post("/generate")
def generate_code(request: PromptRequest):
    usage_log = []
    state = {"monitoring": True}

    def monitor(interval=0.2):
        while state["monitoring"]:
            usage_log.append(gpu_stats.get_gpu_stats())
            time.sleep(interval)

    # Start GPU monitoring
    t = threading.Thread(target=monitor, daemon=True)
    t.start()

    try:
        start = time.perf_counter()
        formatted_prompt = f"<s>[INST]\n{request.prompt.strip()}\n[/INST]"
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(llm.device)
        input_length = inputs["input_ids"].shape[-1]
        max_context_length = 4096
        remaining_length = max_context_length - input_length
        max_new_tokens = min(512, max(128, remaining_length - 64))

        with torch.no_grad():
            outputs = llm.generate(
                **inputs,
                do_sample=True,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                min_new_tokens=100,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.9,
            )

        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        result = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if not result.strip():
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        end = time.perf_counter()
        del outputs
        del inputs
        torch.cuda.empty_cache()

        execution_time = end - start
        num_tokens = len(gen_ids)
        tpms = num_tokens / execution_time

    finally:
        state["monitoring"] = False
        t.join()

    if not usage_log:
        usage_log.append(gpu_stats.get_gpu_stats())

    gpu_util_avg = sum([s['gpu_util_percent'] for s in usage_log]) / len(usage_log)
    mem_util_avg = sum([s['memory_util_percent'] for s in usage_log]) / len(usage_log)
    mem_used_avg = sum([s['memory_used_gb'] for s in usage_log]) / len(usage_log)

    return PromptResponse(
        result=result,
        inference_time=round(execution_time, 1),
        token_throughput=round(tpms, 1),
        num_tokens=round(num_tokens, 1),
        gpu_util=round(gpu_util_avg, 1),
        mem_util=round(mem_util_avg, 1),
        mem_used=round(mem_used_avg, 1),
        total_mem=round(usage_log[0]["memory_total_gb"], 1)
    )