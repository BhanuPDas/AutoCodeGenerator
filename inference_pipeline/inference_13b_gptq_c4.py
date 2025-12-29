from fastapi import FastAPI
from prompt_request_model import PromptRequest
from prompt_response_model import PromptResponse
import time
import gpu_stats
import threading
from gptqmodel import GPTQModel
import redis
import hashlib

app = FastAPI()
model = GPTQModel.load("TheBloke/CodeLlama-13B-Instruct-GPTQ",
                       revision="gptq-4bit-64g-actorder_True",
                       device="cuda")
rd = redis.Redis(host='172.17.0.1', port=6379, decode_responses=True)

def gen_hash_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()

@app.post("/generate")
def generate_code(request: PromptRequest):
    usage_log = []
    state = {"monitoring": True}
    t = None

    def monitor(interval=0.2):
        while state["monitoring"]:
            usage_log.append(gpu_stats.get_gpu_stats())
            time.sleep(interval)

    try:
        cache_key = f"llm:{gen_hash_key(request.prompt.strip())}"
        start = time.perf_counter()
        formatted_prompt = f"<s>[INST]\n{request.prompt.strip()}\n[/INST]"
        inputs = model.tokenizer(formatted_prompt, return_tensors='pt')
        input_length = inputs["input_ids"].shape[-1]
        max_context_length = 4096
        remaining_length = max_context_length - input_length
        max_new_tokens = min(512, max(128, remaining_length - 64))
        cached_res = rd.get(cache_key)
        if cached_res:
            state["monitoring"] = False
            output_text = cached_res
            end = time.perf_counter()
            execution_time = end - start
            return PromptResponse(
                result=output_text,
                inference_time=round(execution_time, 1),
                token_throughput=0.0,
                num_tokens=0.0,
                gpu_util=0.0,
                mem_util=0.0,
                mem_used=9.0,
                total_mem=93.6
            )
        else:
            t = threading.Thread(target=monitor, daemon=True)
            t.start()
            result = model.generate(formatted_prompt,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=0.1,
                                top_p=0.9,
                                repetition_penalty=1.1
                                )[0]
            output_text = model.tokenizer.decode(result, skip_special_tokens=True)
            rd.set(cache_key, output_text)
        end = time.perf_counter()
        execution_time = end - start
        num_tokens = len(result) - input_length
        tpms = num_tokens / execution_time

    finally:
        state["monitoring"] = False
        if t is not None:
            t.join()

    if not usage_log:
        usage_log.append(gpu_stats.get_gpu_stats())

    gpu_util_avg = sum([s['gpu_util_percent'] for s in usage_log]) / len(usage_log)
    mem_util_avg = sum([s['memory_util_percent'] for s in usage_log]) / len(usage_log)
    mem_used_avg = sum([s['memory_used_gb'] for s in usage_log]) / len(usage_log)

    return PromptResponse(
        result=output_text,
        inference_time=round(execution_time, 1),
        token_throughput=round(tpms, 1),
        num_tokens=round(num_tokens, 1),
        gpu_util=round(gpu_util_avg, 1),
        mem_util=round(mem_util_avg, 1),
        mem_used=round(mem_used_avg, 1),
        total_mem=round(usage_log[0]["memory_total_gb"], 1)
    )
