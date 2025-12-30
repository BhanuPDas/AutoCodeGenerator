import time
import torch
import threading
import queue
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_request_model import PromptRequest
from prompt_response_model import PromptResponse
import gpu_stats
import asyncio
import redis
import hashlib

app = FastAPI()

MODEL_NAME = "meta-llama/CodeLlama-13b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
rd = redis.Redis(host="172.17.0.1",port=6379,decode_responses=True)

CACHE_PREFIX = "llm:"


def cache_key(prompt: str) -> str:
    payload = f"{prompt.strip()}"
    return CACHE_PREFIX + hashlib.sha256(payload.encode()).hexdigest()

BATCH_SIZE = 6
REQUEST_QUEUE: queue.Queue = queue.Queue()
SHUTDOWN = False


class BatchItem:
    def __init__(self, request: PromptRequest, loop: asyncio.AbstractEventLoop):
        self.request = request
        self.future = loop.create_future()
        self.usage_log = []
        self.start_time = time.perf_counter()
        self.cache_key = cache_key(request.prompt)

def start_gpu_monitor(batch_item: BatchItem):
    state = {"monitoring": True}

    def monitor_thread():
        while state["monitoring"]:
            batch_item.usage_log.append(gpu_stats.get_gpu_stats())
            time.sleep(0.2)

    t = threading.Thread(target=monitor_thread, daemon=True)
    t.start()
    return state, t

def batch_worker():
    while not SHUTDOWN:
        items = []
        try:
            first = REQUEST_QUEUE.get(timeout=0.1)
        except queue.Empty:
            continue

        items.append(first)

        while len(items) < BATCH_SIZE:
            try:
                items.append(REQUEST_QUEUE.get_nowait())
            except queue.Empty:
                break

        prompts = [f"<s>[INST]\n{item.request.prompt.strip()}\n[/INST]" for item in items]
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(llm.device)
        input_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)

        max_context = 4096
        max_new_tokens = min(
            min(512, max(128, (max_context - int(l) - 64)))
            for l in input_lengths
        )
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

        for i, item in enumerate(items):
            item.monitor_state["monitoring"] = False
            item.monitor_thread.join()

            output_ids = outputs[i]
            input_len = input_lengths[i].item()
            gen_ids = output_ids[input_len:]
            result = tokenizer.decode(gen_ids, skip_special_tokens=True)

            end = time.perf_counter()
            exec_time = end - item.start_time
            num_tokens = len(gen_ids)
            tpms = num_tokens / exec_time

            usage_log = item.usage_log or [gpu_stats.get_gpu_stats()]
            gpu_util = sum(s['gpu_util_percent'] for s in usage_log) / len(usage_log)
            mem_util = sum(s['memory_util_percent'] for s in usage_log) / len(usage_log)
            mem_used = sum(s['memory_used_gb'] for s in usage_log) / len(usage_log)
            total_mem = usage_log[0]['memory_total_gb']
            rd.set(item.cache_key, result)
            response = PromptResponse(
                result=result,
                inference_time=round(exec_time, 1),
                token_throughput=round(tpms, 1),
                num_tokens=num_tokens,
                gpu_util=round(gpu_util, 1),
                mem_util=round(mem_util, 1),
                mem_used=round(mem_used, 1),
                total_mem=round(total_mem, 1),
            )

            item.future.set_result(response)


worker_thread = threading.Thread(target=batch_worker, daemon=True)
worker_thread.start()


@app.post("/generate")
async def generate_code(request: PromptRequest):
    key = cache_key(request.prompt)
    st = time.perf_counter()
    cached = rd.get(key)
    if cached:
        et = time.perf_counter()
        return PromptResponse(
            result=cached,
            inference_time=round((et-st), 1),
            token_throughput=0.0,
            num_tokens=0.0,
            gpu_util=0.0,
            mem_util=0.0,
            mem_used=27.1,
            total_mem=93.6,
        )
    loop = asyncio.get_running_loop()
    batch_item = BatchItem(request, loop)
    state, thread = start_gpu_monitor(batch_item)
    batch_item.monitor_state = state
    batch_item.monitor_thread = thread

    REQUEST_QUEUE.put(batch_item)
    response: PromptResponse = await batch_item.future
    return response
