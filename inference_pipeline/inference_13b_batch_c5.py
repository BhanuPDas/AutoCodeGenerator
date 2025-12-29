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
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
rd = redis.Redis(host='172.17.0.1', port=6379, decode_responses=True)

# ----------------------------
# GLOBAL BATCH QUEUE
# ----------------------------
BATCH_SIZE = 4
REQUEST_QUEUE: queue.Queue = queue.Queue()
SHUTDOWN = False


# A wrapper object for queued requests
class BatchItem:
    def __init__(self, request: PromptRequest):
        self.request = request
        self.future = asyncio.get_event_loop().create_future()
        self.usage_log = []
        self.start_time = time.perf_counter()


# -----------------------------------------
# GPU Monitoring thread (per request)
# -----------------------------------------
def start_gpu_monitor(batch_item: BatchItem):
    state = {"monitoring": True}

    def monitor_thread():
        while state["monitoring"]:
            batch_item.usage_log.append(gpu_stats.get_gpu_stats())
            time.sleep(0.01)

    t = threading.Thread(target=monitor_thread, daemon=True)
    t.start()
    return state, t


# -----------------------------------------
# BATCH WORKER THREAD
# -----------------------------------------
def batch_worker():
    while not SHUTDOWN:
        items = []
        try:
            # Wait for 1 request
            first = REQUEST_QUEUE.get(timeout=0.1)
        except queue.Empty:
            continue

        items.append(first)

        # Try to fill up to batch size
        while len(items) < BATCH_SIZE:
            try:
                items.append(REQUEST_QUEUE.get_nowait())
            except queue.Empty:
                break

        # -------------------------
        # Prepare batch prompts
        # -------------------------
        prompts = [f"Task: {item.request.prompt.strip()}" for item in items]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(llm.device)
        input_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)

        max_context = 4096
        max_new_tokens = min([
            max(200, int((max_context - int(l))*0.9)) for l in input_lengths
        ])

        # -------------------------
        # Run batched inference
        # -------------------------
        outputs = llm.generate(
            **inputs,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )

        # -------------------------
        # Decode and return results
        # -------------------------
        for i, item in enumerate(items):
            state, t = item.monitor_state, item.monitor_thread
            state["monitoring"] = False
            t.join()

            output_ids = outputs[i]
            result = tokenizer.decode(output_ids, skip_special_tokens=True)

            end = time.perf_counter()
            exec_time = end - item.start_time

            input_len = input_lengths[i].item()
            gen_ids = output_ids[input_len:]
            num_tokens = len(gen_ids)
            tpms = num_tokens / exec_time

            usage_log = item.usage_log
            if not usage_log:
                usage_log.append(gpu_stats.get_gpu_stats())

            gpu_util = sum(s['gpu_util_percent'] for s in usage_log) / len(usage_log)
            mem_util = sum(s['memory_util_percent'] for s in usage_log) / len(usage_log)
            mem_used = sum(s['memory_used_gb'] for s in usage_log) / len(usage_log)
            total_mem = usage_log[0]['memory_total_gb']

            response = PromptResponse(
                result=result,
                inference_time=round(exec_time, 1),
                token_throughput=round(tpms, 1),
                num_tokens=num_tokens,
                gpu_util=round(gpu_util, 1),
                mem_util=round(mem_util, 1),
                mem_used=round(mem_used, 1),
                total_mem=round(total_mem, 1)
            )

            # Fulfill the request future
            item.future.set_result(response)


# Start batch worker thread
worker_thread = threading.Thread(target=batch_worker, daemon=True)
worker_thread.start()


# -----------------------------------------
# FastAPI Endpoint
# -----------------------------------------
@app.post("/generate")
async def generate_code(request: PromptRequest):
    batch_item = BatchItem(request)

    # Attach GPU monitor
    state, thread = start_gpu_monitor(batch_item)
    batch_item.monitor_state = state
    batch_item.monitor_thread = thread

    # Enqueue item
    REQUEST_QUEUE.put(batch_item)

    # Wait for batch worker to generate the output
    response: PromptResponse = await batch_item.future
    return response
