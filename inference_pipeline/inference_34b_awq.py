from vllm import LLM, SamplingParams
from fastapi import FastAPI
from prompt_request_model import PromptRequest
from prompt_response_model import PromptResponse
import time
import gpu_stats
import threading

app = FastAPI()
model = "TheBloke/CodeLlama-34B-Instruct-AWQ"
llm = LLM(model=model,
          quantization="awq_marlin",
          dtype="float16",
          max_seq_len=4096,
          seed=0)
sampling_params = SamplingParams(temperature=0.7,
                                 top_p=0.95,
                                 max_tokens=512)

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
        prompts = [formatted_prompt]
        outputs = llm.generate(prompts, sampling_params)

        result = outputs[0].outputs[0].text
        end = time.perf_counter()
        del outputs

        execution_time = end - start
        num_tokens = len(result.split())
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