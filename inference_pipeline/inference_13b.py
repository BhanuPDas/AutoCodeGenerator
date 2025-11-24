from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi import FastAPI, HTTPException
from prompt_request_model import PromptRequest
from prompt_response_model import PromptResponse
import time
import gpu_stats
import threading

app = FastAPI()
model = "meta-llama/CodeLlama-13b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token="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")
llm = AutoModelForCausalLM.from_pretrained(
    model,
    torch_dtype=torch.float16,
    device_map="auto"
)


def measure_request_gpu(func):
    """
    Decorator to measure GPU utilization per request
    and return it alongside inference result.
    """

    def wrapper(*args, **kwargs):
        usage_log = []
        state = {"monitoring": True}

        def monitor(interval=0.01):
            while state["monitoring"]:
                usage_log.append(gpu_stats.get_gpu_stats())
                time.sleep(interval)

        # Start GPU monitoring
        t = threading.Thread(target=monitor, daemon=True)
        t.start()

        # Call the actual inference function
        result, execution_time, tpms, num_tokens = func(*args, **kwargs)

        # Stop monitoring
        state["monitoring"] = False
        t.join()
        if not usage_log:
            usage_log.append(gpu_stats.get_gpu_stats())

        # Compute GPU stats for this request
        gpu_util_avg = sum([s['gpu_util_percent'] for s in usage_log]) / len(usage_log)
        mem_util_avg = sum([s['memory_util_percent'] for s in usage_log]) / len(usage_log)
        mem_used_avg = sum([s['memory_used_gb'] for s in usage_log]) / len(usage_log)

        # Return result + GPU stats
        return PromptResponse(
            result=result,
            inference_time=execution_time,
            token_throughput=tpms,
            num_tokens=num_tokens,
            gpu_util=round(gpu_util_avg, 1),
            mem_util=round(mem_util_avg, 1),
            mem_used=round(mem_used_avg, 1),
            total_mem=round(usage_log[0]["memory_total_gb"], 1)
        )

    return wrapper


@app.post("/generate")
@measure_request_gpu
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
        execution_time = (end - start)
        num_tokens = len(gen_ids)

        # token per ms
        tpms = num_tokens / execution_time

        return result, round(execution_time, 1), round(tpms, 1), round(num_tokens, 1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
