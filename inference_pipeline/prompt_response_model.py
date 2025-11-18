from pydantic import BaseModel


class PromptResponse(BaseModel):
    result: str
    inference_time: float
    token_throughput: float
    num_tokens: float
    gpu_util: float
    mem_util: float
    mem_used: float
    total_mem: float
