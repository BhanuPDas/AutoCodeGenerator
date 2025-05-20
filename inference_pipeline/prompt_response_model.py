from pydantic import BaseModel

class PromptResponse(BaseModel):
    result: str
    inference_time: float