from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/CodeLlama-13b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'write java code using springboot to receive astring message and print it.',
    do_sample=True,
    top_k=10,
    temperature=0.1,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=1000,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
