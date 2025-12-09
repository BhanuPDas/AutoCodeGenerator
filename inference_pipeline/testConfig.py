from transformers import AutoConfig

config = AutoConfig.from_pretrained("meta-llama/CodeLlama-34b-Instruct-hf")
print(config.max_position_embeddings)
print(config.rope_scaling)
