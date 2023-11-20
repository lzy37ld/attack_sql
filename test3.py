import torch
from transformers import file_utils

# Check using PyTorch
pytorch_support = torch.cuda.is_bf16_supported()

# Check using Hugging Face Transformers
transformers_support = file_utils.is_torch_bf16_available()

print(f"PyTorch BF16 support: {pytorch_support}")
print(f"Hugging Face Transformers BF16 support: {transformers_support}")
