
from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
t.save_pretrained("/home/liao.629/rl-prompt-lzy/examples/text-style-transfer/ckpt/trainer.training_mode=sql-offpolicy")