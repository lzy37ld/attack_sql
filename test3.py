
from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
from pathlib import Path
Path("/home/liao.629/rl-prompt-lzy/examples/text-style-transfer/ckpt/trainer.training_mode=sql-offpolicy").mkdir(exist_ok= True,parents= True)
t.save_pretrained("/home/liao.629/rl-prompt-lzy/examples/text-style-transfer/ckpt/trainer.training_mode=sql-offpolicy")