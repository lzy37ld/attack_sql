I tried to run the code from HF deepspeed integration page.

```

from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers.integrations import HfDeepSpeedConfig
import deepspeed
import os
import torch
import torch.distributed as dist

def which_process(idx) -> bool:
    """Check if the current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == idx

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

# distributed setup
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

model_name = "bigscience/T0_3B"

config = AutoConfig.from_pretrained(model_name)
model_hidden_size = config.d_model

# batch size has to be divisible by world_size, but can be bigger than world_size
train_batch_size = 1 * world_size

# ds_config notes
#
# - enable bf16 if you use Ampere or higher GPU - this will run in mixed precision and will be
# faster.
#
# - for older GPUs you can enable fp16, but it'll only work for non-bf16 pretrained models - e.g.
# all official t5 models are bf16-pretrained
#
# - set offload_param.device to "none" or completely remove the `offload_param` section if you don't
# - want CPU offload
#
# - if using `offload_param` you can manually finetune stage3_param_persistence_threshold to control
# - which params should remain on gpus - the larger the value the smaller the offload size
#
# For indepth info on Deepspeed config see
# https://huggingface.co/docs/transformers/main/main_classes/deepspeed

# keeping the same format as json for consistency, except it uses lower case for true/false
# fmt: off
ds_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}
# fmt: on

# next line instructs transformers to partition the model directly over multiple gpus using
# deepspeed.zero.Init when model's `from_pretrained` method is called.
#
# **it has to be run before loading the model AutoModelForSeq2SeqLM.from_pretrained(model_name)**
#
# otherwise the model will first be loaded normally and only partitioned at forward time which is
# less efficient and when there is little CPU RAM may fail
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

# now a model can be loaded.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# initialise Deepspeed ZeRO and store only the engine object
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()  # inference

dist.barrier()
if which_process(0):
    for name,param in ds_engine.named_parameters():
        print(name,param.device)
        print(name,param)


    for name,param in ds_engine.module.named_parameters():
        print(name,param.device)
        print(name,param)

    print("*"*10)
    print("*"*10)
    print("*"*10)
    print("*"*10)
    print("*"*10)
    print("*"*10)

dist.barrier()
if which_process(1):
    for name,param in ds_engine.named_parameters():
        print(name,param.device)
        print(name,param)

    for name,param in ds_engine.module.named_parameters():
        print(name,param.device)
        print(name,param)

    print("*"*10)
    print("*"*10)
    print("*"*10)
    print("*"*10)
    print("*"*10)
    print("*"*10)
```


I set debugger after .from_pretrained and print out the parameters. As you can see they are all empty tensor.
So my question is what itmes that actually occupy the VRAM spaces as shown by the figure below.

Why they are 7k and 22k, which in total is 30k which is larger than the size of T0-3B(3 * 4 = 12k)...
That's weird...


Also How could I ensure that they are actually partitioned across gpus?