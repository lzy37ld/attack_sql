import torch
from torch import nn
import numpy as np
from typing import Optional, List, Dict, Union

from transformers import pipeline, AutoTokenizer

from transformers import LlamaPreTrainedModel,LlamaModel,OPTPreTrainedModel,OPTModel,AutoConfig


def _build_one_layer_mlp(in_dim, out_dim, hidden_size):
    W1 = nn.Linear(in_dim, hidden_size)
    A1 = nn.ReLU()
    W2 = nn.Linear(hidden_size, out_dim)
    return nn.Sequential(W1, A1, W2)



class MyModel(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing= ['lm_head.weight']
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config
        print(config)
        self.mlp = _build_one_layer_mlp(in_dim=config.hidden_size,
                                        out_dim=config.hidden_size,
                                        hidden_size=config.mlp_hidden_size).to(dtype=torch.bfloat16)
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.0001)
                m.bias.data.fill_(-0.0001)
        self.mlp.apply(_init_weights)

        self.post_init()


# x = MyModel.from_pretrained("/home/liao.629/rl-prompt-lzy/examples/text-style-transfer/ckpt/trainer.training_mode=sql-mixed|trainer.mix_strategy=mix|trainer.margin_constant=5|trainer.margin_coefficient=3")

AutoConfig.from_pretrained("/home/liao.629/rl-prompt-lzy/examples/text-style-transfer/ckpt/trainer.training_mode=sql-mixed|trainer.mix_strategy=mix|trainer.margin_constant=5|trainer.margin_coefficient=3")