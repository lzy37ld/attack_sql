from dataclasses import dataclass
from typing import Optional
from rlprompt.models import BaseModel, LMAdaptorModel, SinglePromptModel, InputConditionedPromptModel,MyModel,MyModel_Debug
from transformers import AutoConfig, AutoTokenizer
import torch


def check_torch_dtype(config):
    kwargs = {}
    if config.torch_dtype == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    return kwargs

def set_pad_token(t):
    if t.pad_token is None:
        t.pad_token = t.eos_token
    
    return t


def make_my_base_lm(config,mlp_state_dict):
    if config.lzy.debug:
        my_config = AutoConfig.from_pretrained(config.base_lm.model_name)
        my_config.mlp_hidden_size = 100
        kwargs = check_torch_dtype(config.base_lm)
        model = MyModel_Debug.from_pretrained(config.base_lm.model_name,config = my_config,**kwargs)
        tokenizer = set_pad_token(AutoTokenizer.from_pretrained(config.base_lm.model_name))
        # 是否需要resize一下加入pad token呢？感觉不需要
    
    else:
        my_config = AutoConfig.from_pretrained(config.base_lm.model_name)
        my_config.mlp_hidden_size = config.base_lm.mlp_hidden_size
        kwargs = check_torch_dtype(config.base_lm)
        model = MyModel.from_pretrained(config.base_lm.model_name,config = my_config,**kwargs)
        tokenizer = set_pad_token(AutoTokenizer.from_pretrained(config.base_lm.model_name))
    if mlp_state_dict is not None:
        model.mlp.load_state_dict(mlp_state_dict)

    for param in model.lm_head.parameters():
        param.requires_grad = False
    for param in model.model.parameters():
        param.requires_grad = False    

    return model,tokenizer


# 这里可能考虑吧adaptor_lm的config要和prompt_lm的融合一下
def make_lm_adaptor_model(config, mlp_state_dict = None) -> LMAdaptorModel:
    base_lm,base_tokenizer = make_my_base_lm(config,mlp_state_dict)
    return LMAdaptorModel(base_lm,base_tokenizer,
                          config.adaptor_lm.logit_bias,
                          config.adaptor_lm.fluent,
                          config.adaptor_lm.fluent_top_k,
                          config.adaptor_lm.max_decoding_length,
                          config.adaptor_lm.eos_token_id),\
            base_tokenizer


def make_single_prompt_model(model,
                             config: "DictConfig") -> SinglePromptModel:
    return SinglePromptModel(model,
                             config.prompt_lm.prompt_length,
                             config.prompt_lm.source_train_reps,
                             config.prompt_lm.prompt_infer_batch_size,
                             config.prompt_lm.source_str)


def make_input_conditioned_prompt_model(model,
                                        config: "DictConfig") -> InputConditionedPromptModel:
    return InputConditionedPromptModel(model,
                                       config.prompt_lm.prompt_length,
                                       config.prompt_lm.source_train_reps,
                                       config.prompt_lm.source_infer_reps,
                                       config.prompt_lm.top_k,
                                       config.prompt_lm.top_p,
                                       config.prompt_lm.num_beams)


@dataclass
class LMAdaptorModelConfig:
    policy_lm: str = "distilgpt2"
    # Name of the backbone pretrained LM
    hidden_size: int = 2048
    # Dimension for the hidden state of the enclosed adaptor MLP
    logit_bias: float = 0.0
    # Added to all prompt token logits. Set negative value to encourage exploration.
    fluent: bool = False
    # if True, constrain tokens to be from those with top-k probability under
    # a GPT-2 model
    fluent_top_k: int = 20
    # k for top-k probability above
    max_decoding_length: int = 5
    # Max output token length for the model
    eos_token_id: Optional[int] = None
    # The end-of-sentence token id, set to None for fixed-length prompts


@dataclass
class SinglePromptModelConfig:
    prompt_length: int = 5
    source_train_reps: int = 8
    prompt_infer_batch_size: int = 8
    source_str: str = "<|endoftext|>"

    
@dataclass
class InputConditionedPromptModelConfig:
    prompt_length: int = 5
    source_train_reps: int = 1
    source_infer_reps: int = 1
