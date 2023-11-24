import torch
from torch import nn
import numpy as np
from typing import Optional, List, Dict, Union

from transformers import pipeline, AutoTokenizer

from .base_model import BaseModel
from .model_utils import _top_k_logits, _top_p_logits
from transformers import LlamaPreTrainedModel,LlamaModel,OPTPreTrainedModel,OPTModel


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



class MyModel_Debug(OPTPreTrainedModel):
    _keys_to_ignore_on_load_missing= ['lm_head.weight']
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = OPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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




class LMAdaptorModel(BaseModel):
    """Uses an MLP to modify the hidden states of an pre-trained LM

    The modified hidden state can then be passed into the original LM head
    to obtain output token logits. 
    
    Inspired by Houlsby et al. (2019): https://arxiv.org/abs/1902.00751
    """
    def __init__(
        self,
        # MLP-specific parameters
        model,
        tokenizer,
        logit_bias: float,
        fluent: bool,
        fluent_top_k: Optional[int],
        # Generation parameters
        max_decoding_length: int,
        eos_token_id: Optional[int]
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        for param in self.model.lm_head.parameters():
            param.requires_grad = False
        for param in self.model.model.parameters():
            param.requires_grad = False

        self.logit_bias = logit_bias
        self.fluent = fluent
        self.fluent_top_k = fluent_top_k
        self.max_decoding_length = max_decoding_length
        self.eos_token_id = eos_token_id


    def _mlp_forward(self, state: torch.Tensor) -> torch.Tensor:
        mlp_output = self.model.mlp(state)
        logits = self.model.lm_head(mlp_output)

        if self.fluent:
            lm_logits = self.model.lm_head(state)
            values, _ = torch.topk(lm_logits, k=self.fluent_top_k)
            min_values: torch.Tensor = values[:, -1].unsqueeze(-1)
            logits = torch.where(lm_logits < min_values,
                                 torch.full_like(logits, float('-inf')),
                                 logits)
        return logits

    def teacher_forcing(
        self,
        source_texts: List[str],
        sample_ids: torch.Tensor,
        device,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        state, past_key_values = self._get_generation_cache(source_texts,device = device)

        sample_logits = []
        for i in range(sample_ids.shape[-1]):
            logits = self._mlp_forward(state)
            logits = logits + self.logit_bias

            actions = sample_ids[:, i]
            tokens = [self.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]

            sample_logits.append(logits.unsqueeze(dim=1))
            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values,device = device)

        sample_logits = torch.cat(sample_logits, dim=1)
        output = dict(sample_logits=sample_logits,
                      sample_ids=sample_ids)
        return output

    def sample(
        self,
        source_texts: List[str],
        top_k: Optional[int],
        top_p: float,
        max_new_tokens: Optional[int],
        eos_token_id: Optional[int],
        device,
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        if eos_token_id is not None:
            raise NotImplementedError(
                "Only support fixed length prompt for now")

        state, past_key_values = self._get_generation_cache(source_texts,device = device)
        sample_tokens = [[] for _ in source_texts]
        sample_ids, sample_logits = [], []
        for i in range(max_new_tokens):
            logits = self._mlp_forward(state)  # [batch_size, vocab_size]
            logits = logits + self.logit_bias
            # print(logits[:, 4:].min().item(), logits.max().item())

            if top_k is not None:
                sampling_logits = _top_k_logits(logits, k=top_k)

            if top_p is not None:
                sampling_logits = _top_p_logits(logits, p=top_p)

            sampling_logits = logits

            # can not do bf 16
            # actions = (torch.distributions.categorical
            #            .Categorical(logits=sampling_logits)
            #            .sample())  # [batch_size]

            # from huggingface
            probs = nn.functional.softmax(sampling_logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1)
        
            tokens = [self.tokenizer.convert_ids_to_tokens([a])[0]
                    for a in actions.tolist()]
            token_strs = [self.tokenizer.convert_tokens_to_string([t])
                        for t in tokens]


            for s, t in zip(sample_tokens, tokens): 
                s.append(t)
            sample_ids.append(actions.unsqueeze(dim=1))  # [batch_size, 1]
            sample_logits.append(logits.unsqueeze(dim=1))
            # [batch_size, 1, vocab_size]

            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values,device = device)

        # [batch_size, prompt_length]
        sample_ids = torch.cat(sample_ids, dim=1)
        # [batch_size, prompt_length, vocab_size]
        sample_logits = torch.cat(sample_logits, dim=1)
        sample_lengths = (torch.tensor([max_new_tokens
                                        for _ in range(sample_ids.shape[0])])
                          .to(device))

        output = dict(sample_logits=sample_logits,
                      sample_ids=sample_ids,
                      sample_lengths=sample_lengths)
        return output

    def greedy_search(self,
                      source_texts: List[str],
                      max_new_tokens: Optional[int],
                      eos_token_id: Optional[int],
                      device,
                      **kwargs):
        if eos_token_id is not None:
            raise NotImplementedError(
                "Only support fixed length prompt for now")

        state, past_key_values = self._get_generation_cache(source_texts,device = device)
        sample_tokens = [[] for _ in source_texts]
        sample_ids, sample_logits = [], []
        for i in range(max_new_tokens):
            logits = self._mlp_forward(state)
            logits = logits + self.logit_bias

            actions = logits.argmax(dim=-1)  # [batch_size]
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]

            for s, t in zip(sample_tokens, tokens): 
                s.append(t)
            sample_ids.append(actions.unsqueeze(dim=1))
            sample_logits.append(logits.unsqueeze(dim=1))

            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values,device = device)

        sample_ids = torch.cat(sample_ids, dim=1)
        sample_logits = torch.cat(sample_logits, dim=1)
        sample_lengths = (torch.tensor([max_new_tokens
                                        for _ in range(sample_ids.shape[0])])
                          .to(device))

        output = dict(sample_tokens=sample_tokens,
                      sample_logits=sample_logits,
                      sample_ids=sample_ids,
                      sample_lengths=sample_lengths)
        return output


    def generate(
        self,
        source_texts: List[str],
        do_sample: bool,
        top_k: Optional[int],
        top_p: float,
        num_beams: int,
        device,
        max_new_tokens: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        assert num_beams == 1, "Beam search not supported yet"
        if max_new_tokens is None:
            max_new_tokens = self.max_decoding_length
        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        is_greedy_gen_mode = (do_sample == False) and (num_beams == 1)
        is_sample_gen_mode = (do_sample == True) and (num_beams == 1)
        assert is_greedy_gen_mode or is_sample_gen_mode

        if is_greedy_gen_mode:
            return self.greedy_search(source_texts=source_texts,
                                      max_new_tokens=max_new_tokens,
                                      eos_token_id=eos_token_id,
                                      device = device)
        elif is_sample_gen_mode:
            return self.sample(source_texts=source_texts,
                               top_k=top_k,
                               top_p=top_p,
                               max_new_tokens=max_new_tokens,
                               eos_token_id=eos_token_id,
                               device = device)

    def _get_generation_cache(self,
                              source_texts: List[str],
                              past_key_values=None, 
                              device = None):
        assert device is not None
        token_encoding = (self.tokenizer(source_texts,
                                     padding=True,
                                     truncation=True,
                                     return_tensors='pt').to(device))
        input_ids = token_encoding['input_ids']
        input_lengths = token_encoding['attention_mask'].sum(dim=1)
        outputs = self.model.model(input_ids,past_key_values=past_key_values,use_cache=True)

        last_token_hidden_state = \
            outputs.last_hidden_state[np.arange(input_ids.shape[0]),
                                      (input_lengths - 1)]
        past_key_values = outputs.past_key_values
        return last_token_hidden_state, past_key_values
