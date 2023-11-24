
from transformers import AutoTokenizer,GenerationConfig,AutoModelForCausalLM
from safe_rlhf.models import AutoModelForScore
import os
import torch.nn as nn
import torch
import itertools
from collections import defaultdict
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from rlprompt.losses import sql_loss_with_sparse_rewards
from rlprompt.models import (make_lm_adaptor_model, make_input_conditioned_prompt_model,check_torch_dtype)

import numpy as np

def pair_src_p_target(s,p,t):
    repeat = int(len(t)/len(s))
    l = []
    for i in range(len(s)):
        d = {}
        d["s"] = s[i]
        d['p'] = p[i]
        d["t"] = t[i * repeat: (i+1) * repeat]
        l.append(d)
    return l

# 为什么我不用deepspeed，本质上可以用，但是我一个gpu的vram太小了，所以多放几个model，不一定放得下，还不如分开。。。
def create_reward(config):
        
    if os.environ.get("RANK", "0") == "0":

        class RewardModel(nn.Module): 
            def __init__(
                self, 
                config,
            ): 
                super().__init__()
                model_name = config.model_name
                self.template = config.template
                self.batch_size = config.batch_size
                self.config = config
                kwargs = check_torch_dtype(config)
                if config.debug:
                    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                    tokenizer.padding_side = "left"
                else:
                    model = AutoModelForScore.from_pretrained(model_name, **kwargs)
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                    tokenizer.padding_side = "right"
                if not tokenizer.pad_token:
                    tokenizer.pad_token = tokenizer.eos_token
                self.model = model
                self.tokenizer= tokenizer



            def _reward_run(self, q_and_p_s, ans_s, device):
                outputs_l = []
                batch_size = self.batch_size
                for i in range(0,len(q_and_p_s),batch_size):
                    
                    batch_inputs = q_and_p_s[i: i +batch_size]
                    batch_outputs = ans_s[i: i +batch_size]
                    batch = [self.template.format(model_input = batch_inputs[index], model_output = batch_outputs[index]) for index in range(len(batch_inputs))]
                    try:
                        input_ids = self.tokenizer(batch, return_tensors='pt',padding= True).to(device)
                        outputs = self.model(**input_ids).end_scores
                        outputs_l.append(outputs)
                    except:
                        print("run one by one")
                        single_outputs_l = []
                        for single_batch in batch:
                            input_ids = self.tokenizer(single_batch, return_tensors='pt',padding= True).to(device)
                            outputs = self.model(**input_ids).end_scores
                            single_outputs_l.append(outputs)
                        outputs_l.extend(single_outputs_l)        
                return torch.cat(outputs_l,dim = 0).view(-1),outputs_l
            
            def _reward_run_debug(self, q_and_p_s, ans_s, device):
                outputs_l = []
                batch_size = self.batch_size
                for i in range(0,len(q_and_p_s),batch_size):
                    batch_inputs = q_and_p_s[i: i +batch_size]
                    batch_outputs = ans_s[i: i +batch_size]
                    batch = [self.template.format(model_input = batch_inputs[index], model_output = batch_outputs[index]) for index in range(len(batch_inputs))]
                    try:
                        input_ids = self.tokenizer(batch, return_tensors='pt',padding= True).to(device)
                        outputs = self.model(**input_ids).logits[:,-1,0].unsqueeze(-1)
                        outputs_l.append(outputs)
                    except:
                        print("run one by one")
                        single_outputs_l = []
                        for single_batch in batch:
                            input_ids = self.tokenizer(single_batch, return_tensors='pt',padding= True).to(device)
                            outputs = self.model(**input_ids).logits[:,0,0].unsqueeze(-1)
                            single_outputs_l.append(outputs)
                        outputs_l.extend(single_outputs_l)        
                return torch.cat(outputs_l,dim = 0).view(-1),outputs_l
            
            def reward_run(self, q_and_p_s, ans_s, device, mode):
                # "q_and_p_s are 'harmful input + prompt' question + prompt, ans_s are cost_lm's response"
                if self.config.debug:
                    scores, _ = self._reward_run_debug(q_and_p_s, ans_s, device)
                else:
                    scores, _ = self._reward_run(q_and_p_s, ans_s, device)

                return scores
            
        reward_model = RewardModel(config.reward_lm)
        reward_model.eval()
        reward_model.requires_grad_(False)
        reward_model_device = torch.cuda.device_count() - 1
        reward_model = reward_model.to(reward_model_device)
        @torch.no_grad()
        def get_reward(q_and_p_s,ans_s,mode = "train"):
            # "q_and_p_s are 'harmful input + prompt' question + prompt, ans_s are cost_lm's response"

            scores = reward_model.reward_run(q_and_p_s,ans_s,device = reward_model_device, mode = mode)
            return scores



    else:
        get_reward = True

    return get_reward
        
def create_targetlm(config):

    if os.environ.get("RANK", "0") == "0":

        class Target_Model(nn.Module): 
            def __init__(
                self, 
                config,
            ): 
                super().__init__()
                model_name = config.model_name
                self.template = config.template
                self.template = self.template.format(system = config.system_message, input = "{input}", prompt = "{prompt}")
                self.batch_size = config.batch_size
                kwargs = check_torch_dtype(config)

                model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                tokenizer.padding_side = "left"
                if not tokenizer.pad_token:
                    tokenizer.pad_token = tokenizer.eos_token
                self.model = model
                self.tokenizer= tokenizer
                self.gen_kwargs = {"pad_token_id":self.tokenizer.pad_token_id, "eos_token_id":self.tokenizer.eos_token_id, "bos_token_id":self.tokenizer.bos_token_id}
                
            def create_gen_config(self,gen_config):
                self.gen_config = GenerationConfig(**gen_config, **self.gen_kwargs)

            # q_s questions, p_s prompts
            def _targetlm_run(self, q_s, p_s, device):
                outputs_l = []
                batch_size = self.batch_size
                for i in range(0,len(q_s),batch_size):    
                    batch_inputs = q_s[i: i +batch_size]
                    batch_outputs = p_s[i: i +batch_size]
                    batch = [self.template.format(input = batch_inputs[index], prompt = batch_outputs[index]) for index in range(len(batch_inputs))]
                    try:
                        input_ids = self.tokenizer(batch, return_tensors='pt',padding= True).to(device)
                        output = self.model.generate(**input_ids,generation_config = self.gen_config)
                        output = output[:,input_ids["input_ids"].shape[-1]:]
                        output_text = self.tokenizer.batch_decode(output,skip_special_tokens= True)   
                        outputs_l.extend(output_text)
                    except:
                        print("run one by one")
                        single_outputs_l = []
                        for single_batch in batch:
                            input_ids = self.tokenizer(single_batch, return_tensors='pt',padding= True).to(device)
                            output = self.model.generate(**input_ids,generation_config = self.gen_config)
                            output = output[:,input_ids["input_ids"].shape[-1]:]
                            output_text = self.tokenizer.batch_decode(output,skip_special_tokens= True)   
                            single_outputs_l.extend(output_text)
                        outputs_l.extend(single_outputs_l)        
                return outputs_l
            
            def targetlm_run(self, q_s, p_s, device, mode):
                generations = self._targetlm_run(q_s, p_s, device)

                return generations
            
        target_model = Target_Model(config.target_lm)
        target_model.eval()
        target_model.requires_grad_(False)
        target_model_device = torch.cuda.device_count() - 2
        target_model = target_model.to(target_model_device)
        @torch.no_grad()
        def get_target_lm_generation(q_s,p_s,handler,mode = "train"):
            # q_s : questions  p_s:prompts
            config.target_lm.generation_configs.num_return_sequences = handler.need_N_responses()
            target_model.create_gen_config(config.target_lm.generation_configs)
            assert len(q_s) == len(p_s)
            generation = target_model.targetlm_run(q_s,p_s,device = target_model_device, mode = mode)
            return generation
        
    else:
        get_target_lm_generation = True

    return get_target_lm_generation




def create_reflm(config,mlp_state_dict):

    if os.environ.get("RANK", "0") == "0":
        class ref_lm_class(nn.Module):
            def __init__(self,config,mlp_state_dict):
                super().__init__()
                _ref_model,_ = make_lm_adaptor_model(config,mlp_state_dict)
                ref_model = make_input_conditioned_prompt_model(_ref_model, config)
                ref_model.eval()
                ref_model.requires_grad_(False)
                self.device = f"cuda:{torch.cuda.device_count()-2}"
                ref_model = ref_model.to(self.device)
                self.ref_model = ref_model

            @torch.no_grad()
            def teacher_forcing(self,source_texts,sample_ids):
                return self.ref_model.teacher_forcing(source_texts,sample_ids)
            
        ref_instance = ref_lm_class(config,mlp_state_dict)
        ref_lm_device = ref_instance.device

    else:
        ref_instance = True
        ref_lm_device = -1

    return ref_instance,ref_lm_device


class Handler:
    def __init__(self,config):

        self.num_samples = config.num_samples
        self.num_bootstraps = config.num_bootstraps
        self.compute_zscore = config.compute_zscore
        self.reward_shaping = config.reward_shaping
        self.reward_shaping_old_min = config.reward_shaping_old_min
        self.reward_shaping_old_max = config.reward_shaping_old_max
        self.reward_shaping_new_min = config.reward_shaping_new_min
        self.reward_shaping_new_max = config.reward_shaping_new_max
    
    @staticmethod
    def align_q_with_p(q_s,p_s):
        num_repeats = len(p_s)/len(q_s)
        return list(itertools.chain(*[[q for _ in range(num_repeats)]
                                      for q in q_s]))
    
    # p_s would be several times of p_s
    # before target_lm ,step 1
    # # # # # # # # # # # # # # # # # # # # # # # # 

    @staticmethod
    def _convert_tokens_to_string(tokenizer,p_tokens):
        # p_tokens prompt tokens
        return [tokenizer.convert_tokens_to_string(s)
                for s in p_tokens]
    
    def need_N_responses(self):
        return self.num_samples * self.num_bootstraps
    

    # use this method to get p+q 's answers/responses
    # before target_lm ,step 2
    # # # # # # # # # # # # # # # # # # # # # # # # 

    @staticmethod
    def align_q_and_p_with_a(q_s,p_s,a_s):
        # suppose # q_s is N, then a_s is N * need_N_response
        # q_s is [q_0,q_0,q_1,q_1]  p_s = [p_0,p_1,p_2,p_3](because of num_repeats),  need_N_response = 2, so a_s are [q_0_p_0_a_0,q_0_p_0_a_1, 
        #                                                                                                               q_0_p_1_a_0,q_0_p_1_a_1, 
                                                                                                                #       q_1_p_2_a_0,q_1_p_2_a_1, 
                                                                                                                #       q_1_p_3_a_0,q_0_p_3_a_1 ]
        assert len(q_s) == len(p_s)
        num_repeats = int(len(a_s)/len(q_s))
        q_s = list(itertools.chain(*[[q for _ in range(num_repeats)]
                                      for q in q_s]))
        p_s = list(itertools.chain(*[[q for _ in range(num_repeats)]
                                      for q in p_s]))
        return q_s,p_s,a_s
    
    # use target model
    # after hypos = self.generator.sample_generate(prompt, src, N,    in rl-prompt
    # # # # # # # # # # # # # # # # # # # # # # # # 
    def process_rewards(self,rewards,q_s,repeat_times,mode = "train"):
        # number of rewards should be same with a_s above
        # 注意这里最终返回的是不考虑num_sample 和 num_bootstrap, bootstrap是为了减少variance（可能也不需要bootstrap， bootstrap =1, num_samples也是为了减少variance，因为最后是用的他们的mean作为rewards）
        
        tmp_rewards = []
        input_rewards = defaultdict(list)
        interval = self.need_N_responses()
        for index in range(0,len(rewards),interval):
            _bootstapped_rewards = self._boostrap_max_rewards_k_times(rewards[index: index + interval], self.num_bootstraps)
            assert len(_bootstapped_rewards) == self.num_bootstraps
            tmp_rewards.append(torch.Tensor(_bootstapped_rewards).float().mean())
            tmp_input = q_s[int(index/interval)]
            # assert len(tmp_input) == 1
            input_rewards[tmp_input] += _bootstapped_rewards

        rewards_tensor = torch.stack(tmp_rewards)
        # if # promtps = # source ,then tmp_rewards‘s value = mean(boostrapped_rewards), then rewards must be 0...
        if mode == "train" and self.compute_zscore and repeat_times != 1:
            # each key is the input, we have # of prompts for each input times num_bootstrapping. 
            # according to the paper, it should have 4 for each, but it would be 8 considering the bootstrapping..
            rewards_tensor = self._compute_reward_zscores(rewards_tensor, 
                                                          q_s, 
                                                          input_rewards)
            
            if self.reward_shaping:
                assert self.reward_shaping == self.compute_zscore, "if no compute z-score, then the old range is not from 0 to 1 anymore"
                rewards_tensor = self.get_reward_shaping_func(rewards_tensor)

        return rewards_tensor



        # if we dont use z-score, probably we need to use mean instead of += 
        # for i,q in enumerate(q_s):
        #     input_rewards[q] += bootstapped_rewards[i]

        
    
    @staticmethod
    def _compute_reward_zscores(
        rewards_tensor: torch.Tensor,
        input_texts,
        input_rewards,
        eps: float = 1e-4
    ):
        input_reward_means = {k: np.mean(v) for k, v in input_rewards.items()}
        input_reward_stds = {k: np.std(v) for k, v in input_rewards.items()}
        idx_means = torch.tensor([input_reward_means[s] for s in input_texts])
        idx_stds = torch.tensor([input_reward_stds[s] for s in input_texts])
        # print(idx_means)
        # print(idx_stds)
        return (rewards_tensor - idx_means.float()) / (idx_stds.float() + eps)

    


    @staticmethod
    def _boostrap_max_rewards_k_times(
        rewards,
        k
    ):
        # Segment list rewards into k equal sub-lists
        rewards = rewards.tolist()
        l = len(rewards)
        assert l % k == 0, f'l={l}, k={k}'
        segmented_rewards = [rewards[i*l//k:(i+1)*l//k] for i in range(k)]  # [k, l/k]
        # We use different rewards for each bootstrap for now

        # For each sub-list, take the max as the sub-reward
        values, indices = (torch.tensor(segmented_rewards).float().max(axis = 1))
        # Take numbers from the original list to avoid numerical issues
        bootstrap_max_rewards = [segmented_rewards[i][index]
                                 for i, index in enumerate(indices)]

        return bootstrap_max_rewards
    
    def get_reward_shaping_func(
        self,
        reward
        ):
        old_min = self.reward_shaping_old_min
        old_max = self.reward_shaping_old_max
        new_min = self.reward_shaping_new_min
        new_max = self.reward_shaping_new_max
        
        percentile = (reward - old_min) / (old_max - old_min)
        return percentile * (new_max - new_min) + new_min

        



def run_train_sql_on(batch,prompt_model,prompt_model_tokenizer,accelerator,repeat_texts,ref_lm_instance,target_lm_fn,reward_lm_fn,handler,train_config,data_config,s_p_t_file):
    
    outputs = prompt_model.generate(batch[data_config["keys"][data_config.source_text_pos]], do_sample = True)
    # outputs :sample_tokens, sample_logits, sample_ids, sample_length
    sample_logits,sample_ids,sample_length = outputs['sample_logits'],outputs['sample_ids'],outputs['sample_lengths']
    # 文字不能gather
    # gathered_sample_tokens = accelerator.gather(sample_tokens)
    gathered_sample_ids = accelerator.gather(sample_ids)
    source_texts_tokens = prompt_model_tokenizer(batch[data_config["keys"][data_config.source_text_pos]],padding = True, truncation = True,return_tensors = "pt")["input_ids"].to(accelerator.process_index)
    padded_source_texts_tokens = accelerator.pad_across_processes(source_texts_tokens,dim = 1, pad_index = prompt_model_tokenizer.eos_token_id, pad_first= False)
    gathered_source_texts_tokens = accelerator.gather(padded_source_texts_tokens)
    device = sample_ids.device
    dim = sample_logits.shape[-1]
    if accelerator.is_main_process:
        # 为什么需要gather并且只在第一个process上处理，反证法，如果想每个process都直接处理，那就需要每个process都有一个ref——model，那vram就占比比较高了
        source_texts = prompt_model_tokenizer.batch_decode(gathered_source_texts_tokens,skip_special_tokens = True)
        outputs_ = ref_lm_instance.teacher_forcing(source_texts = source_texts,sample_ids = gathered_sample_ids)

        # sample_tokens 就是sample_ids
        output_ids = gathered_sample_ids.contiguous()
                                                        
        _output_tokens = prompt_model_tokenizer.batch_decode(output_ids,skip_special_tokens = True)
        _source_texts_repeated = repeat_texts(source_texts,int(len(_output_tokens)/len(source_texts)))
        # output_ids are on the first process
        target_lm_generations = target_lm_fn(_source_texts_repeated,_output_tokens,handler)
        s_p_t_file.write_all(pair_src_p_target(_source_texts_repeated,_output_tokens,target_lm_generations))
        source_texts_repeated,output_tokens,target_lm_generations = handler.align_q_and_p_with_a(_source_texts_repeated,_output_tokens,target_lm_generations)
        
        # whether need to use source + prompt / or only source is enough when using the cost model
        _reward_scores = reward_lm_fn(source_texts_repeated,target_lm_generations)
        _reward_scores = _reward_scores * train_config.reward_multiply
        rewards_all = handler.process_rewards(_reward_scores,_source_texts_repeated,int(len(_output_tokens)/len(source_texts)))
        rewards_all = [
            torch.tensor(score, dtype=torch.bfloat16, device=device).view(
                -1,
            )
            for score in rewards_all
        ]
        rewards_all = pad_sequence(rewards_all, batch_first=True, padding_value=-np.inf)
        rewards_all = list(rewards_all.reshape(accelerator.num_processes,-1).unbind())

        ref_logits_all = [
            torch.tensor(sample_logits, dtype=torch.bfloat16, device=device)
            for sample_logits in outputs_["sample_logits"]
        ]
        ref_logits_all = pad_sequence(ref_logits_all, batch_first=True, padding_value=-np.inf)
        ref_logits_all = list(ref_logits_all.chunk(accelerator.num_processes))
    else:
        rewards_all = None
        ref_logits_all = None
        
    # 当为bf 16的时候要考虑是否需要在初始化的考虑bf16,且不同的元素计算也要考虑bf16???
    # 比如reward_model要不要.half()
    if torch.distributed.is_initialized():
        dist.barrier()
    if torch.distributed.is_initialized():
        ref_logits = torch.empty(sample_logits.shape,device = device).to(dtype=torch.bfloat16)
        torch.distributed.scatter(ref_logits, ref_logits_all,src = 0)
    else:
        ref_logits = ref_logits_all[0].clone()

    if torch.distributed.is_initialized():
        dist.barrier()

    if torch.distributed.is_initialized():
        rewards = torch.empty(sample_ids.shape[0],device = device).to(dtype=torch.bfloat16)
        torch.distributed.scatter(rewards, rewards_all, src = 0)
    else:
        rewards = rewards_all[0].clone()

    if torch.distributed.is_initialized():
        dist.barrier()

    if accelerator.is_main_process:
        assert all(rewards == rewards_all[0])

    sql_loss, sql_loss_log = sql_loss_with_sparse_rewards(
        implementation=train_config.sql_loss_impl,
        logits=sample_logits,
        logits_=ref_logits,
        actions=sample_ids,
        sampled_actions=None,
        rewards=rewards,
        sequence_length=sample_length)
    return sql_loss, rewards
    




def run_train_sql_off(batch,prompt_model,prompt_model_tokenizer,accelerator,repeat_texts,ref_lm_instance,target_lm_fn,reward_lm_fn,handler,train_config,data_config,s_p_t_file):
    
    off_tokens = batch[data_config["keys"][data_config.id_tokens_pos]]
    # off_ids = prompt_model_tokenizer(off_tokens,return_tensors = "np",add_special_tokens = False).input_ids
    # off_ids = np.stack(off_ids)
    # off_ids = torch.tensor(off_ids.astype(np.int64)).to(f"cuda:{accelerator.process_index}",dtype=torch.int64)

    if not train_config.debug_off:
        off_ids = prompt_model_tokenizer(off_tokens,return_tensors = "pt",add_special_tokens = False).input_ids.to(f"cuda:{accelerator.process_index}")

    if train_config.debug_off:
        off_ids = prompt_model_tokenizer(off_tokens,return_tensors = "pt",add_special_tokens = False, padding = 'max_length', max_length = prompt_model.prompt_length, truncation = True).input_ids.to(f"cuda:{accelerator.process_index}")



    # TODO: off_ids repeat since in the teacher_forcing, they would repeat!! but this raise a concern about they would learn to know each input would generate the same prompt?? then it has no point to repeat it any more? set rep_train = 1??
    # off_ids = torch.tensor(np.repeat(off_ids,prompt_model.source_train_reps,axis=0)).to(f"cuda:{accelerator.process_index}",dtype=torch.int64)

    assert off_ids.shape[1] == prompt_model.prompt_length
    outputs = prompt_model.teacher_forcing(source_texts = batch[data_config["keys"][data_config.source_text_pos]],
                                           sample_ids = off_ids)
    
    sample_logits,sample_ids = outputs['sample_logits'],outputs['sample_ids']
    gathered_sample_ids = accelerator.gather(sample_ids)
    source_texts_tokens = prompt_model_tokenizer(batch[data_config["keys"][data_config.source_text_pos]],padding = True, truncation = True,return_tensors = "pt")["input_ids"].to(accelerator.process_index)
    padded_source_texts_tokens = accelerator.pad_across_processes(source_texts_tokens,dim = 1, pad_index = prompt_model_tokenizer.eos_token_id, pad_first= False)
    gathered_source_texts_tokens = accelerator.gather(padded_source_texts_tokens)
    device = sample_ids.device
    dim = sample_logits.shape[-1]
    if accelerator.is_main_process:
        # 为什么需要gather并且只在第一个process上处理，反证法，如果想每个process都直接处理，那就需要每个process都有一个ref——model，那vram就占比比较高了
        source_texts = prompt_model_tokenizer.batch_decode(gathered_source_texts_tokens,skip_special_tokens = True)
        outputs_ = ref_lm_instance.teacher_forcing(source_texts = source_texts,sample_ids = gathered_sample_ids)

        # sample_tokens 就是sample_ids
        output_ids = gathered_sample_ids.contiguous()
                                                        
        _output_tokens = prompt_model_tokenizer.batch_decode(output_ids,skip_special_tokens = True)
        _source_texts_repeated = repeat_texts(source_texts,int(len(_output_tokens)/len(source_texts)))
        # output_ids are on the first process
        target_lm_generations = target_lm_fn(_source_texts_repeated,_output_tokens,handler)
        s_p_t_file.write_all(pair_src_p_target(_source_texts_repeated,_output_tokens,target_lm_generations))




        source_texts_repeated,output_tokens,target_lm_generations = handler.align_q_and_p_with_a(_source_texts_repeated,_output_tokens,target_lm_generations)
        
        # whether need to use source + prompt / or only source is enough when using the cost model
        _reward_scores = reward_lm_fn(source_texts_repeated,target_lm_generations)

        _reward_scores = _reward_scores * train_config.reward_multiply
        rewards_all = handler.process_rewards(_reward_scores,_source_texts_repeated,int(len(_output_tokens)/len(source_texts)))
        rewards_all = [
            torch.tensor(score, dtype=torch.bfloat16, device=device).view(
                -1,
            )
            for score in rewards_all
        ]
        rewards_all = pad_sequence(rewards_all, batch_first=True, padding_value=-np.inf)
        rewards_all = list(rewards_all.reshape(accelerator.num_processes,-1).unbind())

        ref_logits_all = [
            torch.tensor(sample_logits, dtype=torch.bfloat16, device=device)
            for sample_logits in outputs_["sample_logits"]
        ]
        ref_logits_all = pad_sequence(ref_logits_all, batch_first=True, padding_value=-np.inf)
        ref_logits_all = list(ref_logits_all.chunk(accelerator.num_processes))
    else:
        rewards_all = None
        ref_logits_all = None
    
    # 当为bf 16的时候要考虑是否需要在初始化的考虑bf16,且不同的元素计算也要考虑bf16???
    # 比如reward_model要不要.half()
    if torch.distributed.is_initialized():
        dist.barrier()
    if torch.distributed.is_initialized():
        ref_logits = torch.empty(sample_logits.shape,device = device).to(dtype=torch.bfloat16)
        torch.distributed.scatter(ref_logits, ref_logits_all,src = 0)
    else:
        ref_logits = ref_logits_all[0].clone()

    if torch.distributed.is_initialized():
        dist.barrier()

    if torch.distributed.is_initialized():
        rewards = torch.empty(sample_ids.shape[0],device = device).to(dtype=torch.bfloat16)
        torch.distributed.scatter(rewards, rewards_all, src = 0)
    else:
        rewards = rewards_all[0].clone()

    if torch.distributed.is_initialized():
        dist.barrier()

    if accelerator.is_main_process:
        assert all(rewards == rewards_all[0])

    sample_length = torch.full((sample_logits.shape[0],), prompt_model.prompt_length).to(f"cuda:{accelerator.process_index}")

    if train_config.only_compute_for_gt0_when_off:
        gt0_index = torch.where(rewards > 0)
        if gt0_index[0].shape[0] == 0:

            return torch.tensor(0).to(f"cuda:{accelerator.process_index}",dtype=torch.bfloat16),torch.tensor([0]).to(f"cuda:{accelerator.process_index}",dtype=torch.bfloat16)

        sample_logits = sample_logits[gt0_index]
        ref_logits = ref_logits[gt0_index]
        sample_ids = sample_ids[gt0_index]
        rewards = rewards[gt0_index]
        sample_length = sample_length[gt0_index]


    sql_loss, sql_loss_log = sql_loss_with_sparse_rewards(
        implementation=train_config.sql_loss_impl,
        logits=sample_logits,
        logits_=ref_logits,
        actions=sample_ids,
        sampled_actions=None,
        rewards=rewards,
        sequence_length=sample_length)
    return sql_loss, rewards.mean()
    
