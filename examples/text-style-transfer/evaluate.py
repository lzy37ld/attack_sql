import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tst_modules import create_reward,create_targetlm,Handler
from tst_helpers import make_attack_datasets,attack_collate_fn
from rlprompt.models import (make_lm_adaptor_model,make_input_conditioned_prompt_model)
from torch.utils.data import DataLoader
import jsonlines
import os
from rlprompt.utils.utils import colorful_print
from tqdm import tqdm
from pathlib import Path


@hydra.main(config_path="./myconfig", config_name="config_evaluate")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')

    if config.eval_config.prompt_own:
        prompt_model,prompt_model_tokenizer = None,None
    else:
        policy_model,prompt_model_tokenizer = make_lm_adaptor_model(config,mode = "infer")
        prompt_model = make_input_conditioned_prompt_model(policy_model, config)

    reward_lm_fn = create_reward(config)
    target_lm_fn = create_targetlm(config)
    train_dataset,test_dataset = make_attack_datasets(config.data)
    eval_dataloader = DataLoader(test_dataset, batch_size= config.eval_config.batch_size,collate_fn=attack_collate_fn)
    rewards,q_ss,p_ss,t_ss = evaluate_fn(prompt_model,prompt_model_tokenizer,reward_lm_fn,target_lm_fn,eval_dataloader,config)
    save_d_l = []
    for i in range(len(rewards)):
        d = dict(q = q_ss[i],p = p_ss[i],t = t_ss[i],reward = rewards[i],harm = 1 if rewards[i] > 0 else 0)
        save_d_l.append(d)

    if config.eval_config.prompt_own:
        model_type="prompt_own"
    else:
        model_type = config.base_lm.model_name.split("/")[-1].replace("trainer.","")

    Path(config.eval_config.s_p_t_dir).mkdir(exist_ok= True, parents= True)
    with jsonlines.open(os.path.join(config.eval_config.s_p_t_dir,f"{model_type}.jsonl"),"w") as f:
        f.write_all(save_d_l)


@torch.no_grad()
def evaluate_fn(prompt_model,prompt_model_tokenizer,reward_lm_fn,target_lm_fn,eval_dataloader,config):
    data_config = config.data
    reward_all = []
    q_ss = []
    p_ss = []
    t_ss = []
    for batch in tqdm(eval_dataloader):
        q_s = batch[data_config["keys"][data_config.source_text_pos]]
        if config.eval_config.prompt_own:
            p_s = batch[data_config["keys"][data_config.id_tokens_pos]]
        else:
            outputs = prompt_model.generate(q_s,do_sample = False, infer = True)
            p_s = prompt_model_tokenizer.batch_decode(outputs['sample_ids'],skip_special_tokens = True)
            
        target_lm_generations = target_lm_fn(q_s,p_s,mode = "infer")
        _reward_scores = reward_lm_fn(q_s,target_lm_generations)
        reward_scores = _reward_scores * config.trainer.reward_multiply
        reward_all.extend(reward_scores.cpu().tolist())
        q_ss.extend(q_s)
        p_ss.extend(p_s)
        t_ss.extend(target_lm_generations)
    return reward_all,q_ss,p_ss,t_ss

        


if __name__ == "__main__":
    main()
