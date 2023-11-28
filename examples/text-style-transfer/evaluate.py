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
        print(config.base_lm.model_name)
        policy_model,prompt_model_tokenizer = make_lm_adaptor_model(config,mode = "infer")
        prompt_model = make_input_conditioned_prompt_model(policy_model, config)

    reward_lm_fn = create_reward(config)
    target_lm_fn = create_targetlm(config)
    train_dataset,test_dataset = make_attack_datasets(config.data)
    eval_dataloader = DataLoader(test_dataset, batch_size= config.eval_config.batch_size,collate_fn=attack_collate_fn)
    rewards,q_ss,p_ss,target_lm_generation_ss,file_ss,label_ss,rep = evaluate_fn(prompt_model,prompt_model_tokenizer,reward_lm_fn,target_lm_fn,eval_dataloader,config)
    save_d_l = []
    for i in range(len(rewards)):
        d = dict(q = q_ss[i],p = p_ss[i],target_lm_generation = target_lm_generation_ss[i],reward = rewards[i],harm = 1 if rewards[i] > 0 else 0,target = label_ss[i],file = file_ss[i])
        save_d_l.append(d)

    if config.eval_config.prompt_own:
        model_type="prompt_own"
    else:
        model_type = config.base_lm.model_name.split("/")[-1].replace("trainer.","")

    Path(config.eval_config.s_p_t_dir).mkdir(exist_ok= True, parents= True)
    with jsonlines.open(os.path.join(config.eval_config.s_p_t_dir,f"rep_{config.eval_config.prompt_lm.source_infer_reps}|{model_type}.jsonl"),"w") as f:
        f.write_all(save_d_l)


@torch.no_grad()
def evaluate_fn(prompt_model,prompt_model_tokenizer,reward_lm_fn,target_lm_fn,eval_dataloader,config):
    reward_all = []
    q_ss = []
    p_ss = []
    target_lm_generation_ss = []
    label_ss = []
    file_ss = []
    
    used_instances = set()
    rep = 1

    for batch in tqdm(eval_dataloader):
        label_s = batch["target"]
        file_s = batch["file"]
        q_s = batch["q"]
        if config.eval_config.prompt_own:

            p_s = batch["p"]
        else:
            q_s = list(set(q_s))
            q_s = list(set(q_s) - set(q_s).intersection(used_instances))
            for _ in q_s:
                used_instances.add(_)
            outputs = prompt_model.generate(q_s, infer = True, do_sample = config.eval_config.prompt_lm.do_sample, source_infer_reps = config.eval_config.prompt_lm.source_infer_reps)
            
            p_s = prompt_model_tokenizer.batch_decode(outputs['sample_ids'],skip_special_tokens = True)
            if len(p_s) != len(q_s) and len(p_s) > len(q_s):
                rep = int(len(p_s)/len(q_s))
                tmp = []
                for i in range(len(q_s)):
                    for j in range(int(len(p_s)/len(q_s))):
                        tmp.append(q_s[i])
                q_s = tmp

            
        target_lm_generations = target_lm_fn(q_s,p_s,mode = "infer")
        _reward_scores = reward_lm_fn(q_s,target_lm_generations)
        reward_scores = _reward_scores * config.trainer.reward_multiply
        reward_all.extend(reward_scores.cpu().tolist())
        q_ss.extend(q_s)
        p_ss.extend(p_s)
        target_lm_generation_ss.extend(target_lm_generations)
        file_ss.extend(file_s)
        label_ss.extend(label_s)
    return reward_all,q_ss,p_ss,target_lm_generation_ss,file_ss,label_ss,rep

        


if __name__ == "__main__":
    main()
