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
from accelerate.utils import set_seed
set_seed(42)



def set_pad_token(t):
    if t.pad_token is None:
        t.pad_token = t.eos_token
    
    return t


def get_batch(l,bs):
    for i in range(0,len(l),bs):
        yield l[i: i+bs]

def do_reps(
    source_texts, 
    num_reps
):
    source_reps = []
    for text in source_texts: 
        for _ in range(num_reps): 
            source_reps.append(text)
    return source_reps


@hydra.main(config_path="./myconfig", config_name="config_evaluate")
def main(config: "DictConfig"):
    Path(config.eval_config.s_p_t_dir).mkdir(exist_ok= True, parents= True)
    if config.eval_config.prompt_own:
        model_type="prompt_own"
        config.eval_config.prompt_lm.source_infer_reps = 1
    else:
        model_type = config.base_lm.model_name.split("/")[-1].replace("trainer.","")

    path = os.path.join(config.eval_config.s_p_t_dir,f"rep_{config.eval_config.prompt_lm.source_infer_reps}|do_sample_{config.eval_config.target_lm.generation_configs.do_sample}|append_label_length_{config.eval_config.append_label_length}|{model_type}.jsonl")


    fp = jsonlines.open(path,"a")
    with open(path) as f:
        existed_lines = len(f.readlines())
    assert existed_lines == 0, "delete it"

    all_unique_qs_datas = []
    all_qs = []
    with jsonlines.open(config.data.path) as reader:
        for line in reader:
            if config.eval_config.prompt_own:
                all_unique_qs_datas.append(line)
            else:
                if line["q"] not in all_qs:
                    all_unique_qs_datas.append(line)
                    all_qs.append(line["q"])
    

    colorful_print(OmegaConf.to_yaml(config), fg='red')

    if config.eval_config.prompt_own:
        prompt_model,prompt_model_tokenizer = None,None
        if config.eval_config.append_label_length != -1:
            # only for selecting tokens at the front
            prompt_model_tokenizer = set_pad_token(AutoTokenizer.from_pretrained(config.base_lm.model_name,padding_side = "right"))
    else:
        print(config.base_lm.model_name)
        policy_model,prompt_model_tokenizer = make_lm_adaptor_model(config,mode = "infer")
        prompt_model = make_input_conditioned_prompt_model(policy_model, config)

    reward_lm_fn = create_reward(config)
    target_lm_fn = create_targetlm(config)

    evaluate_fn(prompt_model,prompt_model_tokenizer,reward_lm_fn,target_lm_fn,all_unique_qs_datas,config,fp)
    # save_d_l = []
    # for i in range(len(rewards)):
    #     d = dict(q = q_ss[i],p = p_ss[i],target_lm_generation = target_lm_generation_ss[i],reward = rewards[i],harm = 1 if rewards[i] > 0 else 0,target = label_ss[i],file = file_ss[i])
    #     save_d_l.append(d)



@torch.no_grad()
def evaluate_fn(prompt_model,prompt_model_tokenizer,reward_lm_fn,target_lm_fn,all_unique_qs_datas,config,fp):
    reward_all = []
    q_ss = []
    p_ss = []
    target_lm_generation_ss = []
    label_ss = []
    file_ss = []
    
    used_instances = set()
    rep = 1

    with tqdm(total=len(all_unique_qs_datas)) as progress:
            
        for batch in get_batch(all_unique_qs_datas,config.eval_config.batch_size):
            batch = attack_collate_fn(batch)
            label_s = batch["target"]
            file_s = batch["file"]
            q_s = batch["q"]
            if config.eval_config.prompt_own:
                
                p_s = batch["p"]

            else:

                outputs = prompt_model.generate(q_s, infer = True, do_sample = config.eval_config.prompt_lm.do_sample, source_infer_reps = config.eval_config.prompt_lm.source_infer_reps)
                
                p_s = prompt_model_tokenizer.batch_decode(outputs['sample_ids'],skip_special_tokens = True)

                q_s = do_reps(q_s,config.eval_config.prompt_lm.source_infer_reps)
                label_s = do_reps(label_s,config.eval_config.prompt_lm.source_infer_reps)
                file_s = do_reps(file_s,config.eval_config.prompt_lm.source_infer_reps)

                # if len(p_s) != len(q_s) and len(p_s) > len(q_s):
                #     rep = int(len(p_s)/len(q_s))
                #     assert rep == int(config.eval_config.prompt_lm.source_infer_reps)
                #     tmp = []
                #     for i in range(len(q_s)):
                #         for j in range(int(len(p_s)/len(q_s))):
                #             tmp.append(q_s[i])
                #     q_s = tmp

            assert len(q_s) == len(p_s)

            if config.eval_config.append_label_length != -1 and config.eval_config.prompt_own:
                label_s_tokens = prompt_model_tokenizer(label_s,padding = True,return_tensors = "pt",add_special_tokens = False).input_ids[:,:config.eval_config.append_label_length]
                label_s_tokens_decode = prompt_model_tokenizer.batch_decode(label_s_tokens,skip_special_tokens = True)

            target_lm_generations = target_lm_fn(q_s,p_s,mode = "infer",after_sys_tokens = label_s_tokens_decode)
            _reward_scores = reward_lm_fn(q_s,target_lm_generations)
            reward_scores = _reward_scores * config.trainer.reward_multiply
            reward_scores = reward_scores.cpu().tolist()

            for i in range(len(reward_scores)):
                fp.write(dict(q = q_s[i],p = p_s[i],target_lm_generation = target_lm_generations[i],reward = reward_scores[i],harm = 1 if reward_scores[i] > 0 else 0,target = label_s[i],file = file_s[i]))
            
            progress.update(config.eval_config.batch_size)


    #     q_ss.extend(q_s)
    #     p_ss.extend(p_s)
    #     target_lm_generation_ss.extend(target_lm_generations)
    #     file_ss.extend(file_s)
    #     label_ss.extend(label_s)
    # return reward_all,q_ss,p_ss,target_lm_generation_ss,file_ss,label_ss,rep

        


if __name__ == "__main__":
    main()
