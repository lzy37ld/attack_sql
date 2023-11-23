import os
import dataclasses
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from rlprompt.trainers import TrainerConfig, make_trainer
from rlprompt.modules import SQLModuleConfig, make_sql_module
from rlprompt.models import (make_lm_adaptor_model,make_input_conditioned_prompt_model)
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)
from tst_helpers import (make_prompted_text_style_transfer_reward,make_attack_datasets,attack_collate_fn,
                         make_text_style_transfer_datasets,
                         get_style_classifier)
from tst_modules import create_reward,create_targetlm, create_reflm,Handler,run_train_sql_on,run_train_sql_off

from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import itertools
from accelerate import Accelerator
from torch.utils.data import DataLoader
import copy
from utils import repeat_texts,is_main_process,is_dist
from accelerate.utils import set_seed
from enum import Enum
import pathlib
from tqdm.auto import tqdm
set_seed(42)


class ForwardMode(Enum):
    SQL_ON = "SQL_ON"
    SQL_OFF_GT = "SQL_OFF_GT"
    INFER = "INFER"

def get_forward_modes(
    training_mode,
    mix_strategy,
    step,
):
    # sql-mixed - alternate
    # sql-mixed - mix
    # sql-onpolicy
    # sql-offpolicy
    if training_mode == "sql-mixed":
        candidate_modes = [
            ForwardMode.SQL_OFF_GT,
            ForwardMode.SQL_ON]

        if mix_strategy == "alternate":
            modes = [candidate_modes[step % len(candidate_modes)]]
        elif mix_strategy == "mix":
            modes = candidate_modes
        else:
            raise NotImplementedError()

    else:
        training_mode_map = {"sql-onpolicy": ForwardMode.SQL_ON,
                            "sql-offpolicy": ForwardMode.SQL_OFF_GT}
        modes = [training_mode_map[training_mode]]

    return modes



def batch_samples(l,bs):
    for i in range(0,len(l),bs):
        yield l[i:i+bs]


@hydra.main(config_path="./myconfig", config_name="config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    import sys
    
    ckpt_name = "|".join(sys.argv[1:])
    if ckpt_name == "":
        ckpt_name = "base"
    print(ckpt_name)

    # train_dataset, val_dataset, test_dataset = \
    #     make_text_style_transfer_datasets(config.data)
    train_dataset,test_dataset = make_attack_datasets(config.data)
    print('Train Size:', len(train_dataset))
    print('Test Size:', len(test_dataset))
    print('Examples:', train_dataset[:5])

    policy_model,prompt_model_tokenizer = make_lm_adaptor_model(config)
    prompt_model = make_input_conditioned_prompt_model(policy_model, config)
    if is_dist():
        dist.barrier()
    reward_lm_fn = create_reward(config)

    target_lm_fn = create_targetlm(config)
    ref_instance,ref_lm_device = create_reflm(config,policy_model.model.mlp.state_dict())
    handler = Handler(config.handler)

    train_config = config.trainer
    data_config = config.data
    prompt_model_config = config.prompt_lm
    train_dataloader = DataLoader(train_dataset,shuffle= True,batch_size= train_config.batch_size,collate_fn=attack_collate_fn)
    optimizer = torch.optim.AdamW(prompt_model.parameters(), lr=train_config.learning_rate)

    accelerator = Accelerator(log_with=train_config.log_with,mixed_precision=train_config.mixed_precision)
    prompt_model, optimizer, train_dataloader = accelerator.prepare(
        prompt_model, optimizer, train_dataloader
    )
    if train_config.log_with is not None:
        accelerator.init_trackers(
        project_name="my_attack", 
        config=dict(config),
        init_kwargs={"wandb": {"entity": "lzy37ld","name":ckpt_name}}
        # 可以考虑要不要设置run_name
        )


    progress_bar = tqdm(range(train_config.num_epochs * len(train_dataloader)), disable=not accelerator.is_local_main_process)
    over_all_steps = 0
    for epoch in range(train_config.num_epochs):
        for step, batch in enumerate(train_dataloader):
            progress_bar.update(1)
            prompt_model.train()
            over_all_steps += 1
            # 请注意有duplicate现象，需要处理
            # 或者我直接batsize不设置太大，让后确保# data = num_process * bathsize.. (我没有gradient accumulation)
            

            # TODO:写一个mode循环，先off_policy再 on(warm up)，可以写不同的strategy
            # TODO:保存reward

            modes = get_forward_modes(training_mode= train_config.training_mode, mix_strategy= train_config.mix_strategy, step = step)
            loss_list = []
            reward_list = []
            for mode in modes:
                if mode == ForwardMode.SQL_OFF_GT:
                    _sql_loss,_rewards = run_train_sql_off(batch,prompt_model,prompt_model_tokenizer,accelerator,repeat_texts,ref_instance,target_lm_fn,reward_lm_fn,handler,train_config,data_config)
                elif mode == ForwardMode.SQL_ON:
                    _sql_loss,_rewards = run_train_sql_on(batch,prompt_model,prompt_model_tokenizer,accelerator,repeat_texts,ref_instance,target_lm_fn,reward_lm_fn,handler,train_config,data_config)
                else:
                    raise NotImplementedError()
                loss_list.append(_sql_loss)
                reward_list.append(_rewards)

            if len(loss_list) == 0:
                continue

            loss = torch.mean(torch.stack(loss_list)).requires_grad_(True)
            rewards = torch.mean(torch.stack(reward_list))

            if train_config.log_with is not None:
                accelerator.log({"train_sql_loss":loss,
                                "rewards_main_process":rewards},
                                step=over_all_steps)
                
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                if train_config.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(prompt_model.parameters(), train_config.max_grad_norm)
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()

            # update ref model
            if accelerator.is_main_process:
                prompt_model_for_update_ref = accelerator.unwrap_model(prompt_model)
                if train_config.ref_update_method == "polyak":

                    for (name_param_,param_), (name_param,param) in zip(ref_instance.ref_model.named_parameters(),
                                            prompt_model_for_update_ref.named_parameters()):

                        param_.data.copy_((1 - train_config.ref_learning_rate) * param_
                                        + train_config.ref_learning_rate * param.to(ref_lm_device))
                        if "mlp" not in name_param:
                            assert (param_.data == param.data.to(ref_lm_device)).all().item(), "only mlps are updating"



            # TODO:eval
            # prompt_model.eval()
        #             model.eval()
            # @torch.no_grad()
        # losses = []
        # for step, batch in enumerate(eval_dataloader):
        #     with torch.no_grad():
        #         outputs = model(**batch)

        #     loss = outputs.loss
        #     losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))


    save_dir = os.path.join(config.trainer.base_root,config.trainer.save_dir)
    save_dir = os.path.join(save_dir,ckpt_name)
    pathlib.Path(save_dir).mkdir(parents= True, exist_ok= True)

    prompt_model_for_save = accelerator.unwrap_model(prompt_model)
    prompt_model_for_save._model.model.save_pretrained(
    save_dir,
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
    state_dict=accelerator.get_state_dict(prompt_model._model.model),
)



# # for stage 3 pls refer to https://huggingface.co/docs/accelerate/usage_guides/deepspeed#saving-and-loading for saving



if __name__ == "__main__":
    main()
