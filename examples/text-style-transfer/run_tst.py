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
from tst_helpers import (make_prompted_text_style_transfer_reward,make_attack_datasets,
                         make_text_style_transfer_datasets,
                         get_style_classifier)
from tst_modules import create_reward,create_targetlm, create_reflm,Handler,run_train

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

set_seed(42)




def batch_samples(l,bs):
    for i in range(0,len(l),bs):
        yield l[i:i+bs]


@hydra.main(config_path="./myconfig", config_name="config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')


    # train_dataset, val_dataset, test_dataset = \
    #     make_text_style_transfer_datasets(config.data)
    train_dataset = make_attack_datasets(config.data)
    print('Train Size:', len(train_dataset))
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
    train_dataloader = DataLoader(train_dataset,shuffle= True,batch_size= train_config.batch_size)
    optimizer = torch.optim.AdamW(prompt_model.parameters(), lr=train_config.learning_rate)

    accelerator = Accelerator(log_with=train_config.log_with)
    prompt_model, optimizer, train_dataloader = accelerator.prepare(
        prompt_model, optimizer, train_dataloader
    )
    if train_config.log_with is not None:
        accelerator.init_trackers(
        project_name="my_attack", 
        config=dict(config),
        init_kwargs={"wandb": {"entity": "lzy37ld"}}
        # 可以考虑要不要设置run_name
        )

    for epoch in range(train_config.num_epochs):
        for step, batch in enumerate(train_dataloader):
            # 请注意有duplicate现象，需要处理
            # 或者我直接batsize不设置太大，让后确保# data = num_process * bathsize.. (我没有gradient accumulation)
            
            sql_loss,rewards = run_train(batch,prompt_model,prompt_model_tokenizer,accelerator,repeat_texts,ref_instance,target_lm_fn,reward_lm_fn,handler,train_config)
            if train_config.log_with is not None:
                accelerator.log({"train_sql_loss":sql_loss,
                                "rewards_main_process":rewards.mean().item()},
                                step=step)
                
            accelerator.backward(sql_loss)
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
                        # print(name_param)
                        # print(name_param_)
                        # if "mlp" in name_param:
                        #     print("Hi")
                        # TODO check if the paras are updating in a expected way...
                        # TODO 需要check是不是只有mlp在变，其他都不变
                        param_.data.copy_((1 - train_config.ref_learning_rate) * param_
                                        + train_config.ref_learning_rate * param.to(ref_lm_device))


    prompt_model_for_save = accelerator.unwrap_model(prompt_model)
    prompt_model_for_save.save_pretrained(
    "./my_save_ckpt",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
    state_dict=accelerator.get_state_dict(prompt_model),
)
    


# for stage 3 pls refer to https://huggingface.co/docs/accelerate/usage_guides/deepspeed#saving-and-loading for saving



if __name__ == "__main__":
    main()
