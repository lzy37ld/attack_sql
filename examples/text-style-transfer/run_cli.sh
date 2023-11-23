
# base:
# 并不是完全的alternate因为有的没有control

accelerate launch --config_file /home/liao.629/rl-prompt-lzy/examples/text-style-transfer/accelerate_config.yaml --main_process_port 1235 --gpu_ids 0,1,2,3 run_tst.py trainer.training_mode=sql-offpolicy
accelerate launch --config_file /home/liao.629/rl-prompt-lzy/examples/text-style-transfer/accelerate_config.yaml --main_process_port 1236 --gpu_ids 4,5,6,7 run_tst.py trainer.training_mode=sql-onpolicy
accelerate launch --config_file /home/liao.629/rl-prompt-lzy/examples/text-style-transfer/accelerate_config.yaml --main_process_port 1237 --gpu_ids 1,2,3,4 run_tst.py trainer.training_mode=sql-offpolicy handler.compute_zscore=true handler.reward_shaping=true
accelerate launch --config_file /home/liao.629/rl-prompt-lzy/examples/text-style-transfer/accelerate_config.yaml --main_process_port 1238 --gpu_ids 4,5,6,7 run_tst.py trainer.training_mode=sql-onpolicy handler.compute_zscore=true handler.reward_shaping=true
