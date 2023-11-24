
# base:
# 并不是完全的alternate因为有的没有control

accelerate launch --config_file ./accelerate_config.yaml --main_process_port 1235 --gpu_ids 0,1,2,3 run_tst.py trainer.training_mode=sql-offpolicy
accelerate launch --config_file ./accelerate_config.yaml --main_process_port 1237 --gpu_ids 1,2,3,4 run_tst.py trainer.training_mode=sql-offpolicy handler.compute_zscore=true handler.reward_shaping=true
accelerate launch --config_file ./accelerate_config.yaml --main_process_port 1238 --gpu_ids 4,5,6,7 run_tst.py trainer.training_mode=sql-offpolicy handler.compute_zscore=true handler.reward_shaping=true trainer.only_compute_for_gt0_when_off=True

accelerate launch --config_file ./accelerate_config.yaml --main_process_port 1235 --gpu_ids 0,1,2,3 run_tst.py trainer.training_mode=sql-offpolicy
accelerate launch --config_file ./accelerate_config.yaml --main_process_port 1237 --gpu_ids 1,2,3,4 run_tst.py trainer.training_mode=sql-offpolicy handler.compute_zscore=true handler.reward_shaping=true


# 应该是用mle来实现。
accelerate launch --config_file ./accelerate_config.yaml --main_process_port 1238 --gpu_ids 4,5,6,7 run_tst.py trainer.training_mode=sql-offpolicy handler.compute_zscore=true handler.reward_shaping=true trainer.only_compute_for_gt0_when_off=True



# because many adversarial prompt from llm-attack would not work..
accelerate launch --config_file ./accelerate_config.yaml --main_process_port 1239 --gpu_ids 1,2,3,4 run_tst.py trainer.training_mode=sql-offpolicy trainer.only_compute_for_gt0_when_off=True
accelerate launch --config_file ./accelerate_config.yaml --main_process_port 1240 --gpu_ids 1,2,3,4 run_tst.py trainer.training_mode=sql-offpolicy trainer.only_compute_for_gt0_when_off=True trainer.learning_rate=4e-5
accelerate launch --config_file ./accelerate_config.yaml --main_process_port 1241 --gpu_ids 0,1,2,3 run_tst.py trainer.training_mode=sql-offpolicy trainer.only_compute_for_gt0_when_off=True trainer.learning_rate=3e-5
accelerate launch --config_file ./accelerate_config.yaml --main_process_port 1239 --gpu_ids 1,2,3,4 run_tst.py trainer.training_mode=sql-offpolicy trainer.only_compute_for_gt0_when_off=True trainer.learning_rate=2e-5
accelerate launch --config_file ./accelerate_config.yaml --main_process_port 1239 --gpu_ids 1,2,3,4 run_tst.py trainer.training_mode=sql-offpolicy trainer.only_compute_for_gt0_when_off=True trainer.learning_rate=1e-5