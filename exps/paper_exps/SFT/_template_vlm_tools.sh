#!/bin/bash
export TOKENIZERS_PARALLELISM=True

### Required variables
exp_name=${exp_name:-''}
train_file=${train_file:-''}
test_file=${test_file:-''}
engine=${engine:-''}
model_name_or_path=${model_name_or_path:-''}
tokenizer_name_or_path=${tokenizer_name_or_path:-''}
n_epochs=${n_epochs:-''}

### Default variables
model_dir="ppo_paper_final_new/_models_outputs_sft/${exp_name}/"
# config_file="./default_config_deepspeed_ga2.yaml"

use_lora="False"
use_qlora="True"

batch_size="32"
eval_batch_size="48"
gradient_accumulation_steps="1"
max_input_length="1024"
num_workers="8"
learning_rate="1e-4"
weight_decay="0.01"
warmup_step="50"
clip_grad_norm="1"
seed="42"
keep_num_ckpt='10'

logging_epoch_freq="1"
evaluating_epoch_freq="1"
saving_epoch_freq="1"

logging_step_freq="10"
evaluating_step_freq="2000"
saving_step_freq="2000"

wandb_log="True"
wandb_project="RL_FT_VLM_tools"
wandb_run_name="dbg_${exp_name}"
#########

num_processes='1'
main_process_port='8888'

# FIXME: remove debug flag
# FIXME: num_processes=1

# --config_file "${config_file}" \
# --debug \

mkdir -p "${model_dir}"
accelerate launch \
            --num_processes=${num_processes} \
            --main_process_port=${main_process_port} \
    train_sft_model_vlm_tools.py \
            --model_name_or_path "${model_name_or_path}" \
            --tokenizer_name_or_path "${tokenizer_name_or_path}" \
            --train_file "${train_file}" \
            --test_file "${test_file}" \
            --model_dir "${model_dir}" \
            --batch_size "${batch_size}" \
            --eval_batch_size "${eval_batch_size}" \
            --n_epochs "${n_epochs}" \
            --num_workers "${num_workers}" \
            --learning_rate "${learning_rate}" \
            --weight_decay "${weight_decay}" \
            --warmup_step "${warmup_step}" \
            --clip_grad_norm "${clip_grad_norm}" \
            --evaluating_epoch_freq "${evaluating_epoch_freq}" \
            --logging_epoch_freq "${logging_epoch_freq}" \
            --saving_epoch_freq "${saving_epoch_freq}" \
            --evaluating_step_freq "${evaluating_step_freq}" \
            --logging_step_freq "${logging_step_freq}" \
            --saving_step_freq "${saving_step_freq}" \
            --seed "${seed}" \
            --max_input_length "${max_input_length}" \
            --gradient_accumulation_steps "${gradient_accumulation_steps}" \
            --keep_num_ckpt "${keep_num_ckpt}" \
            --wandb_log "${wandb_log}" \
            --wandb_project "${wandb_project}" \
            --wandb_run_name "${wandb_run_name}" \
            --engine "${engine}" \
            --use_lora "${use_lora}" \
            --use_qlora "${use_qlora}" \
            1> >(tee "${model_dir}"/"${exp_name}".log) \
            2> >(tee "${model_dir}"/"${exp_name}".err >&2)
