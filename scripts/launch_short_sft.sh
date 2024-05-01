#!/bin/bash

#SBATCH --job-name=rl_r1
#SBATCH --output=logs/output_rl_r1.txt
#SBATCH --error=logs/error_rl_r1.txt
#SBATCH --time=3:00:00
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short-unkillable
#SBATCH --nodes=1

module load anaconda
conda activate idefics2
module load cuda/12.1.1
cd ..


exp_name="clevr_rl_r1" \
train_file='/network/projects/aishwarya_lab/datasets/clevr-math/' \
test_file='/network/projects/aishwarya_lab/datasets/clevr-math/' \
engine='python' \
model_name_or_path='/home/mila/s/sarvjeet-singh.ghotra/scratch/git/rl_ft_vlm_tools/pre_trained_models/idefics2-8b' \
ref_model_name_or_path='/home/mila/s/sarvjeet-singh.ghotra/scratch/git/rl_ft_vlm_tools/pre_trained_models/idefics2-8b' \
tokenizer_name_or_path='/home/mila/s/sarvjeet-singh.ghotra/scratch/git/rl_ft_vlm_tools/pre_trained_models/idefics2-8b' \
n_epochs='2' \
kl_coef='0.01' \
    bash exps/paper_exps/ReFT/_template_vlm.sh









### Clver Math
# Idefics2
# exp_name="clver_math_idefic2_sft" \
# train_file='/network/projects/aishwarya_lab/datasets/clevr-math/' \
# test_file='/network/projects/aishwarya_lab/datasets/clevr-math/' \
# engine='python' \
# model_name_or_path='/home/mila/s/sarvjeet-singh.ghotra/scratch/git/rl_ft_vlm_tools/pre_trained_models/idefics2-8b-base' \
# tokenizer_name_or_path='/home/mila/s/sarvjeet-singh.ghotra/scratch/git/rl_ft_vlm_tools/pre_trained_models/idefics2-8b-base' \
# n_epochs='2' \
#     bash exps/paper_exps/SFT/_template_vlm_tools.sh
