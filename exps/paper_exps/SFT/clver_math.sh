### Clver Math
# Idefics2
exp_name="dbg_clver_math_idefic2_sft" \
train_file='/network/projects/aishwarya_lab/datasets/clevr-math/' \
test_file='/network/projects/aishwarya_lab/datasets/clevr-math/' \
engine='python' \
model_name_or_path='/home/mila/s/sarvjeet-singh.ghotra/scratch/git/rl_ft_vlm_tools/pre_trained_models/idefics2-8b' \
tokenizer_name_or_path='/home/mila/s/sarvjeet-singh.ghotra/scratch/git/rl_ft_vlm_tools/pre_trained_models/idefics2-8b' \
n_epochs='2' \
    bash exps/paper_exps/SFT/_template_vlm_tools.sh

