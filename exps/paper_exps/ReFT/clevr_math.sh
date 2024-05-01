### GSM8K
## Python SDP
# Codellama

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

