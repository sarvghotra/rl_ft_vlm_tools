# Learning planning and tool using in VLMs using RL

## Instructions for setting up the environment
### Install dependencies
```
conda env create -f environment.yml
```
### Download pre-trained models
```
mkdir pre_trained_models
cd pre_trained_models
git lfs install
git clone https://huggingface.co/HuggingFaceM4/idefics2-8b
git clone https://huggingface.co/HuggingFaceM4/idefics2-8b-base
```


## Instructions for running experiments

### Instructions for our method

#### VLM-tools Finetune with RL
```
bash exps/paper_exps/ReFT/clevr_math.sh
```



### Instructions for baselines

#### IDEFICS-2 Base (inference):
```
eval_idefics2_clevr_math.ipynb
```

#### Supervised Fine Tuning
```
bash exps/paper_exps/SFT/clevr_math.sh
```

#### Supervised Fine Tuned model inference
Note: we used the SFT on CLEVR-Math weights released by Huggingface on [link]()
```
eval_idefics2SFT_clevr_math.ipynb
```


## Additional changes made to make the base repo work for our method submitted as a requirement for the project:
1. Dataset code: dataset/clver_math.py
2. Model code: models/create_model.py
3. QLoRA code:
4. TRL for IDEFICS2 (it's not currently supported in TRL):
5. Code changes for working with CLEVR-Math dataset including evaluation and reward computing:
6. Notebooks for evaluting baselines:
    - eval_idefics2_clevr_math.ipynb
    - eval_idefics2SFT_clevr_math.ipynb
