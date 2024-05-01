import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from transformers import Idefics2ForConditionalGeneration
from transformers import BitsAndBytesConfig
from trl import get_kbit_device_map, get_quantization_config
from peft import LoraConfig, get_peft_model
from trl import AutoModelForCausalLMWithValueHead


class Idefics2ForConditionalGenerationwithValueHead(AutoModelForCausalLMWithValueHead):
    transformers_parent_class = Idefics2ForConditionalGeneration
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )
    def __init__(self, pretrained_model, **kwargs):
        pretrained_model.config.hidden_size = pretrained_model.config.text_config.hidden_size
        super().__init__(pretrained_model, **kwargs)


def create_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args['tokenizer_name_or_path'], use_fast=True)
    tokenizer.pad_token_id = 1
    tokenizer.eos_token_id = 2

    model = AutoModelForCausalLM.from_pretrained(args['model_name_or_path'], low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer, None


def create_idefics_model(args):
    # FIXME: implement quantization
    # quant_config = {
    #     "use_peft": True,
    #     "lora_r": 64,
    #     "lora_alpha": 16,
    #     "lora_target_modules": "all-linear"
    # }
    # quantization_config = get_quantization_config(quant_config)
    # quantization_config = None
    # model_kwargs = dict(
    #     revision=args['model_config_model_revision'],
    #     trust_remote_code=args['model_config_trust_remote_code'],
    #     attn_implementation=args['model_config_attn_implementation'],
    #     torch_dtype=args['torch_dtype'],
    #     use_cache=False,
    #     device_map=get_kbit_device_map() if quantization_config is not None else None,
    #     quantization_config=quantization_config,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(args['model_name_or_path'], use_fast=True)
    # # tokenizer.chat_template = IDEFICS2_CHAT_TEMPLATE
    # processor = AutoProcessor.from_pretrained(args['model_name_or_path'])
    # processor.tokenizer = tokenizer
    processor = AutoProcessor.from_pretrained(
        args['model_name_or_path'],
        do_image_splitting=False
    )

    # model = Idefics2ForConditionalGeneration.from_pretrained(args['model_name_or_path'], **model_kwargs)
    if args['use_qlora'] or args['use_lora']:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
            use_dora=False if args['use_qlora'] else True,
            init_lora_weights="gaussian"
        )
        if args['use_qlora']:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        model = Idefics2ForConditionalGeneration.from_pretrained(
            args['model_name_or_path'],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config if args['use_qlora'] else None,
        )
        # model.add_adapter(lora_config)
        # model.enable_adapters()
        model = get_peft_model(model, lora_config)
    else:
        model = Idefics2ForConditionalGeneration.from_pretrained(
            args['model_name_or_path'],
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2", # Only available on A100 or H100
        )

    model.print_trainable_parameters()
    return model, processor.tokenizer, processor


def create_idefics_model_rl(args):
    # FIXME: implement quantization
    # quant_config = {
    #     "use_peft": True,
    #     "lora_r": 64,
    #     "lora_alpha": 16,
    #     "lora_target_modules": "all-linear"
    # }
    # quantization_config = get_quantization_config(quant_config)
    # quantization_config = None
    # model_kwargs = dict(
    #     revision=args['model_config_model_revision'],
    #     trust_remote_code=args['model_config_trust_remote_code'],
    #     attn_implementation=args['model_config_attn_implementation'],
    #     torch_dtype=args['torch_dtype'],
    #     use_cache=False,
    #     device_map=get_kbit_device_map() if quantization_config is not None else None,
    #     quantization_config=quantization_config,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(args['model_name_or_path'], use_fast=True)
    # # tokenizer.chat_template = IDEFICS2_CHAT_TEMPLATE
    # processor = AutoProcessor.from_pretrained(args['model_name_or_path'])
    # processor.tokenizer = tokenizer
    processor = AutoProcessor.from_pretrained(
        args['model_name_or_path'],
        do_image_splitting=False
    )

    # model = Idefics2ForConditionalGeneration.from_pretrained(args['model_name_or_path'], **model_kwargs)
    if args['use_qlora'] or args['use_lora']:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
            use_dora=False if args['use_qlora'] else True,
            init_lora_weights="gaussian"
        )
        if args['use_qlora']:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        model = Idefics2ForConditionalGenerationwithValueHead.from_pretrained(
            args['model_name_or_path'],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config if args['use_qlora'] else None,
            peft_config=lora_config
        )
        # model.add_adapter(lora_config)
        # model.enable_adapters()
        # model = get_peft_model(model, lora_config)
    else:
        model = Idefics2ForConditionalGenerationwithValueHead.from_pretrained(
            args['model_name_or_path'],
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2", # Only available on A100 or H100
        )

    # model.print_trainable_parameters()
    return model, processor.tokenizer, processor
