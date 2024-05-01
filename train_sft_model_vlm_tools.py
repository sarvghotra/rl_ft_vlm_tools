# Copyright 2023 Bytedance Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import pad_across_processes, broadcast
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
from datetime import timedelta
from functools import partial
import json
import os
import random
from models.create_model import create_idefics_model
from src.python_engine import run_python_code
from src.utils import set_seed, floatify, compute_ETA
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, AdamW, get_constant_schedule_with_warmup
import wandb
import pandas as pd
import shutil
from dataset.clver_math import ClverMathDataset, MyDataCollator, ID_TO_TEMPLATE
tqdm = partial(tqdm, ncols=0, leave=False)

TIMEOUT = 10
instruction=None
cot_trigger=None
answer_trigger=None
def setup_cot(src_name):
    assert src_name in ['gsm8k', 'mathqa', 'svamp', 'mathqa-numeric']
    global instruction
    global cot_trigger
    global answer_trigger
    # Complete output is in this form: f'{instruction}{question.strip()}{cot_trigger}{answer_cot.strip()}'
    instruction = 'Question:\n'
    cot_trigger = '\nAnswer reasoning:\n'
    answer_trigger = '\nTherefore, the answer is: '
    return

post_process_final_answer_fn_mapper = {
    'gsm8k': lambda x: float(x.replace(',','').strip()),
    'svamp': lambda x: float(x.replace(',','').strip()),
    'mathqa': lambda x: x.lower().replace('"','').replace("'",'').strip(),
    'mathqa-numeric': lambda x: float(x),
}
### the answer_cot is a list of answer_cot
post_process_answer_cot_fn_mapper = {
    ('python', 'gsm8k'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('python', 'svamp'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('python', 'mathqa'): lambda answer_cot: [str(res).lower().replace('"','').replace("'",'').strip() for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('python', 'mathqa-numeric'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('nl', 'gsm8k'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    ('nl', 'svamp'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    ('nl', 'mathqa'): lambda answer_cot: [res.split(answer_trigger)[-1].lower().replace('"','').replace("'",'').strip() for res in answer_cot],
    ('nl', 'mathqa-numeric'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
}
compare_answer_fn_mapper = {
    'gsm8k': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    'svamp': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    'mathqa': lambda extracted_ans, target_answer: extracted_ans == target_answer,
    'mathqa-numeric': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
}

def prepare_datasets_and_data_loaders(args, tokenizer):
    with accelerator.main_process_first():
        raw_dataset = DatasetDict({
            'train': Dataset.from_list(json.load(open(args['train_file'],'r'))),
            'test': Dataset.from_list(json.load(open(args['test_file'],'r'))),
        })
        accelerator.print('Raw data:', raw_dataset)
        src_name = raw_dataset['train'][0]['item_id'].split('_')[0]  # e.g., gsm8k_0, gsm8k_1, gsm8k_2, ...
        setup_cot(src_name)
        accelerator.print('Using instruction:', instruction)
        accelerator.print('Using cot_trigger:', cot_trigger)
        accelerator.print('Using answer_trigger:', answer_trigger)
        def tokenize_fn(batch, args, tokenizer):
            assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
            new_batch = defaultdict(list)
            all_keys = list(batch.keys())
            for item_values in zip(*(batch[k] for k in all_keys)):
                item = {k: item_values[i] for i, k in enumerate(all_keys)}
                item_id, question, answer_value, answer_cot = \
                        item['item_id'], \
                        item['question'], \
                        item['answer_value'], \
                        item.get('answer_cot', None), \

                question = question.strip()
                if answer_value is not None:
                    answer_value = answer_value.strip()

                if answer_cot is not None:
                    answer_cot = answer_cot.strip()
                    if args['engine'] == 'nl':
                        answer_cot += f'{answer_trigger}{answer_value}'

                input = f'{instruction}{question}{cot_trigger}'
                output = f'{answer_cot}'
                prefix_text = f'{instruction}{question}{cot_trigger}'

                input_encode = tokenizer(input, add_special_tokens=False)
                output_encode = tokenizer(output, add_special_tokens=False)
                prefix_encode = tokenizer(prefix_text, add_special_tokens=False)

                input_ids = input_encode['input_ids'] + output_encode['input_ids'] + [tokenizer.eos_token_id]
                labels = [-100]*len(input_encode['input_ids']) + output_encode['input_ids'] + [tokenizer.eos_token_id]
                attention_mask = [1]* len(input_ids)
                prefix = prefix_encode['input_ids']
                prefix_attention_mask = prefix_encode['attention_mask']

                # Truncation
                input_ids_max_length = len(input_ids)
                # assert input_ids_max_length <= args['max_input_length'], input_ids_max_length
                input_ids = input_ids[:args['max_input_length']]
                labels = labels[:args['max_input_length']]
                attention_mask = attention_mask[:args['max_input_length']]
                prefix = prefix[:args['max_input_length']]
                prefix_attention_mask = prefix_attention_mask[:args['max_input_length']]

                ##
                new_batch['input_ids'].append(input_ids)
                new_batch['labels'].append(labels)
                new_batch['attention_mask'].append(attention_mask)
                new_batch['prefix'].append(prefix)
                new_batch['prefix_attention_mask'].append(prefix_attention_mask)
                ##
                new_batch['item_id'].append(item_id)
                new_batch['question'].append(question)
                new_batch['answer_cot'].append(answer_cot)
                new_batch['answer_value'].append(answer_value)
                new_batch['input_ids_max_length'].append(input_ids_max_length)

            return new_batch

        tokenized_dataset = DatasetDict({
            mode: dataset.map(
                tokenize_fn, fn_kwargs={'args': args, 'tokenizer': tokenizer}, batched=True, remove_columns=dataset.column_names,
                num_proc=8, load_from_cache_file=False
            ) for mode, dataset in raw_dataset.items()})
        accelerator.print('Processed data:', tokenized_dataset)
        for mode, dataset in tokenized_dataset.items():
            accelerator.print(mode, f'{mode}_input_ids_max_length', max(dataset['input_ids_max_length']))

        if accelerator.is_main_process and args['wandb_log']:
            wandb.config.update({
                "src_name": src_name,
                "instruction": instruction,
                "cot_trigger": cot_trigger,
                "answer_trigger": answer_trigger,
                "raw_dataset": str(raw_dataset),
                "tokenized_dataset": str(tokenized_dataset),
                "train_input_ids_max_length": max(tokenized_dataset['train']['input_ids_max_length']),
                "test_input_ids_max_length": max(tokenized_dataset['test']['input_ids_max_length']),
            })

    def collate_fn(batch, args, tokenizer):
        max_input_length = max([len(item['input_ids']) for item in batch])
        max_target_length = max([len(item['labels']) for item in batch])
        max_prefix_length = max([len(item['prefix']) for item in batch])
        input_ids  = []
        attention_mask  = []
        labels, labels_left_padded  = [], []
        prefix_left_padded  = []
        prefix_attention_mask_left_padded  = []
        for item in batch:
            input_ids.append(item['input_ids'] + [tokenizer.pad_token_id]*(max_input_length - len(item['input_ids'])))
            attention_mask.append(item['attention_mask'] + [0]*(max_input_length - len(item['attention_mask'])))
            labels.append(item['labels'] + [-100]*(max_target_length - len(item['labels'])))

            labels_left_padded.append([-100]*(max_target_length - len(item['labels'])) + item['labels'])
            prefix_left_padded.append([tokenizer.pad_token_id]*(max_prefix_length - len(item['prefix'])) + item['prefix'])
            prefix_attention_mask_left_padded.append([0]*(max_prefix_length - len(item['prefix_attention_mask'])) + item['prefix_attention_mask'])
        forward_kwargs = {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.BoolTensor(attention_mask),
            'labels': torch.LongTensor(labels)
        }
        generate_prefix_kwargs = {
            'input_ids': torch.LongTensor(prefix_left_padded),
            'attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
            'labels': torch.LongTensor(labels_left_padded)
        }
        return {
            'forward_kwargs': forward_kwargs,
            'generate_prefix_kwargs': generate_prefix_kwargs,
        }

    train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=args['batch_size'], num_workers=args['num_workers'], pin_memory=True,
                        collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))

    test_dataloader = DataLoader(tokenized_dataset['test'], shuffle=False, batch_size=args['eval_batch_size'], num_workers=args['num_workers'], pin_memory=True,
                        collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))

    return (tokenized_dataset['train'], train_dataloader), (tokenized_dataset['test'], test_dataloader)


def prepare_datasets_and_data_loaders_idefics2(args, processor):
    data_collator = MyDataCollator(processor)

    train_dataset = ClverMathDataset(args['train_file'], 'train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        collate_fn=data_collator,
        num_workers=args['num_workers'],
        pin_memory=True,
        shuffle=True,
    )

    val_dataset = ClverMathDataset(args['train_file'], 'validation')
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args['eval_batch_size'],
        collate_fn=data_collator,
        num_workers=args['num_workers'],
        pin_memory=True,
        shuffle=False,
    )

    test_dataset = ClverMathDataset(args['train_file'], 'test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args['eval_batch_size'],
        collate_fn=data_collator,
        num_workers=args['num_workers'],
        pin_memory=True,
        shuffle=False,
    )

    if accelerator.is_main_process and args['wandb_log']:
        wandb.config.update({
            "train_dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset),
            "test_dataset_size": len(test_dataset),
        })

    return (train_dataset, train_dataloader), (val_dataset, val_dataloader), (test_dataset, test_dataloader)


def do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths=None):
    os.makedirs(save_path, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_path, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
    tokenizer.save_pretrained(save_path)
    if accelerator.is_main_process and most_recent_ckpts_paths is not None:
        most_recent_ckpts_paths.append(save_path)
        if args['keep_num_ckpt'] is not None and len(most_recent_ckpts_paths) > args['keep_num_ckpt']:
            ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
            # os.remove(ckpt_to_be_removed)
            shutil.rmtree(ckpt_to_be_removed)

def train_one_epoch(args, model, train_dataset, train_dataloader, optimizer, scheduler, tokenizer,
                    processor, global_step, val_dataset, val_dataloader, test_dataset, test_dataloader,
                    prefix, epoch, best_eval_log_dict, summary_log_dict, most_recent_ckpts_paths):
    model_dir = args['model_dir']
    clip_grad_norm = args.get('clip_grad_norm', None)
    evaluating_step_freq = args.get('evaluating_step_freq', None)
    logging_step_freq = args.get('logging_step_freq', None)
    saving_step_freq = args.get('saving_step_freq', None)
    model.train()
    epoch_result_dict = defaultdict(list)

    with tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=not accelerator.is_main_process, desc='Train Loop') as t:
        for idx, batch in t:

            with accelerator.accumulate(model):
                output = model(**batch)
                # Get some metrics
                loss = output[0]
                result_dict, extra = {}, None
                # Update
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                # model.zero_grad()
                if accelerator.sync_gradients:
                    scheduler.step()

            if accelerator.sync_gradients:
                global_step += 1
                # Step update metric
                epoch_result_dict['loss'].append(loss.item())
                for k, v in result_dict.items():
                    epoch_result_dict[k].append(v)

                # Step evaluating
                eval_log_dict = {}
                test_log_dict = {}
                is_best = False
                if evaluating_step_freq is not None and global_step % evaluating_step_freq == 0:
                    accelerator.print(f"Step {global_step} Evaluating...")
                    evaluate_result_dict = {f'Eval.Gen.{k}':  v for k, v in evaluate_generation(args, model, val_dataset, val_dataloader, processor, tokenizer).items()}
                    eval_log_dict.update(evaluate_result_dict)
                    if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', 0):
                        is_best = True
                        best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']
                    if 'Eval.Gen.value_accuracy' not in summary_log_dict:
                        summary_log_dict['Eval.Gen.value_accuracy'] = []
                    summary_log_dict['Eval.Gen.value_accuracy'].append(eval_log_dict['Eval.Gen.value_accuracy'])

                    test_result_dict = {f'Test.Gen.{k}':  v for k, v in evaluate_generation(args, model, test_dataset, test_dataloader, processor, tokenizer).items()}
                    test_log_dict.update(test_result_dict)
                    if 'Test.Gen.value_accuracy' not in summary_log_dict:
                        summary_log_dict['Test.Gen.value_accuracy'] = []
                    summary_log_dict['Test.Gen.value_accuracy'].append(test_log_dict['Test.Gen.value_accuracy'])

                # Step logging
                train_log_dict = {}
                if logging_step_freq is not None and global_step % logging_step_freq == 0:
                    train_log_dict = {f'T.{k}': sum(v)/len(v) if isinstance(v, list) else v for k, v in epoch_result_dict.items()}

                if eval_log_dict or train_log_dict:
                    log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict, **test_log_dict}
                    if accelerator.is_main_process and args['wandb_log']:
                        wandb.log(log_dict, step=global_step)
                        log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'], **log_dict}
                    log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k,v in log_dict.items()}
                    accelerator.print(f"{prefix}[E={epoch}/{args['n_epochs']}, S={global_step}] {log_dict}")

                # Step saving
                if saving_step_freq is not None and global_step % saving_step_freq == 0:
                    accelerator.print(f"Step {global_step} Saving...")
                    if is_best:
                        save_path = os.path.join(model_dir, f'best')
                        do_checkpoint(args, model, tokenizer, save_path)
                    if args['keep_num_ckpt'] > 0:
                        save_path = os.path.join(model_dir, f'global_step_{str(global_step)}')
                        do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)

                # Keep only max_record items
                for k, v in epoch_result_dict.items():
                    if len(v) > 1:
                        epoch_result_dict[k] = v[-1:]


    # Metric summary:
    epoch_result_dict = {k:(sum(v)/len(v) if isinstance(v, list) else v) for k, v in epoch_result_dict.items()}
    return epoch_result_dict, global_step

def evaluate_generation(args, model, dataset, dataloader, processor, tokenizer):
    model.eval()
    predictions = []
    targets = []
    templates = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process, desc='Evaluation Gen Loop'):
        batch_templates = batch['templates']
        # remove 'templates' from batch
        batch.pop('templates')
        output_ = model.module.generate(
                        **batch,
                        # max_length=args['max_input_length'],
                        # output_scores=True,
                        return_dict_in_generate=True,
                        num_beams=1,
                        use_cache=True,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=16,
                    )

        generated_ids = output_.sequences
        generated_ids = pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)
        labels = batch['labels']
        # labels = pad_across_processes(labels, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)
        # labels[labels==-100]=tokenizer.pad_token_id

        generated_ids, labels = accelerator.gather(generated_ids), accelerator.gather(labels)

        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        pred_ans = [x.split("Assistant: ")[1][:-1].strip() for x in generated_texts]    # -1 removes "."
        # pred_ans = pred_ans.strip()

        # generated_ids = output_.sequences
        # generated_ids = pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)

        # labels = batch['generate_prefix_kwargs']['labels']
        # labels = pad_across_processes(labels, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)
        # labels[labels==-100]=tokenizer.pad_token_id

        # generated_ids, labels = accelerator.gather(generated_ids), accelerator.gather(labels)

        # preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in generated_ids]
        predictions.extend(pred_ans)
        # target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for t in labels]
        targets.extend(labels)
        templates.extend(batch_templates)

    predictions = predictions[:len(dataset)]
    targets = targets[:len(dataset)]
    templates = templates[:len(dataset)]

    if accelerator.is_main_process and accelerator.is_local_main_process:
        results = []
        corr = 0
        total = 0
        temp_based_acc = {}
        for pred, tar, temp in zip(predictions, targets, templates):
            tar = str(tar.detach().cpu().numpy())
            cur_res = {
                'pred': pred,
                'target': tar,
            }
            results.append(cur_res)

            temp = str(temp.detach().cpu().numpy())
            if temp not in temp_based_acc:
                temp_based_acc[temp] = {}
                temp_based_acc[temp]['corr'] = 0
                temp_based_acc[temp]['total'] = 0

            if pred == tar:
                corr += 1
                temp_based_acc[temp]['corr'] += 1
            total += 1
            temp_based_acc[temp]['total'] += 1

        # save first before execute to trace error.
        res_path = args['model_dir'].rstrip('/')+ '/' + '_res.json'
        accelerator.print(f"Saving results to {res_path}")
        with open(res_path, 'w') as f:
            json.dump(results, f, indent=2)

        # if args['wandb_log']:
        #     table = wandb.Table(dataframe=pd.DataFrame(results))
        #     wandb.log({"predictions": table})

        value_accuracy = corr / len(results) * 100
        accelerator.print(f"[Eval Info] value_accuracy: {value_accuracy:.5g}%")
        value_accuracy = torch.FloatTensor([value_accuracy]).to(accelerator.device)

        templates_value_accuracy = {}
        for temp, acc in temp_based_acc.items():
            acc = acc['corr'] / acc['total'] * 100
            temp_name = ID_TO_TEMPLATE[int(temp)]
            accelerator.print(f"[Eval Info] value_accuracy on {temp_name} category: {acc:.5g}%")
            templates_value_accuracy[temp_name] = torch.FloatTensor([acc]).to(accelerator.device)
    else:
        value_accuracy = torch.FloatTensor([-1.0]).to(accelerator.device)
        templates_value_accuracy = {k: torch.FloatTensor([-1.0]).to(accelerator.device) for k in ID_TO_TEMPLATE.values()}

    value_accuracy = broadcast(value_accuracy).cpu().numpy().tolist()[0]
    for temp_category, acc in templates_value_accuracy.items():
        templates_value_accuracy[temp_category] = broadcast(acc).cpu().numpy().tolist()[0]

    # Metric summary:
    out_stats = {'value_accuracy': value_accuracy}
    for temp_category, acc in templates_value_accuracy.items():
        out_stats[temp_category] = acc

    model.train()
    return out_stats

def main(args):
    set_seed(args['seed'] + accelerator.process_index)
    if torch.distributed.get_rank() == 0 and args['wandb_log']:
        wandb.init(project=args['wandb_project'], name=args['wandb_run_name'])
        wandb.config.update(args)

    model, tokenizer, processor = create_idefics_model(args)
    (train_dataset, train_dataloader), (val_dataset, val_dataloader), (test_dataset, test_dataloader) = prepare_datasets_and_data_loaders_idefics2(args, processor)
    accelerator.print(f'[Vocab size]: {len(tokenizer)}')

    if accelerator.is_main_process and args['wandb_log']:
        wandb.run.summary.update({
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'unk_token_id': tokenizer.unk_token_id,
            'vocab_size': len(tokenizer)
        })

    n_epochs = args['n_epochs']
    num_training_steps = (len(train_dataloader) // accelerator.num_processes * n_epochs) // args['gradient_accumulation_steps']
    warmup_step = args['warmup_step'] if args['warmup_step'] is not None and args['warmup_step'] >= 0 else int(0.1 * num_training_steps)
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
    #         "weight_decay": args['weight_decay'],
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
    #         "weight_decay": 0.0,
    #     },
    # ]
    optimizer = AdamW(model.parameters(), lr=args['learning_rate'], eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps)
    # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step)
    accelerator.print(
        f"***** Running training *****\n"
        f"  Num examples = {len(train_dataset)}\n"
        f"  Num Epochs = {n_epochs}\n"
        f"  Instantaneous batch size per device = {args['batch_size']}\n"
        f"  Total train batch size (w. parallel, distributed & accumulation) = {args['batch_size']*accelerator.num_processes*args['gradient_accumulation_steps']}\n"
        f"  Total optimization steps = {num_training_steps}\n"
        f"  Warm up step: {warmup_step}\n"
        f"  Learning rate: {args['learning_rate']}\n"
    )
    model, optimizer, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, test_dataloader)

    global_step = 0
    evaluating_epoch_freq = args['evaluating_epoch_freq']
    logging_epoch_freq = args['logging_epoch_freq']
    saving_epoch_freq = args['saving_epoch_freq']
    model_dir=args['model_dir']
    best_eval_log_dict = {}
    summary_log_dict = {}
    os.makedirs(model_dir, exist_ok=True)
    most_recent_ckpts_paths = []
    with tqdm(range(1, n_epochs+1), total=n_epochs, disable=False) as t:
        for epoch in t:
            kwargs = {
                'args': args,
                'model': model,
                'train_dataset': train_dataset,
                'train_dataloader': train_dataloader,
                'val_dataset': val_dataset,
                'val_dataloader': val_dataloader,
                'test_dataset': test_dataset,
                'test_dataloader': test_dataloader,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'global_step': global_step,
                'processor': processor,
                'tokenizer': tokenizer,
                'prefix':'',
                'epoch': epoch,
                'best_eval_log_dict': best_eval_log_dict,
                'summary_log_dict': summary_log_dict,
                'most_recent_ckpts_paths': most_recent_ckpts_paths,
            }
            train_epoch_result_dict, global_step = train_one_epoch(**kwargs)

            eval_log_dict = {}
            test_log_dict = {}
            is_best = False
            if evaluating_epoch_freq is not None and epoch % evaluating_epoch_freq == 0:
                evaluate_result_dict = {f'Eval.Gen.{k}':  v for k, v in evaluate_generation(args, model, val_dataset, val_dataloader, processor, tokenizer).items()}
                eval_log_dict.update(evaluate_result_dict)
                if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', 0):
                    is_best = True
                    best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']
                if 'Eval.Gen.value_accuracy' not in summary_log_dict:
                    summary_log_dict['Eval.Gen.value_accuracy'] = []
                summary_log_dict['Eval.Gen.value_accuracy'].append(eval_log_dict['Eval.Gen.value_accuracy'])

                test_result_dict = {f'Test.Gen.{k}':  v for k, v in evaluate_generation(args, model, test_dataset, test_dataloader, processor, tokenizer).items()}
                test_log_dict.update(test_result_dict)
                if 'Test.Gen.value_accuracy' not in summary_log_dict:
                    summary_log_dict['Test.Gen.value_accuracy'] = []
                summary_log_dict['Test.Gen.value_accuracy'].append(test_log_dict['Test.Gen.value_accuracy'])

            train_log_dict = {}
            if logging_epoch_freq is not None and epoch % logging_epoch_freq == 0:
                train_log_dict = {f'T.{k}': sum(v)/len(v) if isinstance(v, list) else v for k, v in train_epoch_result_dict.items()}

            if eval_log_dict or train_log_dict:
                log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict, **test_log_dict}
                if accelerator.is_main_process and args['wandb_log']:
                    wandb.log(log_dict, step=global_step)
                    log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'], **log_dict}
                log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k,v in log_dict.items()}
                accelerator.print(f"[E={epoch}/{args['n_epochs']}, S={global_step}] {log_dict}")

            if saving_epoch_freq is not None and epoch % saving_epoch_freq == 0:
                if is_best:
                    save_path = os.path.join(model_dir, f'best')
                    do_checkpoint(args, model, tokenizer, save_path)
                if args['keep_num_ckpt'] > 0:
                    save_path=os.path.join(args['model_dir'], f'global_step_{str(global_step)}_epoch_{epoch}')
                    do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)

    return


if __name__ == '__main__':
    from transformers import HfArgumentParser
    NONE_INT = -100
    NONE_STR = 'None'
    @dataclass
    class Arguments:
        model_name_or_path: str
        tokenizer_name_or_path: str
        model_dir: str
        train_file: str
        test_file: str
        batch_size: int = field(default=4)
        eval_batch_size: int = field(default=8)
        n_epochs: int = field(default=40)
        num_workers: int = field(default=8)
        learning_rate: float = field(default=2e-5)
        weight_decay: float = field(default=1e-6)
        warmup_step: int = field(default=0)
        clip_grad_norm: float = field(default=1)
        evaluating_epoch_freq: int = field(default=1)
        logging_epoch_freq: int = field(default=1)
        saving_epoch_freq: int = field(default=1000)
        evaluating_step_freq: int = field(default=NONE_INT)
        logging_step_freq: int = field(default=NONE_INT)
        saving_step_freq: int = field(default=NONE_INT)
        seed: int = field(default=42)
        max_input_length: int = field(default=700)
        gradient_accumulation_steps: int = field(default=1)
        keep_num_ckpt: int = field(default=1)
        # wandb stuff
        wandb_log: bool = field(default=False)
        wandb_project: str = field(default='tmp_anvfupsadfn')
        wandb_run_name: str = field(default='default_run_name')
        ###
        engine: str = field(default='python')
        # lora
        use_lora: bool = field(default=False)
        use_qlora: bool = field(default=False)

    parser = HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = asdict(args)
    for k,v in args.items():
        if v in [NONE_INT, NONE_STR]:
            args[k] = None
    accelerator = Accelerator(gradient_accumulation_steps=args['gradient_accumulation_steps'], kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))]) # wait for processing upto 5hrs
    accelerator.print(args)
    accelerator.print(json.dumps(args, indent=2, ensure_ascii=False))
    main(args)
