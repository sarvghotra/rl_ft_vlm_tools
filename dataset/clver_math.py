import torch
from torch.utils.data import Dataset
from datasets import load_dataset

TEMPLATE_TO_ID = {
    'subtraction': 0,
    'addition': 1,
    'adversarial': 2,
    'subtraction-multihop': 3
}

ID_TO_TEMPLATE = {
    0: 'subtraction',
    1: 'addition',
    2: 'adversarial',
    3: 'subtraction-multihop'
}


class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, examples):
        texts = []
        images = []
        answers = []
        temps = []
        for example in examples:
            image = example["image"]
            question = example["query"]
            answer = example["answer"]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            if example["split"] == "train":
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer}
                        ]
                    }
                )
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False if example["split"] == "train" else True)
            texts.append(text.strip())
            images.append([image])
            answers.append(answer)

            if examples[0]['split'] != 'train':
                temps.append(TEMPLATE_TO_ID[example["template"]])

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        ans_tokens = torch.IntTensor(answers)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels
        batch["answers"] = ans_tokens

        if examples[0]['split'] != 'train':
            batch["labels"] = ans_tokens
            batch["templates"] = torch.IntTensor(temps)

        return batch


class ClverMathDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.dataset = load_dataset('dali-does/clevr-math',
                               cache_dir=data_dir,
                               split=split)
        self.split = split.lower()
        self.len = len(self.dataset)
        if self.split != 'train':
            # self.dataset = self.dataset[:100]
            # FIXME: change it to something larger
            self.len = 1600

    def __getitem__(self, idx):
        ques = self.dataset[idx]['question']
        label = self.dataset[idx]['label']
        img = self.dataset[idx]['image'].convert('RGB')
        template = self.dataset[idx]['template']
        dp = {
            'query': ques,
            'image': img,
            'answer': label,
            'split': self.split
        }
        if self.split != 'train':
            dp['template'] = template
        return dp

    def __len__(self):
        return self.len
