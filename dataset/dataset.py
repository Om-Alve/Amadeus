import json
from torch.utils.data import Dataset
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, max_length=512):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Get the total number of lines in the file
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.total_samples = sum(1 for _ in f)
        # No need to load all data at once
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, index):
        # Seek to the right line in the file - using linecache for efficiency
        import linecache
        line = linecache.getline(self.dataset_path, index + 1).strip()
        if not line:
            # Handle empty lines
            line = "{\"text\": \"\"}"
        
        sample = json.loads(line)
        
        encoding = self.tokenizer(
            str(sample["text"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encoding.input_ids.squeeze()

        X = input_ids[:-1].clone().detach()
        Y = input_ids[1:].clone().detach()
        loss_mask = (input_ids[1:] != self.tokenizer.pad_token_id).clone().detach()

        return X, Y, loss_mask

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.total_samples = sum(1 for _ in f)
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        import linecache
        line = linecache.getline(self.jsonl_path, index + 1).strip()
        if not line:
            # Handle empty lines if any
            return self.__getitem__((index + 1) % len(self))
        
        sample = json.loads(line)
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

