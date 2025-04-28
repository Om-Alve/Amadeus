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
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask