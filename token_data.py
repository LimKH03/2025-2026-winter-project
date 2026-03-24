import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from token_config import LABEL_LIST, LABEL2ID, ID2LABEL

def load_jsonl_lazy(path):
    samples = []
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def align_labels_to_tokens_sliding_window(text, char_labels, tokenizer, max_length, stride):
    """최초 모델의 핵심: Passage/Question 없이 Answer 텍스트만 처리하던 로직"""
    enc = tokenizer(text, truncation=False, return_offsets_mapping=True, padding=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    
    token_labels = []
    for start, end in offsets:
        # Special tokens (CLS, SEP 등)
        if start == end == 0:
            token_labels.append(-100)
            continue
        
        is_hal = False
        is_first = False
        for s, e in char_labels:
            if start < e and end > s:
                is_hal = True
                if start <= s: is_first = True
                break
        
        if is_hal:
            token_labels.append(LABEL2ID["B-HAL"] if is_first else LABEL2ID["I-HAL"])
        else:
            token_labels.append(LABEL2ID["O"])

    chunks = []
    total = len(ids)
    if total <= max_length:
        pad = max_length - total
        chunks.append({
            "input_ids": ids + [tokenizer.pad_token_id] * pad,
            "attention_mask": [1] * total + [0] * pad,
            "labels": token_labels + [-100] * pad
        })
    else:
        step = max_length - stride
        for i in range(0, total, step):
            end = min(i + max_length, total)
            c_ids = ids[i:end]
            c_labels = token_labels[i:end]
            pad = max_length - len(c_ids)
            chunks.append({
                "input_ids": c_ids + [tokenizer.pad_token_id] * pad,
                "attention_mask": [1] * len(c_ids) + [0] * pad,
                "labels": c_labels + [-100] * pad
            })
            if end == total: break
    return chunks

class TokenHalDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length, stride):
        self.features = []
        for s in tqdm(samples, desc="Preprocessing"):
            text = s.get("llm_answer", "")
            char_labels = s.get("labels", [])
            self.features.extend(align_labels_to_tokens_sliding_window(text, char_labels, tokenizer, max_length, stride))
            
    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.features[idx].items()}
