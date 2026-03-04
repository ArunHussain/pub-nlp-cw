import html
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel


def clean_text(text):
    text = re.sub(r"<[^>]+>", "", str(text))
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


class PCLClassifier(nn.Module):
    def __init__(self, model_name, n_communities, dropout, grad_ckpt=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, return_dict=True)
        if grad_ckpt:
            self.backbone.gradient_checkpointing_enable()

        hidden_size = self.backbone.config.hidden_size
        merged = hidden_size + n_communities

        self.hidden = nn.Linear(merged, merged)
        self.classifier = nn.Linear(merged, 1)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, communities):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).type_as(hidden)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        pooled = self.dropout(pooled)
        x = torch.cat((pooled, communities), dim=1)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.classifier(x)


class PCLDataset(Dataset):
    def __init__(self, df, tokenizer, community_columns, max_len, text_col="text", label_col="label"):
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_col = text_col
        self.label_col = label_col

        dummies = pd.get_dummies(self.df["community"])
        dummies = dummies.reindex(columns=community_columns, fill_value=0)
        self.community_matrix = dummies.astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row[self.text_col]),
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "communities": torch.tensor(self.community_matrix[idx], dtype=torch.float32),
            "labels": torch.tensor([float(row[self.label_col])], dtype=torch.float32),
        }
