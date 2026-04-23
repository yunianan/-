import os
import sys
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from aim import Run
from model import Model


batch_size = 6  # How many batches per training step
context_length = 128  # Length of the token chunk each batch
max_iters = 20000  # Total of training iterations <- Change this to smaller number for testing
learning_rate = 1e-3  # 0.001
eval_interval = 50  # How often to evaluate
eval_iters = 20  # Number of iterations to average for evaluation
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if it's available.
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)


# 准备训练数据
with open('data/scif.txt', 'r', encoding="utf-8") as file:
    text = file.read()

vocab = sorted(list(set(text)))
vocab_size = max_token_value = len(vocab)

char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for char, idx in char2idx.items()}

encode = lambda x: [char2idx[char] for char in x]
decode = lambda idx_list: ''.join([idx2char[idx] for idx in idx_list])

tokenized_text = torch.tensor(encode(text), dtype=torch.long)

# Split train and validation
train_size = int(len(tokenized_text) * 0.8)
train_data = tokenized_text[:train_size]
val_data = tokenized_text[train_size:]

# Initialize the model
model = Model(max_token_value=vocab_size).to(device)

# get batch
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=(len(data) - context_length), size=(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx+1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y