from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from models.decoder import BERT, MultiHeadTransformer
from utils.param_counter import count_parameters

text = "AABBCC" * 10
tokeniser = {"<s>": 0, "A": 1, "B": 2, "C": 3}
id_to_token = {idx: key for key, idx in tokeniser.items()}
tokens = [tokeniser[char] for char in list(text)]
inpt = torch.LongTensor([0] + tokens[:-1])
target = torch.LongTensor(tokens)

CONTEXT_WINDOW = len(text)
EMBEDDING_DIM = 20
FF_DIM = 4 * EMBEDDING_DIM

VOCAB_SIZE = len(tokeniser)

device = "cuda" if torch.cuda.is_available() else "cpu"

transformer_layer = MultiHeadTransformer(EMBEDDING_DIM, 1, FF_DIM)
model = BERT(transformer_layer, VOCAB_SIZE, EMBEDDING_DIM, CONTEXT_WINDOW + 1)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for i in range(10000):
    optimizer.zero_grad()

    outputs = model(inpt.unsqueeze(0))

    loss = criterion(outputs.squeeze(), target)

    loss.backward()

    optimizer.step()

    if (i + 1) % 1000 == 0:
        print("", end="\r")
        print(f"loss: {loss.item()} i: {i+1}", end="\r")
print("")
seq = torch.LongTensor([0])

print("inference mode...")
with torch.inference_mode():
    for i in range(CONTEXT_WINDOW):
        out = model(seq.unsqueeze(0)).squeeze()
        next_token = torch.argmax(out[-1]) if i > 0 else torch.argmax(out)

        seq = torch.concat([seq, next_token.unsqueeze(dim=-1)])

        print(id_to_token[int(next_token)], end="")
