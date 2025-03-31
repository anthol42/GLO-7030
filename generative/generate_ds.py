import numpy as np
import pandas as pd
from torchinfo import summary
import tiktoken
import math
from dataclasses import dataclass
import torch
from sympy.printing.tree import print_node
from torch import nn
from torch.nn import functional as F
from pyutils import progress
from utils.bin import *
from models.gpt2 import GPT
from generate import generate, generate_labelintext
from typing import *

def norm_label(min_val, max_mal):
    def normalize(label):
        return  2 * (label - min_val) / (max_mal - min_val) - 1
    return normalize

# , log: bool = True
def generate_dataset(model: nn.Module, enc, device, max_length=256, temperature=1.5,
                     tox_range: Tuple[[float, float]] = (-2., 1.), n_per_tox: int = 100, num: int = 500):
    toxicities = np.linspace(tox_range[0], tox_range[1], num, dtype=np.float32)
    log(f"Generating a dataset of {len(toxicities) * n_per_tox} samples")
    df = pd.DataFrame(columns=["body","score"])
    norm = norm_label(tox_range[0], tox_range[1])
    for tox in progress(toxicities, desc="Generating dataset"):
        data = generate_labelintext(model, enc, "", tox, device, b_size=n_per_tox, max_length=max_length,
                        temperature=temperature, log=False)
        labels = np.full_like(data, norm(tox))
        sub_df = pd.DataFrame({"body": data, "score": labels})
        df = pd.concat([df, sub_df], ignore_index=True)
    return df

if __name__ == "__main__":
    device = "cuda"#  utils.get_device()
    num_return_sequences = 5
    max_length = 256
    config = dict(n_layer=12, n_head=12, n_embd=768, vocab_size=50257, block_size=1024, bias=True)

    enc = tiktoken.get_encoding("gpt2")

    model = GPT(config, labelintext=True)
    state_dict = torch.load("saved_models/classifier/DEBUG/GPT2.pth", map_location="cpu")["model_state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    dataset = generate_dataset(model, enc, device)
    dataset.to_csv("data/gpt2_unif.csv", index=False)
