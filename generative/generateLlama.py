from torchinfo import summary
import tiktoken
import math
from dataclasses import dataclass
import torch
from sympy.printing.tree import print_node
from torch import nn
from torch.nn import functional as F
from pyutils import ConfigFile
import matplotlib.pyplot as plt
from utils.bin import *
import utils
from models.gpt2 import GPT
from models.test import Tokenizer,ModelArgs,Transformer
import json
from pathlib import Path

def generate(model, enc, str2complete, tox: float, device, b_size, max_length, temperature, log: bool = True):
    tokens = enc.encode(str2complete)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(b_size, 1).to(device)
    tox = torch.tensor(tox).unsqueeze(0).repeat(b_size, 1).to(device)
    x = tokens
    # Let's generate
    torch.manual_seed(42)
    while x.size(1) < max_length:
        with torch.inference_mode():
            logits = model(x, tox)[:, -1, :]

            probs = F.softmax(logits / temperature, dim=-1).cpu()

            # We keep only the top 50 most likely tokens, so we do not sample a random token (non-zero probability)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

            ix = torch.multinomial(topk_probs, 1)

            xcol = torch.gather(topk_indices, -1, ix)  # Shape(B, 1)

            x = torch.cat((x, xcol.to(device)), dim=1)

    all_text = []
    for i in range(b_size):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        end_idx = decoded.find("<|endoftext|>")
        if end_idx == -1:
            decoded += "..."
            all_text.append(decoded)
            print(decoded) if log else None
        else:
            all_text.append(decoded[:end_idx])
            print(decoded[:end_idx]) if log else None
        print("="*100) if log else None
    return all_text

def generate_labelintext(model, enc, str2complete, tox: float, device, b_size, max_length, temperature, log: bool = True):
    print("prompt: ")
    print(f"Toxicity : {round(tox*100,0)}; {str2complete}")
    tokens = enc.encode(f"Toxicity : {round(tox*100,0)}; {str2complete}")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(b_size, 1).to(device)
    x = tokens
    # Let's generate
    torch.manual_seed(42)
    while x.size(1) < max_length:
        with torch.inference_mode():
            logits = model(x, tox)[:, -1, :]

            probs = F.softmax(logits / temperature, dim=-1).cpu()

            # We keep only the top 50 most likely tokens, so we do not sample a random token (non-zero probability)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

            ix = torch.multinomial(topk_probs, 1)

            xcol = torch.gather(topk_indices, -1, ix)  # Shape(B, 1)

            x = torch.cat((x, xcol.to(device)), dim=1)

    all_text = []
    for i in range(b_size):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        decoded = "".join(decoded.split(";")[1:])[1:]
        end_idx = decoded.find("<|end_of_text|>")
        if end_idx == -1:
            decoded += "..."
            all_text.append(decoded)
            print(decoded) if log else None
        else:
            all_text.append(decoded[:end_idx])
            print(decoded[:end_idx]) if log else None
        print("=" * 100) if log else None
    return all_text

if __name__ == "__main__":
    device = "cuda"#  utils.get_device()
    b_size = 8
    max_length = 256
    temperature = 1.5




  
    ckpt_dir = "saved_models/Llama3.2-1B"
    tokenizer_path = "saved_models/Llama3.2-1B/tokenizer.model"
 
  

    temperature = 1.0
    top_p= 0.8
    max_seq_len = 256
    max_gen_len = 256
    max_batch_size = 8
    flash = False
    with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            flash=flash,
            **params,
        )
    

    enc = Tokenizer(model_path=tokenizer_path)


   # enc = tiktoken.get_encoding("gpt2")

    #model = GPT(config, labelintext=True)
    #checkpoint = torch.load("saved_models/Llama3.2-1B/6/Llama.pth", map_location="cpu")["model_state_dict"]
    checkpoint = torch.load("saved_models/Llama3.2-1B/consolidated.00.pth", map_location="cpu")# ["model_state_dict"]

    model = Transformer(model_args)
    

    model.load_state_dict(checkpoint, strict=True)
    #model.bfloat16()

    device = torch.device("cuda")
    model.to(device=device)


    str2complete = ""
    generate_labelintext(model, enc, str2complete, 2., device, b_size, max_length, temperature, log=True)
    # generate_labelintext(model, str2complete, -0.99, device)