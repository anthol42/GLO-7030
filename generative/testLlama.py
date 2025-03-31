import os
import glob


import time
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary
from dataLlama import make_dataloaders
from configs.formats import config_format
from pyutils import ConfigFile
from models.llama import Transformer, Tokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from training.train import train


import tiktoken
from tiktoken.load import load_tiktoken_bpe
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False
    max_batch_size: int = 32
    max_seq_len: int = 2048
    flash: bool = False # use flash attention?

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0




def main():
    device = torch.device("cuda")
    LLAMA32_CONFIG = {
    "vocab_size": 128_256,      # Vocabulary size
    "context_length": 128,     # Context length
    "emb_dim": 2048,            # Embedding dimension
    "n_heads": 32,              # Number of attention heads
    "n_layers": 16,             # Number of layers
    "hidden_dim": 8192,         # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,           # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,     # The base in RoPE's "theta"
    "dtype": torch.bfloat16,    # Lower-precision dtype to reduce memory usage
    "rope_freq": {              # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}
    model_args = ModelArgs()
    model_args.dim = LLAMA32_CONFIG["emb_dim"]
    model_args.n_layers = LLAMA32_CONFIG["n_layers"]
    model_args.n_heads = LLAMA32_CONFIG["n_heads"]
    model_args.n_kv_heads = LLAMA32_CONFIG["n_kv_groups"]
    model_args.vocab_size = LLAMA32_CONFIG["vocab_size"]
    model_args.multiple_of = 128  # Vous pouvez ajuster cette valeur si nécessaire
    model_args.ffn_dim_multiplier = None  # Vous pouvez ajuster cette valeur si nécessaire
    model_args.norm_eps = 1e-5  # Vous pouvez ajuster cette valeur si nécessaire
    model_args.rope_theta = LLAMA32_CONFIG["rope_base"]
    model_args.use_scaled_rope = True  # Si vous utilisez RoPE avec scaling
    model_args.max_batch_size = 8  # Vous pouvez ajuster cette valeur si nécessaire
    model_args.max_seq_len = LLAMA32_CONFIG["context_length"]
    model_args.flash = False  # Vous pouvez ajuster cette valeur si nécessaire
    
    config = "configs/configLlama.yml"
    config = ConfigFile(config, config_format, verify_path=True, profiles=["default"])
    

    ckpt_path = "saved_models/Llama3.2-1B/1consolidated.00.pth"
    tokenizer_path = "saved_models/Llama3.2-1B/1tokenizer.model"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = Transformer(model_args,labelintext=False)
    print(model_args.__dict__)
    
    model.load_state_dict(checkpoint, strict=False)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    train_loader, val_loader, test_loader = make_dataloaders(config=config)
    model.freeze(transformer=1,embeddings=True, ln_f=True)
    summary(model)
    model.cuda()

    def conf_optimizer(model,config):
        params = []
        for name, param in model.named_parameters():
            if "tok_embeddings" in name:
                params.append(param)
            if "tox" in name :
                params.append(param)
        optimizer = torch.optim.AdamW(
        params=params, lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"])
        #scheduler = CosineAnnealingLR(optimizer, config["training"]["num_epochs"] * len(train_loader), eta_min=config["training"]["min_lr"])
        return optimizer#, scheduler



    optimizer = conf_optimizer(model,config)

    train(model,optimizer,train_loader,val_loader,10,device,config)
    
    exit()


    optimizer = model.configure_optimizers(learning_rate=1e-5, weight_decay=0.0)
    model.cpu()
    for step in range(20):
        
        optimizer.zero_grad()
        for text, X, scores, y in train_loader:
            print("X : ", X[0])
            print("y : ", y[0])
            x, y = X.cpu(), y.cpu()

            loss = model.forward_loss(x, y)
            loss.backward()
            optimizer.step()
            print(f"step {step}, loss: {loss.item()}")


    token = tokenizer.encode("",eos=False,bos=True)
    model.to(device)
    sample_rng = torch.Generator(device='cuda')
    sample_rng.manual_seed(1337)
    tox = torch.tensor([[1.]]).to(device)
    model.freeze(ln_f=1., embeddings=True)

    summary(model)
    exit()


    #model.generate(tokenizer,[token],sample_rng,256,tox=tox )
    prompts = ["here is a list of farm animals : "]

    print(model.text_completion(tokenizer,prompts,sample_rng,max_gen_len=256,tox=tox))

    



if __name__ == '__main__' : 
    main()