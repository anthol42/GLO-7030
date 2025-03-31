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
from models.llama import Transformer, ModelArgs,Tokenizer
from configs.formats import config_format
from time import time


def generate(model,enc, prompts:list[str], device, max_gen_len=256):
    print(model.text_completion(enc,prompts,sample_rng,max_gen_len=max_gen_len,device=device))

if __name__ == "__main__":
    num_return_sequences = 5
    max_length = 256
    config = dict(n_layer=12, n_head=12, n_embd=768, vocab_size=50257, block_size=1024, bias=True)
    device = torch.device("cuda")
    LLAMA32_CONFIG = {
    "vocab_size": 128_256,      # Vocabulary size
    "context_length": 8192,     # Context length
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
    model_args.multiple_of = 256  # Vous pouvez ajuster cette valeur si nécessaire
    model_args.ffn_dim_multiplier = None  # Vous pouvez ajuster cette valeur si nécessaire
    model_args.norm_eps = 1e-5  # Vous pouvez ajuster cette valeur si nécessaire
    model_args.rope_theta = LLAMA32_CONFIG["rope_base"]
    model_args.use_scaled_rope = True  # Si vous utilisez RoPE avec scaling
    model_args.max_batch_size = 32  # Vous pouvez ajuster cette valeur si nécessaire
    model_args.max_seq_len = LLAMA32_CONFIG["context_length"]
    model_args.flash = False  # Vous pouvez ajuster cette valeur si nécessaire
    
    config = "configs/configLlama.yml"
    config = ConfigFile(config, config_format, verify_path=True, profiles=["default"])
    print(config)

    ckpt_path = "saved_models/Llama3.2-1B/1/Llama.pth"
    ckpt_path = "saved_models/Llama3.2-1B/1consolidated.00.pth"
    tokenizer_path = "saved_models/Llama3.2-1B/1tokenizer.model"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)

    print('Model Loaded...')
    enc = Tokenizer(model_path=tokenizer_path)
    

    model.to(device)
    sample_rng = torch.Generator(device='cuda')
    sample_rng.manual_seed(1337)
    
   
    


    #model.generate(tokenizer,[token],sample_rng,256,tox=tox )
    prompts = ["Hi, ", "hello,"]
    t1 = time()
    print(model.text_completion(enc,device,prompts,sample_rng,max_gen_len=128))
    print(f"TOTAL GEN TIME for {len(prompts)} generations : ", time()-t1 )

    
