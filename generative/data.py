import os.path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pyutils import ConfigFile
import numpy as np
from typing import *
import tiktoken

class TextDataset(Dataset):
    def __init__(self, data, labelintext):
        self.data = data
        self.labelintext = labelintext

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        comment = row["body"]
        label = row["score"]
        if self.labelintext:
            comment = f"{100*round(label, 2)}: {comment}"
        return comment, torch.tensor(label)

class TextCollator:
    def __init__(self, max_len: int = 256, labelintext: bool = False):
        self.max_len = max_len
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.labelintext = labelintext

    def prep_tensors(self, B, L):
        if self.labelintext:
            tokens = torch.zeros((B, L + 1), dtype=torch.int64) # We add 1 because of the <bos> token, aka <cls>
        else:
            tokens = torch.zeros((B, L), dtype=torch.int64) # We already have a custom token
        tokens.fill_(-1)
        targets = torch.zeros((B, L + 1), dtype=torch.int64)
        targets.fill_(-1)

        return tokens, targets

    def __call__(self, raw_batch: Sequence[Tuple[str, torch.Tensor]]) -> \
            Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Called by the dataloader to collate a batch of data.
        :param raw_batch: The raw batch of data
        :return: A list of the raw sequences, the tokenized ground truth, the tokenized input and the label
        """
        B = len(raw_batch)
        texts, labels = zip(*raw_batch)

        # Tokenize
        toks = [torch.tensor(self.tokenizer.encode(text)[:self.max_len - 1]) for text in texts]
        L = max([len(tok) for tok in toks])

        # Prepare the tensors
        tokens, targets = self.prep_tensors(B, L)
        scores = torch.tensor(labels).unsqueeze(1)

        # Fill the tensors
        if self.labelintext:
            for i, tok in enumerate(toks):
                tokens[i, 1:len(tok) + 1] = tok
                tokens[i, 0] = torch.tensor(50256,
                                                    dtype=torch.int64) # End of text token is also start of text for GPT2
                targets[i, :len(tok)] = tok
                targets[i, len(tok)] = torch.tensor(50256,
                                                    dtype=torch.int64)  # End of text token : print(enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

        else:
            for i, tok in enumerate(toks):
                tokens[i, :len(tok)] = tok
                targets[i, :len(tok)] = tok
                targets[i, len(tok)] = torch.tensor(50256, dtype=torch.int64) # End of text token : print(enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

        return texts, tokens, scores, targets

def make_dataloaders(config: ConfigFile):
    train_data = pd.read_csv("../data/train.csv", index_col=0)
    valid_data = pd.read_csv("../data/valid.csv", index_col=0)
    test_data = pd.read_csv("../data/test.csv", index_col=0)

    train = TextDataset(train_data, config["data"]["labelintext"])
    valid = TextDataset(valid_data, config["data"]["labelintext"])
    test = TextDataset(test_data, config["data"]["labelintext"])

    collator = TextCollator(max_len=config["data"]["context_len"], labelintext=config["data"]["labelintext"])
    num_workers = config["data"]["num_workers"]
    train_dl = DataLoader(train, batch_size=config["data"]["batch_size"], collate_fn=collator,
                          num_workers=num_workers, persistent_workers=num_workers > 0,
                          shuffle=config["data"]["shuffle"])
    val_dl = DataLoader(valid, batch_size=config["data"]["batch_size"], collate_fn=collator,
                          num_workers=num_workers, persistent_workers=num_workers > 0,
                          shuffle=False)
    test_dl = DataLoader(test, batch_size=config["data"]["batch_size"], collate_fn=collator,
                          num_workers=num_workers, persistent_workers=num_workers > 0,
                          shuffle=False)

    return train_dl, val_dl, test_dl

if __name__ == "__main__":
    import pandas as pd
    split_dataset({"seed": 42}, 0.8)

    ds = TextDataset({"seed": 42}, "data/train.csv")
    # lengths = pd.read_csv("data/train.csv", index_col=0)["body"].str.len()
    # plt.hist(lengths.values)
    # plt.show()
    cfg = {
        "seed": 42,
        "data":{
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 2
        }
    }
    train_dl, val_dl, test_dl = make_dataloaders(config=cfg)
    for text, tokens, target in train_dl:
        print(text)
        print(tokens)
        print(target)
        break
    # df = pd.read_csv("data/ruddit_comments_score.csv")
    #
    # # Filter deleted
    # df = df.loc[df["body"] != "[deleted]"]
    # for i, row in df.sample(frac=0.05).iterrows():
    #     print(row["score"])
    #     print(row["body"])
    #     print("-" * 100)
