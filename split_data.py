from pyutils import ConfigFile
import numpy as np
import pandas as pd


def split_dataset(config: ConfigFile, train_size: float = 0.8):
    if config["seed"] is not None:
        np.random.seed(config["seed"])
        seed = config["seed"]
    else:
        seed = None

    df = pd.read_csv("data/ruddit_comments_score.csv")

    # Filter deleted
    df = df.loc[df["body"] != "[deleted]"]
    df = df.loc[df["body"] != "[removed]"]

    train = df.sample(frac=train_size, random_state=seed)
    test = df.drop(index=train.index)

    train.to_csv("data/train.csv")
    test.to_csv("data/test.csv")

if __name__ == "__main__":
    cfg = dict(seed=42)
    split_dataset(cfg)
    data = pd.read_csv("data/train.csv", index_col=0)
    train_data = data.sample(frac=0.9, random_state=42)
    valid_data = data.drop(index=train_data.index)
    train_data.to_csv("data/train.csv")
    valid_data.to_csv("data/valid.csv")