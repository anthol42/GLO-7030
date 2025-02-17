import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def make_dataset(config):
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    return training_data, test_data

def make_dataloader(config):
    train_ds, test_ds = make_dataset(config)
    train_dataloader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=config["data"]["shuffle"])
    test_dataloader = DataLoader(test_ds,
                                 batch_size=config["data"]["batch_size"], shuffle=False)
    return train_dataloader, test_dataloader, test_dataloader

if __name__ == "__main__":
    import pandas