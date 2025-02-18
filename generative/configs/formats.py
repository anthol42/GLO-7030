config_format = {
    "data":{
        "batch_size": int,
        "shuffle": bool,
        "num_workers": int
    },
    "training":{
        "num_epochs": int,
        "lr": float,
        "min_lr": float,
        "weight_decay": float,
    },
    "model":{
        "model_dir": "opath",
        "name": "opath"
    },
    "seed": int
}