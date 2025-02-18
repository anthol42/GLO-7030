config_format = {
    "data":{
        "batch_size": int,
        "shuffle": bool,
        "num_workers": int,
        "context_len": int
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
    "freezer": {
        "transformer": float,
        "embeddings": bool,
        "pos": bool,
        "lm_head": bool,
        "ln_f": bool
    },
    "seed": int
}