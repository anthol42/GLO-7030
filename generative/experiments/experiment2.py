import torch
from torch.utils.tensorboard import SummaryWriter
import os
from data import make_dataloaders
from models import GPT
from training.train import train, evaluate
import sys
import shutil
from utils import Table, State, get_profile
import utils
from pyutils import ConfigFile
from utils.bin import *
from torchmetrics import Accuracy
from torchinfo import summary
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

# To verify if the config has the good format
from configs.formats import config_format

metrics = {
}


def experiment2(args, kwargs):
    config_loggers_with_verbose(args.verbose)
    # Setup
    device = utils.get_device(args.cpu)
    log(f"Running on {device}")
    hyper = utils.clean_dict(vars(args).copy())

    DEBUG = args.debug

    # Loading the config file
    # Note: There are no profiles in the template config file
    config = ConfigFile(args.config, config_format, verify_path=True, profiles=["default"])
    config.change_profile(get_profile())
    config.override_config(kwargs)

    # Preparing Result Table
    rtable = Table("results/resultTable.json")
    sys.excepthook = rtable.handle_exception(sys.excepthook)
    if DEBUG:
        run_id = "DEBUG"
        log(f"Running in {Colors.warning}DEBUG{ResetColor()} mode!")
    else:
        resultSocket = rtable.registerRecord(__name__, args.config, category=None, **hyper)
        run_id = resultSocket.get_run_id()

    config["model"]["model_dir"] = f'{config["model"]["model_dir"]}/{run_id}'

    comment = '' if hyper.get("comment") is None else hyper.get("comment")
    if os.path.exists(f'runs/{run_id}'):
        log(f"Clearing tensorboard logs for id: {run_id}")
        shutil.rmtree(f'runs/{run_id}')
    State.writer = SummaryWriter(log_dir=f'runs/{run_id}', comment=comment)

    torch.random.manual_seed(config["seed"])
    # Loading the data
    train_loader, val_loader, test_loader = make_dataloaders(config=config)
    log("Data loaded successfully!")

    # Loading the model
    model = GPT.from_pretrained("gpt2", labelintext=config["data"]["labelintext"])
    model.freeze(
        transformer=config["freezer"]["transformer"],
        embeddings = config["freezer"]["embeddings"],
        pos = config["freezer"]["pos"],
        lm_head = config["freezer"]["lm_head"],
        ln_f = config["freezer"]["ln_f"]
    )
    if args.verbose >= 3:
        summary(model, input_data=(torch.randint(0, 128, size=(config["data"]["batch_size"], 256)), torch.randn(config["data"]["batch_size"], 1)))
    model.to(device)
    log("Model loaded successfully!")

    # Loading optimizer, loss and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, config["training"]["num_epochs"] * len(train_loader), eta_min=config["training"]["min_lr"])

    # Training
    # Prepare the path of input sampling if flag is set
    if args.sample_inputs:
        sample_inputs = f"{config['model']['model_dir']}/inputs.pth"
    else:
        sample_inputs = None
    log("Begining training...")
    log(f"Watching: {args.watch}")
    try:
        log("Phase 1")
        train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config["training"]["num_epochs"],
            device=device,
            scheduler=scheduler,
            config=config,
            metrics=metrics,
            watch=args.watch,
            sample_inputs=sample_inputs,
            verbose=args.verbose
        )
        # Reactivate all parameters, and train for 1 epoch
        log("Phase 2")
        weights = torch.load(f'{config["model"]["model_dir"]}/{config["model"]["name"]}.pth', weights_only=False)[
            "model_state_dict"]
        model.load_state_dict(weights)
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.1 * config["training"]["lr"],
            weight_decay=0.)
        train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            device=device,
            scheduler=None,
            config=config,
            metrics=metrics,
            watch=args.watch,
            sample_inputs=sample_inputs,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        log("Keyboard Interrupt detected. Ending training...", start="\n", end="\n\n")

    log("Training done!")

    # Load best model
    log("Loading best model")
    weights = torch.load(f'{config["model"]["model_dir"]}/{config["model"]["name"]}.pth', weights_only=False)[
        "model_state_dict"]
    model.load_state_dict(weights)
    # Test
    results = evaluate(model, test_loader, device, metrics=metrics)
    log("Training done!  Saving...")

    save_dict = {
        "epoch": config["training"]["num_epochs"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_results": results
    }
    torch.save(
        save_dict, f"{config['model']['model_dir']}/final_{config['model']['name']}.pth")
    # Copy config file to model dir
    shutil.copy(args.config, config['model']["model_dir"])

    # Print stats of code
    if config.have_warnings():
        warn(config.get_warnings())

    State.writer.flush()
    State.writer.close()

    # Save results
    if not DEBUG:
        resultSocket.write(loss=results["loss"])
        rtable.toTxt()




