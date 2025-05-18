from torch import nn, optim
import os
import json
import subprocess
import torch
import csv
from sklearn.metrics import mean_absolute_error
import numpy as np

def update_configs_with_dict(config_obj, override_dict, debug = False):
    for key, value in override_dict.items():
        if debug:
            print(f"🔍 Overriding: {key} = {value} | Present? {hasattr(config_obj, key)}")
        if hasattr(config_obj, key):
            setattr(config_obj, key, value)
    return config_obj

def get_optimizer(optimizer_name, learning_rate, model):
    optimizers = {
        "adam": lambda: optim.Adam(model.parameters(), lr=learning_rate),
        "sgd": lambda: optim.SGD(model.parameters(), lr=learning_rate),
        "adamw": lambda: optim.AdamW(model.parameters(), lr=learning_rate),
    }

    try:
        return optimizers[optimizer_name.lower()]()
    except KeyError:
        raise ValueError(f"❌ Optimizer '{optimizer_name}' not recognized. Available: {list(optimizers.keys())}")

def get_loss_func(loss_name):
    losses = {
        "mae": nn.L1Loss,
        "mse": nn.MSELoss,
        "huber": nn.HuberLoss,
    }

    try:
        return losses[loss_name.lower()]()
    except KeyError:
        raise ValueError(f"❌ Loss function '{loss_name}' not recognized. Available: {list(losses.keys())}")

def save_run_state(configs,  run_dir):
    """Save training configuration as a JSON file, serializing any non-JSON-safe values as strings."""
    # Create timestamped folder
    os.makedirs(run_dir, exist_ok=True)

    configs_dict = {}
    for k, v in configs.__dict__.items():
        if k.startswith("__"):
            continue
        try:
            json.dumps(v)  # test if it's serializable
            configs_dict[k] = v
        except (TypeError, OverflowError):
            configs_dict[k] = str(v)  # fallback: save as string

    config_path = os.path.join(run_dir, "train_config.json")
    # Save CONSTS
    with open(config_path, "w") as f:
        json.dump(configs_dict, f, indent=4)

    try:
        subprocess.run(['attrib', '+R', config_path], check=True)
        print(f"🗃️  Config saved to: {config_path} (🔒 read-only)")
    except subprocess.CalledProcessError:
        print(f"⚠️ Failed to mark config file as read-only. File still saved at: {config_path}")
    print(f"🗃️  Run saved to: {run_dir}")

def save_checkpoint(run_dir, model, optimizer, scheduler, epoch, train_loss, val_loss, extra_info=None):
    """
    Save model checkpoint with flexible metadata.

    Parameters:
        run_dir (str): Directory to save the checkpoint in.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer used.
        scheduler (optional): The learning rate scheduler used.
        epoch (int): Current epoch number.
        train_loss (float): Training loss.
        val_loss (float): Validation loss.
        extra_info (dict, optional): Additional items to include in the checkpoint.
    """
    filename = f"train_loss_{train_loss:.2f}_val_loss_{val_loss:.2f}.pt"
    full_path = os.path.join(run_dir, filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }

    if extra_info:
        checkpoint.update(extra_info)

    torch.save(checkpoint, full_path)
    print(f"🧪 Best model saved: {full_path}")

def save_evaluation_summary(run_dir, metrics_dict):
    os.makedirs(run_dir, exist_ok=True)
    file_path = os.path.join(run_dir, "evaluation_results.csv")

    with open(file_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
        writer.writeheader()
        writer.writerow(metrics_dict)

    print(f"📊 Evaluation results saved to: {file_path}")


def relative_mae_percentage(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mean_target = np.mean(y_true)
    if mean_target == 0:
        return np.inf  # Prevent division by zero
    return (mae / mean_target) * 100
