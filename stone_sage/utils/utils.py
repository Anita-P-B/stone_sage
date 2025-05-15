from torch import nn, optim
import os
import json
import subprocess

def update_configs_with_dict(config_obj, override_dict):
    for key, value in override_dict.items():
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
