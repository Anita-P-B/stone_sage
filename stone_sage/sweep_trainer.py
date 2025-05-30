import argparse
import csv
import json
import os
from copy import deepcopy
from datetime import datetime
import pandas as pd
from stone_sage.arg_parser import get_args
from stone_sage.main import main as run_training
from stone_sage.utils.utils import log_sweep_result
import itertools

def sweep_train(user_config= None):

    # Compose sweep run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    sweep_run_dir = os.path.join("sweeps", f"sweep_{args.sweep_name}_{timestamp}")
    os.makedirs(sweep_run_dir, exist_ok=True)

    # Define sweep options
    model_types = ["mlp", "tiny_deep", "wide_shallow", "tiny_cnn"]
    batch_sizes = [16, 32, 64]
    dropouts = [0.0, 0.1, 0.3]
    learning_rates = [1e-2, 5e-3, 1e-3]

    sweep_configs = [
        {"MODEL": m, "BATCH_SIZE": b, "DROPOUT": d, "LEARNING_RATE": lr}
        for m, b, d, lr in itertools.product(model_types, batch_sizes, dropouts, learning_rates)
    ]

    # Track used flattened configs to detect duplicates
    used_names = set()

    for i, config in enumerate(sweep_configs):
        print(f"\n🌀 Starting sweep {i + 1}/{len(sweep_configs)} with config: {config}")
        flat_config_str = "_".join(f"{k}_{str(v)}" for k, v in sorted(config.items()))

        # Warn if duplicate
        if flat_config_str in used_names:
            print(f"⚠️ Warning: Duplicate config detected — '{flat_config_str}' already used. Skipping.")
            continue
        used_names.add(flat_config_str)

        # Inject sweep-specific paths
        config["SAVE_PATH"] = flat_config_str
        config["RUN_DIR_BASE"] = sweep_run_dir

        # Train
        run_training(sweep_configs = config, user_configs = user_config, sweep = True)

        # Locate run directory
        run_dir = os.path.join(sweep_run_dir, flat_config_str)
        master_log_path = os.path.join(sweep_run_dir, "all_sweep_results.csv")

        # save run logs into sweep master log file
        log_sweep_result(run_dir, config, master_log_path)

    # 📊 Show full sweep log
    if os.path.exists(master_log_path):
        sweep_df = pd.read_csv(master_log_path)
        print("\n📜 Final sweep results:\n")
        print(sweep_df.sort_values(by = "val_rel_mae", ascending = False))

if __name__ == '__main__':

    args = get_args()
    args_dict = vars(args)

    # Remove keys with None values (those not passed via CLI)
    user_configs = {k.upper(): v for k, v in args_dict.items() if v is not None}
    sweep_train(user_config = user_configs )