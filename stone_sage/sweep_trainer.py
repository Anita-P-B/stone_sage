import argparse
import csv
import json
import os
from copy import deepcopy
from datetime import datetime
import pandas as pd
from stone_sage.arg_parser import get_args
from stone_sage.main import main as run_training

def sweep_train(user_config= None):

    # Compose sweep run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    sweep_run_dir = os.path.join("sweeps", f"sweep_{args.sweep_name}_{timestamp}")
    os.makedirs(sweep_run_dir, exist_ok=True)


    # Define sweep options
    sweep_configs = [
        {"AUGMENTATION_PROB": 0.6, "BATCH_SIZE": 32},
        {"AUGMENTATION_PROB": 0.7}
    ]

    # Track used flattened configs to detect duplicates
    used_names = set()
    sweep_log_df = []


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
        metrics_path = os.path.join(run_dir, "evaluation_results.csv")

        # save run logs into sweep master log file
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            if not metrics_df.empty:
                final_row = metrics_df.iloc[-1]  # take the last row (final epoch)
            else:
                print(f"⚠️ Empty metrics file for {flat_config_str}.")
                final_row = pd.Series()
        else:
            print(f"⚠️ Missing metrics file for {flat_config_str}.")
            final_row = pd.Series()

        # Merge metrics and config info
        full_row = pd.Series(config)  # insert sweep config columns
        full_row["run_dir"] = run_dir

        # Merge with metrics
        combined_row = pd.concat([full_row, final_row])
        sweep_log_df.append(combined_row)

    # Save full sweep log
    if sweep_log_df:
        sweep_df = pd.DataFrame(sweep_log_df)
        sweep_summary_path = os.path.join(sweep_run_dir, "all_sweep_results.csv")
        sweep_df.to_csv(sweep_summary_path, index=False)
        print(f"\n📜 All sweep results saved to: {sweep_summary_path}")
    else:
        print("\n⚠️ No successful sweeps to log.")

if __name__ == '__main__':

    args = get_args()
    args_dict = vars(args)

    # Remove keys with None values (those not passed via CLI)
    user_configs = {k.upper(): v for k, v in args_dict.items() if v is not None}
    sweep_train(user_config = user_configs )