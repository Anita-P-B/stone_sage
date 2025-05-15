import os
from datetime import datetime

import torch

from stone_sage.arg_parser import get_args
from stone_sage.configs.config import Config
from stone_sage.datasets.dataset_utils import load_or_download_data, split_and_save_partitions
from stone_sage.utils.utils import update_configs_with_dict
from stone_sage.datasets.partition_analysis import analyze_partitioned_data


def main(sweep_config=None, user_configs=None):
    # set configs
    static_configs = Config()
    # Merge order: static < sweep < user (CLI wins)
    configs = update_configs_with_dict(static_configs, sweep_config or {})
    configs = update_configs_with_dict(configs, user_configs or {})

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    if device.type == 'cuda':
        print(f"🧠 GPU name: {torch.cuda.get_device_name(device)}")
        print(f"🧮 Memory available: {torch.cuda.get_device_properties(device).total_memory // (1024 ** 2)} MB")

    # get data
    df = load_or_download_data(
        path=configs.DATA_PATH,
        force_download=configs.FORCE_DOWNLOAD,
        expected_checksum=configs.CHECKSUM,
        debug=configs.DEBUG
    )
    if configs.DEBUG:
        print(df.head())
    # create run dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(configs.RUN_DIR_BASE, f"{configs.SAVE_PATH}_run_{timestamp}")

    # split dataset to partitions
    _, train_df, val_df, test_df = (
        split_and_save_partitions(df=df, run_dir=run_dir,
                                  val_ratio=configs.VAL_RATIO,
                                  test_ratio=configs.TEST_RATIO,
                                  random_state=configs.SPLIT_SEED))
    # save data statistics
    analyze_partitioned_data(
        {"train": train_df, "val": val_df, "test": test_df},
        user_config,
        run_data_path=os.path.join(run_dir, "dataset_with_partitions.csv"),
        plot_statistics= configs.PLOT_STATISTICS
    )






if __name__ == '__main__':
    args = get_args()
    args_dict = vars(args)

    # Remove keys with None values (those not passed via CLI)
    user_config = {k.upper(): v for k, v in args_dict.items() if v is not None}
    main(user_configs=user_config)
