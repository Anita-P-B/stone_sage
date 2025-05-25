import os
from datetime import datetime
from torchinfo import summary
import torch
from torch.utils.data import DataLoader

from stone_sage.arg_parser import get_args
from stone_sage.configs.config import Config
from stone_sage.datasets.dataset_utils import (load_or_download_data, split_and_save_partitions,
                                               get_train_mean_and_std)
from stone_sage.datasets.partition_analysis import analyze_partitioned_data
from stone_sage.datasets.stone_dataset import StoneDataset
from stone_sage.models.stone_regressor import StoneRegressor
from stone_sage.train import Trainer
from stone_sage.utils.utils import (update_configs_with_dict, get_loss_func, get_optimizer,
                                    save_run_state, overfit_mode)


def main(sweep_configs=None, user_configs=None, sweep=False):
    # set configs
    static_configs = Config()
    # Merge order: static < sweep < user (CLI wins)
    configs = update_configs_with_dict(static_configs, sweep_configs or {})
    configs = update_configs_with_dict(configs, user_configs or {})

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    if device.type == 'cuda':
        print(f"🧠 GPU name: {torch.cuda.get_device_name(device)}")
        print(f"🧮 Memory available: {torch.cuda.get_device_properties(device).total_memory // (1024 ** 2)} MB")

    # set random seed for reproducability
    g = torch.Generator()
    g.manual_seed(configs.SPLIT_SEED)

    # get data
    df = load_or_download_data(
        path=configs.DATA_PATH,
        force_download=configs.FORCE_DOWNLOAD,
        expected_checksum=configs.CHECKSUM,
        debug=configs.DEBUG
    )
    if configs.DEBUG:
        print(df.head())

    if sweep:
        run_dir = os.path.join(configs.RUN_DIR_BASE, f"{configs.SAVE_PATH}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(configs.RUN_DIR_BASE, f"{configs.SAVE_PATH}_run_{timestamp}")

    # split dataset to partitions
    _, train_df, val_df, test_df = split_and_save_partitions(df=df, run_dir=run_dir,
                                                             val_ratio=configs.VAL_RATIO,
                                                             test_ratio=configs.TEST_RATIO,
                                                             random_state=configs.SPLIT_SEED)
    # save data statistics
    analyze_partitioned_data(
        {"train": train_df, "val": val_df, "test": test_df},
        user_configs,
        run_data_path=os.path.join(run_dir, "dataset_with_partitions.csv"),
        plot_statistics=configs.PLOT_STATISTICS
    )

    mean, std = get_train_mean_and_std(train_df, configs.TARGET_COLUMN)

    train_set = StoneDataset(train_df, target_column=configs.TARGET_COLUMN, mean=mean, std=std)
    train_loader = DataLoader(train_set, batch_size=configs.BATCH_SIZE,
                              shuffle=configs.SHUFFLE, generator=g)
    val_set = StoneDataset(val_df, target_column=configs.TARGET_COLUMN, mean=mean, std=std)
    val_loader = DataLoader(val_set, batch_size=configs.BATCH_SIZE,
                            shuffle=configs.SHUFFLE, generator=g)

    input_dim = train_set.input_dim
    # get model loss and optimizer
    model = StoneRegressor.build(configs, input_dim)
    if configs.DEBUG:
        summary(model, input_size=(configs.BATCH_SIZE, input_dim))
    optimizer = get_optimizer(configs.OPTIMIZER, configs.LEARNING_RATE, model)
    loss = get_loss_func(configs.LOSS)

    # save run configurations
    save_run_state(configs=configs, run_dir=run_dir)
    # Train


    if configs.OVERFIT_TEST:
        print("🔬 Running overfit test on a tiny subset...")

        tiny_loader = overfit_mode(train_set, g)
        # Train only on tiny_loader
        trainer = Trainer(model, optimizer, loss, tiny_loader, tiny_loader, run_dir,
                          n_best_checkpoints=configs.N_BEST_CHECKPOINTS)
    else:
        trainer = Trainer(model, optimizer, loss, train_loader, val_loader, run_dir,
                          n_best_checkpoints=configs.N_BEST_CHECKPOINTS)


    trainer.train(num_epochs=configs.EPOCHS)


if __name__ == '__main__':
    args = get_args()
    args_dict = vars(args)

    # Remove keys with None values (those not passed via CLI)
    user_config = {k.upper(): v for k, v in args_dict.items() if v is not None}
    main(user_configs=user_config)
