from stone_sage.arg_parser import get_args
import torch
import os
import pandas as pd
from stone_sage.models.stone_regressor import StoneRegressor  # adapt to your actual path
from stone_sage.datasets.stone_dataset import StoneDataset
from torch.utils.data import DataLoader
import json
from stone_sage.utils.utils import update_configs_with_dict, save_evaluation_summary, denormalize
from stone_sage.utils.dict_to_class import DotDict
from sklearn.metrics import mean_absolute_error
from stone_sage.datasets.dataset_utils import get_train_mean_and_std
import numpy as np
import matplotlib.pyplot as plt

class Predictor:
    def __init__(self, user_configs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.configs = {}
        self.debug = user_config.get("DEBUG", False)
        self._load_configs(user_configs)

    def _load_configs(self, user_configs):
        checkpoint_path = user_configs.get("CHECKPOINT_PATH")
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError("You must provide a valid --checkpoint_path")

        # 🧩 Load train_config.json from same folder
        self.run_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(self.run_dir, "train_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config file: {config_path}")

        with open(config_path, "r") as f:
            configs = json.load(f)
        configs_obj = DotDict(configs)

        self.configs = update_configs_with_dict(configs_obj, user_configs, self.debug)

    def load_model(self, input_dim):
        model = StoneRegressor.build(self.configs, input_dim)
        checkpoint = torch.load(self.configs.CHECKPOINT_PATH, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def calculate_evaluation_results(self, df, target_mean, target_std):
        partitions = ["train", "val", "test"]
        metrics = {}
        for partition in partitions:
            part_df = df[df["partition"] == partition]
            true_targets = part_df["true_values"].values
            predictions = part_df["prediction"].values
            if self.debug:

                target_mean = true_targets.mean()
                target_std = true_targets.std()

                normalized = (true_targets - target_mean) / target_std
                reconstructed = normalized * target_std + target_mean
                # usefull one line sanity check
                # assert np.allclose(reconstructed, true_targets, atol=1e-5)
                # Compare original and reconstructed
                absolute_diff = np.abs(reconstructed - true_targets)
                relative_diff = absolute_diff / np.maximum(np.abs(true_targets), 1e-8)
                print(f"Partition: {partition}")
                print(f"Target mean: {float(target_mean):.6f}, Target std: {float(target_std):.6f}")
                print(f"Max absolute error: {float(absolute_diff.max()):.6f}")
                print(f"Mean absolute error: {float(absolute_diff.mean()):.6f}")
                print(f"Max relative error: {100 * float(relative_diff.max()):.4f}%")
                print(f"Mean relative error: {100 * float(relative_diff.mean()):.4f}%")
                print("-" * 50)

            mae = mean_absolute_error(true_targets, predictions)
            rel_mae = self.relative_mae_percentage(true_targets, predictions)
            smape = self.smape(true_targets, predictions)
            metrics[f"{partition}_mae"] = mae
            metrics[f"{partition}_rel_mae"] = rel_mae
            metrics[f"{partition}_smape"] = smape


        save_evaluation_summary(self.run_dir, metrics)

    def predict(self):
        # Load data
        df_path = os.path.join(self.run_dir, "dataset_with_partitions.csv")
        df = pd.read_csv(df_path)

        train_df = df[df["partition"] == "train"]
        train_dataset = StoneDataset(train_df, target_column=self.configs.TARGET_COLUMN)
        target_mean, target_std =  train_dataset.mean_targets, train_dataset.std_targets

        dataset = StoneDataset(df, target_column=self.configs.TARGET_COLUMN)
        loader = DataLoader(dataset, batch_size=self.configs.BATCH_SIZE, shuffle = False)

        # Load model
        input_dim = dataset.input_dim  # from property
        model = self.load_model(input_dim)

        # Make predictions
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(inputs)
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        # # Denormalize
        pred_vals = denormalize(all_preds,target_mean, target_std)
        true_vals = denormalize(all_targets, target_mean, target_std)

        if self.debug:
            raw_pred = all_preds[0]
            denorm_pred = pred_vals[0]
            raw_target = all_targets[0]
            denorm_target = true_vals[0]
            print("Raw prediction (normalized):", raw_pred)
            print("Prediction (denormalized):", denorm_pred)
            print("target value (normalized):", raw_target)
            print("True value (denormalized):", denorm_target)

            mae_norm = mean_absolute_error([raw_pred], [raw_target])
            mae_denorm = mean_absolute_error([denorm_pred], [denorm_target])
            print("MAE (normalized):", mae_norm)
            print("Target std:", target_std)
            print("Expected denorm MAE:", mae_norm * target_std)
            print("Actual denorm MAE:", mae_denorm)

        # pred_vals = all_preds
        # true_vals = pred_vals

        # Output results
        out_path = os.path.join(self.run_dir, "predictions.csv")
        df["true_values"] = true_vals
        df["prediction"] = pred_vals
        df.to_csv(out_path, index=False)
        print(f"✅ Predictions saved to {out_path}")
        self.calculate_evaluation_results(df, target_mean, target_std)

    def relative_mae_percentage(self,y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mean_target = np.mean(np.abs(y_true))
        if mean_target == 0:
            return np.inf  # Prevent division by zero
        return (mae / mean_target) * 100

    def smape(self,y_true, y_pred):
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
        return smape


if __name__ == '__main__':
    args = get_args()
    args_dict = vars(args)

    # Remove keys with None values (those not passed via CLI)
    user_config = {k.upper(): v for k, v in args_dict.items() if v is not None}

    predictor = Predictor(user_configs=user_config)
    predictor.predict()
