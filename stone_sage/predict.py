
from stone_sage.arg_parser import get_args
import torch
import os
import pandas as pd
from stone_sage.models.stone_regressor import StoneRegressor  # adapt to your actual path
from stone_sage.datasets.stone_dataset import StoneDataset
from torch.utils.data import DataLoader
import json
from stone_sage.utils.utils import update_configs_with_dict, save_evaluation_summary, relative_mae_percentage
from stone_sage.utils.dict_to_class import DotDict
from sklearn.metrics import mean_absolute_error
from stone_sage.datasets.dataset_utils import get_train_mean_and_std

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

        self.configs = update_configs_with_dict(configs_obj, user_configs,self.debug)

    def load_model(self, input_dim):
        model = StoneRegressor.build(self.configs, input_dim)
        checkpoint = torch.load(self.configs.CHECKPOINT_PATH, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def calculate_evaluation_results(self, df):
        partitions = ["train", "val", "test"]
        metrics = {}
        for partition in partitions:
            part_df = df[df["partition"] == partition]
            true_targets = part_df[self.configs.TARGET_COLUMN].values
            predictions = part_df["prediction"].values

            mae = mean_absolute_error(true_targets, predictions)
            rel_mae = relative_mae_percentage(true_targets, predictions)

            metrics[f"{partition}_mae"] = mae
            metrics[f"{partition}_rel_mae"] = rel_mae


        save_evaluation_summary(self.run_dir, metrics)

    def predict(self):
        # Load data
        df_path = os.path.join(self.run_dir, "dataset_with_partitions.csv")
        df = pd.read_csv(df_path)



        train_df = df[df["partition"] == "train"]
        mean, std = get_train_mean_and_std(train_df, self.configs.TARGET_COLUMN)

        dataset = StoneDataset(df, target_column=self.configs.TARGET_COLUMN, mean=mean, std=std)
        loader = DataLoader(dataset, batch_size=self.configs.BATCH_SIZE)

        # Load model
        input_dim = dataset.input_dim  # from property
        model = self.load_model(input_dim)

        # Make predictions
        all_preds = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                all_preds.extend(outputs.cpu().numpy().flatten())

        # Output results
        out_path = os.path.join(self.run_dir, "predictions.csv")
        df["prediction"] = all_preds
        df.to_csv(out_path, index=False)
        print(f"✅ Predictions saved to {out_path}")

        self.calculate_evaluation_results(df)


if __name__ == '__main__':
    args = get_args()
    args_dict = vars(args)

    # Remove keys with None values (those not passed via CLI)
    user_config = {k.upper(): v for k, v in args_dict.items() if v is not None}

    predictor = Predictor(user_configs=user_config)
    predictor.predict()