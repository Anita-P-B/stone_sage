
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from stone_sage.arg_parser import get_args
from stone_sage.configs.config import Config
from stone_sage.utils.utils import update_configs_with_dict
from stone_sage.datasets.data_loader import load_or_download_data

class DataExplorer:
    def __init__(self, user_configs = None):
        # set configs
        self.static_configs = Config()
        self.configs = update_configs_with_dict(self.static_configs, user_configs or {})
        self.df = load_or_download_data(
        path=self.configs.DATA_PATH,
        force_download=self.configs.FORCE_DOWNLOAD,
        expected_checksum=self.configs.CHECKSUM,
        debug = self.configs.DEBUG
    )
        self.numeric_columns = self.df.select_dtypes(include=['number']).columns
        self.stats_dir = os.path.join(os.path.dirname(self.configs.DATA_PATH), "statistics")
        os.makedirs(self.stats_dir, exist_ok=True)

    def basic_info(self):
        print("\n🧾 Basic Information")
        print(f"Shape: {self.df.shape}")
        print("\nTypes:\n", self.df.dtypes)
        print("\nMissing Values:\n", self.df.isnull().sum())
        return self

    def plot_distributions(self):
        print(f"\n🎨 Plotting Distributions with Mean & Median: {list(self.numeric_columns)}")
        for col in self.numeric_columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[col], bins=30, kde=False, edgecolor='black')
            plt.axvline(self.df[col].mean(), color='red', linestyle='--', label=f"Mean: {self.df[col].mean():.2f}")
            plt.axvline(self.df[col].median(), color='blue', linestyle='-',
                        label=f"Median: {self.df[col].median():.2f}")
            plt.legend()
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            output_path = os.path.join(self.stats_dir, f"{col}_distribution.png")
            plt.savefig(output_path)
            if self.configs.DEBUG:
                plt.show()
        return self

    def correlation_heatmap(self):
        print("\n🔥 Correlation Heatmap:")
        plt.figure(figsize=(10, 8))
        corr = self.df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Feature Correlation Heatmap")
        output_path = os.path.join(self.stats_dir, "correlation_heatmap.png")
        plt.savefig(output_path)
        if self.configs.DEBUG:
            plt.show()
        return self

    def plot_boxplots(self):
        print("\n📦 Plotting Boxplots for Numerical Features")
        for col in self.numeric_columns:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=self.df[col])
            plt.title(f"Boxplot of {col}")
            output_path = os.path.join(self.stats_dir, f"{col}_box_plot.png")
            plt.savefig(output_path)
            if self.configs.DEBUG:
                plt.show()
        return self

    def save_statistics_log(self):
        # Add median to the describe DataFrame
        desc = self.df[self.numeric_columns].describe().T
        print("\n📊 Descriptive Statistics:")
        print(desc)
        # Save to CSV
        output_path = os.path.join(self.stats_dir, "dataset_statistics.csv")
        desc.to_csv(output_path)
        print(f"\n📜 Dataset statistics saved to: {output_path}")
        return self

if __name__ == "__main__":
    DATA_PATH = "./data/concrete_data.csv"
    args = get_args()
    args_dict = vars(args)

    # Remove keys with None values (those not passed via CLI)
    user_config = {k.upper(): v for k, v in args_dict.items() if v is not None}
    explorer = DataExplorer(user_configs= user_config)
    explorer\
        .basic_info()\
        .plot_distributions()\
        .correlation_heatmap()\
        .plot_boxplots() \
        .save_statistics_log()