import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--small_dataset', action='store_true', default=None,
                        help="Train on a small subset (10%) of the dataset for debugging purposes.")
    parser.add_argument('--debug', action='store_true', default=None,
                        help="set on for debugging purposes")
    parser.add_argument('--plot_statistics', action='store_true', default=None,
                        help="saves plots of statistical distributions")
    parser.add_argument('--force_download', action='store_true', default=None,
                        help="set on for redownload the dataset")
    parser.add_argument('--save_path', type=str, default=None, help="Path to save the model.")
    parser.add_argument('--data_path', type=str, default="./data\concrete_data.csv", help="Path of dataset csv file.")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="resumes training from "
                                                                          "given checkpoint.")
    parser.add_argument('--epochs', type=int, default=None, help="Number of epochs in train.")
    args = parser.parse_args()
    return args