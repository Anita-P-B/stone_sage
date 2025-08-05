import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--debug', action='store_true', default=None,
                        help="set on for debugging purposes")
    parser.add_argument('--overfit_test', action='store_true', default=None,
                        help="train on tiny dataset in overfit mode")
    parser.add_argument('--plot_statistics', action='store_true', default=None,
                        help="saves plots of statistical distributions")
    parser.add_argument('--force_download', action='store_true', default=None,
                        help="set on for redownload the dataset")
    parser.add_argument('--save_path', type=str, default=None, help="Path to save the model.")
    parser.add_argument('--optimizer', type=str, help="type of optimizer. options: adam| sgd|adamw")
    parser.add_argument('--loss', type=str, help="type of loss function. options: mae|mse|huber")
    parser.add_argument('--data_path', type=str, default="./data\concrete_data.csv", help="Path of dataset csv file.")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="path for saved checkpoint "
                                                                          "for evaluattion or resume training")
    parser.add_argument('--epochs', type=int, default=None, help="Number of epochs in train.")
    parser.add_argument('--learning_rate', type=float, help="Trainning learning rate")
    parser.add_argument('--model', type=str, help="Model archetecture.")
    parser.add_argument('--sweep_name', type=str, default="default",
                        help="Name for the sweep run folder.")
    parser.add_argument('--normalization', type=str, default=None,
                        help="normalization method.")
    parser.add_argument('--sweep_path', type=str,
                        help="paht of sweep log file")
    args = parser.parse_args()
    return args