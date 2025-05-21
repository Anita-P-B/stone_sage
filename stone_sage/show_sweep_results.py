from stone_sage.arg_parser import get_args
import pandas as pd
import os

def show_sweep_results(sweep_path):
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.width", 0)
    log_path = os.path.join(sweep_path, "all_sweep_results.csv")
    df = pd.read_csv(log_path)
    sorted_df = df.sort_values(by= "val_rel_mae", ascending= True)
    print(sorted_df.head(10))



if __name__ == '__main__':
    args = get_args()
    sweep_path = args.sweep_path
    show_sweep_results(sweep_path)
