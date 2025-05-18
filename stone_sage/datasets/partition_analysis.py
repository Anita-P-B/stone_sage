from stone_sage.datasets.statistics_explorer import DataStats


def analyze_partitioned_data(partitions: dict, user_config, run_data_path, plot_statistics = False, verbose = False):
    for label, df in partitions.items():
        if verbose:
            print(f"\n🔍 Exploring {label} data...")
        explorer = DataStats(
            user_configs=user_config,
            df=df,
            label=label,
            data_path=run_data_path
        )
        if plot_statistics:
            explorer()
        else:
            explorer.save_statistics_log()