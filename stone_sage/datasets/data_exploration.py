from stone_sage.datasets.data_explorer import DataExplorer


def explore_partitioned_data(partitions: dict, user_config, run_data_path, plot_statistics = False):
    for label, df in partitions.items():
        print(f"\n🔍 Exploring {label} data...")
        explorer = DataExplorer(
            user_configs=user_config,
            df=df,
            label=label,
            data_path=run_data_path
        )
        if plot_statistics:
            explorer()
        else:
            explorer.save_statistics_log()