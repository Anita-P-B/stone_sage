from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

def download_concrete_dataset(data_path, debug= False):
    # fetch dataset
    concrete_compressive_strength = fetch_ucirepo(id=165)
    # data (as pandas dataframes)
    X = concrete_compressive_strength.data.features
    y = concrete_compressive_strength.data.targets

    # metadata
    print(concrete_compressive_strength.metadata)
    if debug:
        # variable information
        print(concrete_compressive_strength.variables)
    df = pd.concat([X, y], axis=1)

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path,index = False)
    print(f"Dataset saved to {data_path}")
