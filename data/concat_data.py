import glob

import pandas as pd


def load(raw_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data from raw_path and return as a tuple of dataframes."""
    df_train: pd.DataFrame = None
    df_test: pd.DataFrame = None

    for type in ["train", "test"]:
        csv_files = list(glob.glob(f"{raw_path}/{type}/**.csv", recursive=True))
        df = pd.concat(pd.read_csv(f, sep="\t") for f in csv_files)
        df.to_parquet(f"{type}.parquet")

        if type == "train":
            df_train = df.copy()
        else:
            df_test = df.copy()
    return df_train, df_test


if __name__ == "__main__":
    df_train, df_test = load("/ssd/recsys2023/data")
