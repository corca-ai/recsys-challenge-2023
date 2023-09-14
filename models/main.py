import argparse
import random
from math import e
from typing import List, Optional

import numpy as np
import pandas as pd


def ensemble(
    file_list: List[str],
    weight: Optional[List[float]] = None,
    output_path: str = None,
    row_id_name: str = "row_id",
    target_name: str = "is_installed",
):
    if weight is None:
        weight = [1 / len(file_list)] * len(file_list)
    values = []
    for file in file_list:
        df = pd.read_csv(file, sep="\t")
        if len(df) != 160973:
            print(file)
            continue
        df = df.sort_values(row_id_name)
        values.append(df[target_name].to_numpy().copy())
    result = sum([w * v for w, v in zip(weight, values)])
    df["is_clicked"] = 0.0
    df.loc[:, target_name] = result

    if output_path is not None:
        df.to_csv(output_path, sep="\t", index=False)
    else:
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", nargs="+", help="input files (comma separated)"
    )
    parser.add_argument("--id_name", help="row id name", default="row_id")
    parser.add_argument("--target_name", help="target name", default="is_installed")
    args = parser.parse_args()

    ensemble(
        args.input,
        [0.55, 0.2, 0.15, 0.1],
        "real_final.csv",
        args.id_name,
        args.target_name,
    )
