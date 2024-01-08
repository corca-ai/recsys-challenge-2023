import argparse
from typing import List, Optional

import numpy as np
import pandas as pd


def ensemble(
    file_list: List[str],
    weight: Optional[List[float]],
    method: str,
    output_path: str,
    row_id_name: str,
    target_name: str,
):
    if weight is None:
        weight = [1 / len(file_list)] * len(file_list)
    values = []
    for file in file_list:
        df = pd.read_csv(file, sep="\t")

        df = df.sort_values(row_id_name)
        values.append(df[target_name].to_numpy().copy())

    if method == "linear":
        result = sum([w * v for w, v in zip(weight, values)])
    elif method == "sigmoid":
        result = sum([w * np.log(v / (1 - v)) for w, v in zip(weight, values)])
        result = 1 / (1 + np.exp(-result))
    elif method == "tanh":
        result = sum([w * np.arctanh(2 * v - 1) for w, v in zip(weight, values)])
        result = (np.tanh(result) + 1) / 2
    else:
        raise NotImplementedError

    df.loc[:, target_name] = result
    df.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", nargs="+", help="input files (comma separated)"
    )
    parser.add_argument("-w", "--weight", nargs="+", help="weight", default=None)
    parser.add_argument(
        "-m",
        "--method",
        help="""how to ensemble (default: linear).
                Options: linear, sigmoid""",
        default="linear",
    )
    parser.add_argument("-o", "--output", help="output file", default="output.csv")
    parser.add_argument("--id_name", help="row id name", default="row_id")
    parser.add_argument("--target_name", help="target name", default="is_installed")
    args = parser.parse_args()

    ensemble(
        args.input,
        [float(w) for w in args.weight] if args.weight is not None else None,
        args.method,
        args.output,
        args.id_name,
        args.target_name,
    )
