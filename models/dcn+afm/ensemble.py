import argparse
from typing import List, Optional

import pandas as pd
import numpy as np


def ensemble(
    file_list: List[str],
    weight: Optional[List[float]],
    method: str,
    output_path: str,
    row_id_name: str,
    target_name: str,
    p: float,
):
    if weight is None:
        weight = [1 / len(file_list)] * len(file_list)
    values = []
    for file in file_list:
        df = pd.read_csv(file, sep="\t")

        df = df.sort_values(row_id_name)
        values.append(df[target_name].to_numpy())

    if method == "linear":
        result = sum([w * v for w, v in zip(weight, values)])
    elif method == "sigmoid":
        result = sum([w * np.log(v / (1 - v)) for w, v in zip(weight, values)])
        result = 1 / (1 + np.exp(-result))
    elif method == "tanh":
        result = sum([w * np.arctanh(2 * v - 1) for w, v in zip(weight, values)])
        result = (np.tanh(result) + 1) / 2
    elif method == "power":
        result = sum([w * v**p for w, v in zip(weight, values)])
    else:
        raise NotImplementedError

    df.loc[:, target_name] = result
    df.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input files (comma separated)")
    parser.add_argument("-w", "--weight", help="weight", default=None)
    parser.add_argument(
        "-m",
        "--method",
        help="""how to ensemble (default: linear).
                Options: linear, sigmoid""",
        default="linear",
    )
    parser.add_argument("-o", "--output", help="output file", default="output.csv")
    parser.add_argument("--id_name", help="row id name", default="row_id")
    parser.add_argument("--target_name", help="target name", default="target")
    parser.add_argument("-p", help="power", default=2, type=float)
    args = parser.parse_args()

    ensemble(
        args.input.split(","),
        [float(w) for w in args.weight.split(",")] if args.weight is not None else None,
        args.method,
        args.output,
        args.id_name,
        args.target_name,
        args.p,
    )
