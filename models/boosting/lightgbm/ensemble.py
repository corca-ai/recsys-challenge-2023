from typing import List, Optional

import numpy as np
import pandas as pd
import seaborn as sns


def ensemble(
    file_list: List[str],
    weight: Optional[List[float]] = None,
    method: str = "sigmoid",
    output_path: str = None,
    row_id_name: str = "row_id",
    target_name: str = "is_installed",
    plot: bool = False,
):
    if weight is None:
        weight = [1 / len(file_list)] * len(file_list)
    values = []

    all_models = pd.read_csv(file_list[0], sep="\t").copy()
    all_models.rename(columns={"f_0": "row_id"}, inplace=True)
    all_models["is_clicked"] = 0
    all_models.sort_values("row_id", inplace=True)
    for i, file in enumerate(file_list):
        df = pd.read_csv(file, sep="\t")
        if len(df) != 160973:
            print(file)
            continue
        df = df.sort_values(row_id_name)
        values.append(df[target_name].to_numpy().copy())

        df.sort_values("row_id", inplace=True)
        df.rename(columns={"is_installed": i}, inplace=True)
        all_models = all_models.merge(df[["row_id", i]], on="row_id", how="left")

    if plot:
        sns.heatmap(
            all_models.iloc[:, -len(file_list) :].corr(),
            annot=True,
            fmt=".4f",
            cmap="Blues",
        )

    name = "|".join([str(w).split(".")[-1][:3] for w in weight])
    tmp_output_path = output_path.replace(".csv", f"_{method}_{name}.csv")
    if method == "linear":
        result = sum([w * v for w, v in zip(weight, values)])
    elif method == "sigmoid":
        result = sum([w * np.log(v / (1 - v)) for w, v in zip(weight, values)])
        result = 1 / (1 + np.exp(-result))
    else:
        raise NotImplementedError

    df = pd.read_csv(file_list[0], sep="\t").copy().sort_values(row_id_name)
    df.loc[:, target_name] = result
    df[target_name] = df[target_name] / df[target_name].mean() * 0.13581734064740902
    df[target_name] = np.clip(df[target_name], 0, 1)
    if output_path is not None:
        df.to_csv(tmp_output_path, sep="\t", index=False)
    else:
        return df
