import warnings

warnings.filterwarnings("ignore")

import random
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import early_stopping, log_evaluation
from tqdm import tqdm

# set seed
seed = 42
random.seed(seed)

non_null_con_dict = {
    "f_42": 0.0385640684536896,
    "f_44": 0.5711214712545996,
    "f_45": 0.5711214712545996,
    "f_46": 0.5711214712545996,
    "f_47": 0.5711214712545996,
    "f_48": 0.5711214712545996,
    "f_49": 0.5711214712545996,
    "f_50": 0.5711214712545996,
    "f_52": 0.0385640684536896,
    "f_53": 0.0385640684536896,
    "f_54": 0.0385640684536896,
    "f_55": 0.0385640684536896,
    "f_56": 0.0385640684536896,
    "f_57": 0.0385640684536896,
    "f_60": 8.07946038858253,
    "f_61": 0.1478508992888889,
    "f_62": 0.1292997091990755,
    "f_63": 0.3552210926047521,
    "f_71": 0.5711214712545996,
    "f_72": 0.5711214712545996,
    "f_73": 0.5711214712545996,
    "f_74": 0.0385640684536896,
    "f_75": 0.0385640684536896,
    "f_76": 0.0385640684536896,
    "f_77": 37.38457512430372,
    "f_78": 37.38457512430372,
    "f_79": 37.38457512430372,
}


def target_encoder(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cols: List[str],
    prefix_name: str = "TE",
    target_col: str = "is_clicked",
    alpha: float = 5.0,
    slice_recent_days: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    cut_day = (
        train.f_1.min()
        if slice_recent_days is None
        else train.f_1.max() - slice_recent_days
    )
    train_2 = train[train.f_1 < cut_day]
    train_1 = train[train.f_1 >= cut_day]
    print(f"Columns to target encode on {target_col} : {cols}")

    te_maps = {}
    global_mean = train[target_col].mean()
    # use tqdm
    for col in tqdm(cols):
        feat_name = f"{prefix_name}-{col}-{target_col}"
        agg = train_1[[col, target_col]].groupby(col)[target_col].agg(["count", "mean"])
        counts = agg["count"]
        means = agg["mean"]
        smooth = (counts * means + alpha * global_mean) / (counts + alpha)

        train_1.loc[:, feat_name] = train_1[col].map(smooth)
        if len(train_2) > 0:
            train_2.loc[:, feat_name] = train_2[col].map(smooth)
            train_2.loc[train_2[feat_name].isna(), feat_name] = global_mean
            train = pd.concat([train_2, train_1], axis=0)
        else:
            train = train_1

        test.loc[:, feat_name] = test[col].map(smooth)
        test.loc[test[feat_name].isna(), feat_name] = global_mean

        val.loc[:, feat_name] = val[col].map(smooth)
        val.loc[val[feat_name].isna(), feat_name] = global_mean

        smooth_dict = smooth.to_dict()
        smooth_dict.setdefault("-1", global_mean)
        te_maps[col] = smooth_dict
    return train, val, test, te_maps


def normalized_binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1 - y_true) * np.log(1 - y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)

    p = np.sum(y_true) / len(y_true)

    return np.mean(term_0 + term_1, axis=0) / (p * np.log(p) + (1 - p) * np.log(1 - p))


## Load Data
train = pd.read_parquet("/ssd/recsys2023/base/train.parquet")
train = train[train.f_1 != 60]  # KEY POINT
test = pd.read_parquet("/ssd/recsys2023/base/test.parquet")

## Preprocessing
# Fill Null Cols
sliced = train[train.f_1 == 66]
for idx in [30, 31, 43, 51, 58, 59, 64, 65, 66, 67, 68, 69, 70]:
    if idx == 51:
        f_group = train[["f_6", "f_51"]].groupby("f_6").mean().to_dict()["f_51"]
        f_group_sliced = sliced[["f_6", "f_51"]].groupby("f_6").mean().to_dict()["f_51"]
        train["f_51"] = train["f_51"].fillna(train["f_6"].map(f_group))
        test["f_51"] = test["f_51"].fillna(test["f_6"].map(f_group_sliced))
        # for data in [train, test]:
        #     data["f_51"] = data["f_51"].fillna(data["f_6"].map(f_group))
        # data["f_51"] = data["f_51"].fillna(data["f_51"].median())
    if idx == 67:
        f_group = train[["f_21", "f_67"]].groupby("f_21").median().to_dict()["f_67"]
        f_group_sliced = (
            sliced[["f_21", "f_67"]].groupby("f_21").median().to_dict()["f_67"]
        )
        train["f_67"] = train["f_67"].fillna(train["f_21"].map(f_group))
        test["f_67"] = test["f_67"].fillna(test["f_21"].map(f_group_sliced))
        # for data in [train, test]:
        #     data[f"f_{idx}"] = data[f"f_{idx}"].fillna(data["f_21"].map(f_group))
        # data[f"f_{idx}"] = data[f"f_{idx}"].fillna(data[f"f_21"].median())
    # if idx == 69:
    #     f_group = train[["f_2", "f_67"]].groupby("f_2").mean().to_dict()["f_67"]
    #     for data in [train, val, test]:
    #         data[f"f_{idx}"] = data[f"f_{idx}"].fillna(data[f"f_2"].map(f_group))
    if idx in [30, 31]:
        for data in [train, test]:
            data[f"f_{idx}"] = data[f"f_{idx}"].fillna(2)
            data[f"f_{idx}"] = data[f"f_{idx}"].astype(int)
    else:
        for data in [train, test]:
            data[f"f_{idx}"] = data[f"f_{idx}"].fillna(0)
    data[f"f_{idx}"] = data[f"f_{idx}"].fillna(0)

val = train[train.f_1 == 61]  # 일부러 이렇게 함. early stopping 안되게 하려고.
train = train[train.f_1 != 60]

# Adjust Continual Features
for k, v in non_null_con_dict.items():
    for data in [train, val, test]:
        data[k] = data[k] // v

# Cyclic Encode
for col, max_val in zip(["f_9", "f_11"], [7, 24]):
    sorted = train[col].value_counts().index[::-1]
    sorted_dict = {k: v for v, k in enumerate(sorted)}
    for data in [train, val, test]:
        data[f"{col}"] = data[col].map(sorted_dict)
        data[f"{col}-sin"] = np.sin(2 * np.pi * data[f"{col}"] / max_val)
        data[f"{col}-cos"] = np.cos(2 * np.pi * data[f"{col}"] / max_val)

# Target Encode
train, val, test, te_maps = target_encoder(
    train,
    val,
    test,
    cols=["f_2", "f_4", "f_6", "f_15", "f_16"],
    target_col="is_installed",
    slice_recent_days=7,
)

# Define Columns
add_cat_features = []
add_con_columns = [
    "f_9-cos",
    "f_9-sin",
    "f_11-cos",
    "f_11-sin",
    "TE-f_2-is_installed",
    "TE-f_4-is_installed",
    "TE-f_6-is_installed",
    "TE-f_15-is_installed",
    "TE-f_16-is_installed",
]
columns = list(
    set([f"f_{i}" for i in range(2, 80)] + add_cat_features + add_con_columns)
    - set(
        [
            "f_7",
            "f_8",
            "f_27",
            "f_28",
            "f_29",
            ## rm originals of cross features
            "f_17",
            "f_21",
            "f_12",
            "f_9",
            "f_11",
            "f_3",
            "f_22",
            "f_74",
            "f_76",
        ]
        + [f"f_{n}" for n in range(30, 41)]
    )
)
cat_features = [f"f_{i}" for i in range(2, 42)] + add_cat_features
for col in cat_features:
    train[col] = train[col].astype("category")
    val[col] = pd.Categorical(val[col], categories=train[col].cat.categories)
    test[col] = pd.Categorical(test[col], categories=train[col].cat.categories)

fit_params = {
    "eval_metric": "logloss",
    "eval_set": [
        (val[columns], val["is_installed"]),
    ],
    "eval_names": ["val"],
    "callbacks": [log_evaluation(200), early_stopping(10)],
}

for seed in [419]:
    clf = lgb.LGBMClassifier(n_estimators=200, random_state=seed, n_jobs=4)
    clf.fit(train[columns], train["is_installed"], **fit_params)

    # # feature importance
    # feature_importance = pd.DataFrame(
    #     {
    #         "feature": columns,
    #         "importance": clf.feature_importances_,
    #     }
    # )
    # feature_importance = feature_importance.sort_values(
    #     by="importance", ascending=False
    # )
    # feature_importance.to_csv("feature_importance.csv", index=False)

    # predict for validation
    y_pred = clf.predict_proba(val[columns])[:, 1]
    score = normalized_binary_cross_entropy(val["is_installed"], y_pred)
    print(f"score: {score}")

    submission = pd.DataFrame(columns=["row_id", "is_clicked", "is_installed"])

    # predict
    df = test[["f_0"]].copy().rename(columns={"f_0": "row_id"})
    df["is_clicked"] = 0
    df["is_installed"] = clf.predict_proba(test[columns])[:, 1]
    submission = pd.concat((submission, df))
    submission[["row_id", "is_clicked", "is_installed"]].to_csv(
        "lgb.csv",
        index=False,
        sep="\t",
    )
