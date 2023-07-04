import warnings

warnings.filterwarnings("ignore")

from typing import Tuple, Dict, List
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

# set seed
seed = 269
random.seed(seed)


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


def sigmoid_ensemble(weights, model_results):
    num_models = len(model_results.columns)
    if weights == None:
        weights = [1 / num_models] * num_models
    values = []
    for column in model_results:
        values.append(model_results[column].to_numpy())
    result = sum([w * np.log(v / (1 - v)) for w, v in zip(weights, values)])
    result = 1 / (1 + np.exp(-result))
    return result


## Load Data
train = pd.read_parquet("/ssd/recsys2023/base/train.parquet")
test = pd.read_parquet("/ssd/recsys2023/base/test.parquet")

## Preprocessing
# Fill Null Cols
sliced = train[train.f_1 == 66]
for idx in [30, 31, 43, 51, 58, 59, 64, 65, 66, 67, 68, 69, 70]:
    f_group = train[["f_6", f"f_{idx}"]].groupby("f_6").mean().to_dict()[f"f_{idx}"]
    f_group_sliced = (
        sliced[["f_6", f"f_{idx}"]].groupby("f_6").mean().to_dict()[f"f_{idx}"]
    )
    train[f"f_{idx}"] = train[f"f_{idx}"].fillna(train["f_6"].map(f_group))
    test[f"f_{idx}"] = test[f"f_{idx}"].fillna(test["f_6"].map(f_group_sliced))

val = train[train.f_1 == 61]

# Target Encode
train, val, test, te_maps = target_encoder(
    train,
    val,
    test,
    cols=[
        "f_2",
        "f_4",
        "f_5",
        "f_6",
        "f_8",
        "f_10",
        "f_13",
        "f_14",
        "f_15",
        "f_16",
        "f_17",
        "f_18",
        "f_12",
        "f_19",
        "f_42",
        "f_71",
        "f_72",
        "f_73",
        "f_74",
        "f_75",
        "f_76",
        "f_77",
        "f_78",
        "f_79",
    ],
    target_col="is_installed",
    slice_recent_days=3,
    alpha=20,
)

# Define Columns
add_cat_features = [
    "f_71",
    "f_72",
    "f_73",
    "f_74",
    "f_75",
    "f_76",
    "f_77",
    "f_78",
    "f_79",
]
add_con_columns = [
    "TE-f_2-is_installed",
    "TE-f_4-is_installed",
    "TE-f_6-is_installed",
    "TE-f_15-is_installed",
    "TE-f_16-is_installed",
    "TE-f_17-is_installed",
    "TE-f_18-is_installed",
    "TE-f_19-is_installed",
    "TE-f_12-is_installed",
    "TE-f_42-is_installed",
    "TE-f_71-is_installed",
    "TE-f_72-is_installed",
    "TE-f_73-is_installed",
    "TE-f_74-is_installed",
    "TE-f_75-is_installed",
    "TE-f_76-is_installed",
    "TE-f_77-is_installed",
    "TE-f_78-is_installed",
    "TE-f_79-is_installed",
]
X_columns = list(
    set([f"f_{i}" for i in range(1, 80)] + add_cat_features + add_con_columns)
    - set(
        [
            "f_7",
            # "f_8",
            "f_9",
            "f_11",
            "f_24",
            "f_26",
            "f_27",
            "f_28",
            "f_29",
            "f_30",
            "f_31",
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
        + [f"f_{n}" for n in range(33, 41)]
    )
)
cat_features = [f"f_{i}" for i in range(2, 42)] + add_cat_features
cat_features = [col for col in cat_features if col in X_columns]
for col in cat_features:
    for data in [train, val, test]:
        data[col] = data[col].astype(int)
df_all = pd.concat([train, val], axis=0)
condition = df_all["f_1"] == 66
indices_to_change = df_all[condition].index
sample_size = int(len(indices_to_change) * 0.1)
indicies_selected = np.random.choice(indices_to_change, size=sample_size, replace=False)
df_all.loc[indicies_selected, "f_1"] = 67

# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
submission = pd.DataFrame(columns=["row_id", "is_clicked", "is_installed"])
submission["row_id"] = test[["f_0"]].copy().rename(columns={"f_0": "row_id"})
submission["is_clicked"] = 0

for i in [50, 65]:
    print(f"date: {i}")
    train = df_all[df_all.f_1 != i]
    val = df_all[df_all.f_1 == i]
    iter = 3000
    model_params = {
        "iterations": iter,
        "learning_rate": 0.1,
        "loss_function": "Logloss",
        "random_seed": seed,
        "task_type": "GPU",
    }
    fit_params = {
        "cat_features": cat_features,
        "verbose": 500,
        "eval_set": (val[X_columns], val["is_installed"]),
        "early_stopping_rounds": 100,
    }

    clf = CatBoostClassifier(**model_params)
    clf.fit(X=train[X_columns], y=train["is_installed"], **fit_params)

    # feature importance
    feature_importance = pd.DataFrame(
        {
            "feature": X_columns,
            "importance": clf.feature_importances_,
        }
    )
    feature_importance = feature_importance.sort_values(
        by="importance", ascending=False
    )
    feature_importance.to_csv("feature_importance.csv", index=False)

    y_pred = clf.predict_proba(val[X_columns])[:, 1]
    score = normalized_binary_cross_entropy(val["is_installed"], y_pred)
    print(f"score: {score}")

    # predict
    submission[f"date{i}"] = clf.predict_proba(test[X_columns])[:, 1]

submission["is_installed"] = sigmoid_ensemble(
    [0.5, 0.5], submission[[f"date{i}" for i in [50, 65]]]
)
submission[["row_id", "is_clicked", "is_installed"]].to_csv(
    f".cb_dayfold.csv", index=False, sep="\t"
)
