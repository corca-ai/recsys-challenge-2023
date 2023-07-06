import warnings

warnings.filterwarnings("ignore")

import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from tqdm import tqdm

# set seed
seed = 269
random.seed(seed)
np.random.seed(seed)


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
train = train[train.f_1 != 60]
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
        "f_6",
        "f_15",
        "f_16",
        "f_17",
        "f_18",
        "f_12",
        # "f_74",
        "f_75",
        # "f_76",
    ],
    target_col="is_installed",
    slice_recent_days=3,
)

# Define Columns
add_cat_features = ["f_74", "f_75", "f_76", "f_77", "f_78", "f_79"]
add_con_columns = [f"TE-{col}-is_installed" for col in te_maps.keys()]
X_columns = list(
    set([f"f_{i}" for i in range(2, 80)] + add_cat_features + add_con_columns)
    - set(
        [
            "f_7",
            "f_8",
            "f_24",
            "f_26",
            "f_27",
            "f_28",
            "f_29",
            "f_30",
            "f_31",
            "f_19",
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
for data in [train, val, test]:
    data[cat_features] = data[cat_features].astype("int")

f_19s = [
    [26758],
    [29982],
    [25370],
    [21743],
    [25952],
    [16851],
    [27564],
    [25773],
    [20534],
    [28773, 28370, 22146, 20131, 30385, 17657, 18866, 26355, 27967, 0],
]

iter = 300
learning_rate = 0.1

y_pred = []
y_true = []
valid = pd.DataFrame(columns=["f_0", "y_pred"])
submission = pd.DataFrame(columns=["row_id", "is_clicked", "is_installed"])
for f_19 in f_19s:
    print("[+] f_19: ", f_19)
    train_ = train[train.f_19.isin(f_19)]
    val_ = val[val.f_19.isin(f_19)]
    test_ = test[test.f_19.isin(f_19)]

    if len(f_19) > 1:
        X_columns += ["f_19"]

    model = CatBoostClassifier(
        iterations=iter,
        learning_rate=learning_rate,
    )
    model.fit(
        X=train_[X_columns],
        y=train_["is_installed"],
        cat_features=cat_features,
        eval_set=(val_[X_columns], val_["is_installed"]),
        use_best_model=True,
        # early_stopping_rounds=5,
        verbose=False,
    )

    # predict
    y_pred += model.predict_proba(val_[X_columns])[:, 1].tolist()
    y_true += val_["is_installed"].tolist()

    df = val_[["f_0"]].copy()
    df["y_pred"] = model.predict_proba(val_[X_columns])[:, 1]
    valid = pd.concat((valid, df))

    # predict
    df = test_[["f_0"]].copy().rename(columns={"f_0": "row_id"})
    df["is_clicked"] = 0
    df["is_installed"] = model.predict_proba(test_[X_columns])[:, 1]
    submission = pd.concat((submission, df))

# evaluate
score = normalized_binary_cross_entropy(np.array(y_true), np.array(y_pred))
print(f"[+] Logloss: {log_loss(y_true, y_pred)}")
print(f"[+] NBCE: {score}")

submission[["row_id", "is_clicked", "is_installed"]].to_csv(
    f"cb_19_{iter}_{score:.5f}.csv", index=False, sep="\t"
)
