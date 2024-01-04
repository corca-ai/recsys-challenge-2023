import warnings

warnings.filterwarnings("ignore")

import os
import random
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import log_loss
from tqdm import tqdm

# set seed
seed = 42
random.seed(seed)

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")

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


def bin_encoder(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    prefix_name: str = "BIN",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    print(f"Columns to binary columns on {cols}")

    feature_list = []
    for col in tqdm(cols):
        feat_name = f"{prefix_name}-{col}"
        first_index = train[col].value_counts().index[0]
        train[feat_name] = train[col] == first_index
        val[feat_name] = val[col] == first_index
        test[feat_name] = test[col] == first_index
        feature_list.append(feat_name)
    print(feature_list)
    return train, val, test, feature_list


def nuniq_encoder(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    target_col: str = "is_clicked",
    prefix_name: str = "NQ",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    print(f"Columns to nuniq on {cols}")

    feature_list = []
    for col in tqdm(cols):
        feat_name = f"{prefix_name}-{col}-{target_col}"

        grouped_train = (
            train.groupby(col)[target_col]
            .nunique()
            .reset_index()
            .rename(columns={target_col: feat_name})
        )
        train = train.merge(grouped_train, on=col, how="left")
        val = val.merge(grouped_train, on=col, how="left")
        test = test.merge(grouped_train, on=col, how="left")

        feature_list.append(feat_name)
    print(feature_list)
    return train, val, test, feature_list


def combine_encoder(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[list[str]],
    prefix_name: str = "CB",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    print(f"Columns to combining on {cols}")

    feature_list = []
    for col in tqdm(cols):
        feat_name = f"{prefix_name}-{'-'.join(col)}"
        train[feat_name] = train[col].apply(
            lambda x: "-".join([str(i) for i in x]), axis=1
        )
        val[feat_name] = val[col].apply(lambda x: "-".join([str(i) for i in x]), axis=1)
        test[feat_name] = test[col].apply(
            lambda x: "-".join([str(i) for i in x]), axis=1
        )
        feature_list.append(feat_name)
    print(feature_list)
    return train, val, test, feature_list


def group_encoder(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    prefix_name: str = "GR",
    quantile: int = 5,
    target_col: str = "is_clicked",
    slice_recent_days: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    cut_day = (
        train.f_1.min()
        if slice_recent_days is None
        else train.f_1.max() - slice_recent_days
    )
    train_2 = train[train.f_1 < cut_day]
    train_1 = train[train.f_1 >= cut_day]
    print(f"Columns to grouping on {target_col} : {cols}")

    feature_list = []
    for col in tqdm(cols):
        feat_name = f"{prefix_name}-{col}-{target_col}"
        agg = train_1[[col, target_col]].groupby(col)[target_col].agg(["count", "mean"])
        counts = agg["count"]
        means = agg["mean"]
        quantiled = agg.sort_values("mean")["mean"].quantile(
            [round(1 / quantile * i, 3) for i in range(quantile + 1)]
        )
        agg[feat_name] = 0
        for i in range(quantile):
            agg.loc[
                (agg["mean"] >= quantiled[round(1 / quantile * i, 3)])
                & (agg["mean"] < quantiled[round(1 / quantile * (i + 1), 3)]),
                feat_name,
            ] = (
                i + 1
            )

        train_1 = train_1.merge(agg[feat_name], on=col, how="left")
        if len(train_2) > 0:
            train_2 = train_2.merge(agg[feat_name], on=col, how="left")
            train = pd.concat([train_2, train_1], axis=0)
        else:
            train = train_1

        val = val.merge(agg[feat_name], on=col, how="left")
        test = test.merge(agg[feat_name], on=col, how="left")

        feature_list.append(feat_name)

    return train, val, test, feature_list


def frequency_encoder(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cols: List[str],
    prefix_name: str = "FREQ",
    plot: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, List[str]]:
    fe_maps = {}
    feature_list = []
    for col in tqdm(cols):
        # fe = train[col].value_counts() / len(train)
        fe = np.log1p(len(train) / train[col].value_counts())
        feat_name = f"{prefix_name}-{col}"
        train.loc[:, feat_name] = train[col].map(fe)

        val.loc[:, feat_name] = val[col].map(fe)
        val.loc[val[feat_name].isna(), feat_name] = 0
        test.loc[:, feat_name] = test[col].map(fe)
        test.loc[test[feat_name].isna(), feat_name] = 0

        fe_dict = fe.to_dict()
        fe_dict.setdefault("-1", 0)
        fe_maps[col] = fe_dict
        feature_list.append(feat_name)

    if plot:
        fe.plot.bar(stacked=True)
        train.head(10)

    return train, val, test, fe_maps, feature_list


def target_encoder(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    prefix_name: str = "TE",
    target_col: str = "is_clicked",
    alpha: float = 5.0,
    slice_recent_days: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, list[str]]:
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
    feature_list = []
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

        feature_list.append(feat_name)
    return train, val, test, te_maps, feature_list


def normalized_binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1 - y_true) * np.log(1 - y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)

    p = np.sum(y_true) / len(y_true)

    return np.mean(term_0 + term_1, axis=0) / (p * np.log(p) + (1 - p) * np.log(1 - p))


def main():
    ## Load Data
    train = pd.read_parquet(os.path.join(DATA_PATH, "train.parquet"))
    train = train[train.f_1 != 60]  # KEY POINT
    test = pd.read_parquet(os.path.join(DATA_PATH, "test.parquet"))

    ## Preprocessing
    # Fill Null Cols
    sliced = train[train.f_1 == 66]
    for idx in [30, 31, 43, 51, 58, 59, 64, 65, 66, 67, 68, 69, 70]:
        f_group = train[["f_2", f"f_{idx}"]].groupby("f_2").mean().to_dict()[f"f_{idx}"]
        f_group_sliced = (
            sliced[["f_2", f"f_{idx}"]].groupby("f_2").mean().to_dict()[f"f_{idx}"]
        )
        train[f"f_{idx}"] = train[f"f_{idx}"].fillna(train["f_2"].map(f_group))
        test[f"f_{idx}"] = test[f"f_{idx}"].fillna(test["f_2"].map(f_group_sliced))

    f_2_day = train.groupby("f_2")["f_1"].min().to_dict()
    train["f_2_elasped"] = (
        (train["f_1"] - train["f_2"].map(f_2_day)).round(0).astype(int)
    ) > 10
    test["f_2_elasped"] = (
        (test["f_1"] - test["f_2"].map(f_2_day).fillna(67)).round(0).astype(int)
    ) > 10

    val = train[train.f_1 == 66]  # 일부러 이렇게 함. early stopping 안되게 하려고.

    # Adjust Continual Features
    for k, v in non_null_con_dict.items():
        for data in [train, val, test]:
            data[k] = data[k] // v

    bin_columns = ["f_52", "f_53", "f_54", "f_55", "f_56", "f_57"]
    combine_columns = [
        ["f_44", "f_45", "f_46", "f_47"],
        ["f_48", "f_49", "f_50"],
    ]
    nuniq_columns = []
    group_columns = []
    te_columns = [
        "f_2",
        "f_3",
        "f_4",
        "f_6",
        "f_12",
        "f_15",
        "f_16",
        "f_17",
        "f_18",
        "f_74",
        "f_75",
        "f_76",
    ]
    freq_columns = []
    remove_columns = [
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

    train, val, test, bin_features = bin_encoder(train, val, test, cols=bin_columns)
    train, val, test, nuniq_features = nuniq_encoder(
        train, val, test, cols=nuniq_columns, target_col="f_2"
    )
    train, val, test, combine_features = combine_encoder(
        train,
        val,
        test,
        cols=combine_columns,
    )
    train, val, test, group_features = group_encoder(
        train,
        val,
        test,
        cols=group_columns,
        target_col="is_installed",
        slice_recent_days=7,
    )
    train, val, test, _, te_features = target_encoder(
        train,
        val,
        test,
        cols=te_columns + combine_features + nuniq_features,
        target_col="is_installed",
        slice_recent_days=7,
    )
    train, val, test, _, freq_features = frequency_encoder(
        train, val, test, cols=freq_columns
    )

    # Define Columns
    add_cat_columns = (
        [
            "f_74",
            "f_75",
            "f_76",
            "f_77",
            "f_78",
            "f_79",
            "f_2_elasped",
        ]
        + group_features
        + combine_features
        + bin_features
    )
    add_con_columns = te_features + freq_features
    columns = list(
        set([f"f_{i}" for i in range(2, 80)] + add_cat_columns + add_con_columns)
        - set(remove_columns + [f"f_{n}" for n in range(30, 42)])
    )
    cat_features = set([f"f_{i}" for i in range(2, 42)] + add_cat_columns)

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
    }

    n_estimator = 200
    seed = 3258
    clf = lgb.LGBMClassifier(n_estimators=n_estimator, random_state=seed, n_jobs=4)
    clf.fit(train[columns], train["is_installed"], **fit_params)

    # feature importance
    feature_importance = pd.DataFrame(
        {
            "feature": columns,
            "importance": clf.feature_importances_,
        }
    )
    feature_importance = feature_importance.sort_values(
        by="importance", ascending=False
    )
    feature_importance.to_csv("feature_importance.csv", index=False)

    # predict for validation
    y_pred = clf.predict_proba(val[columns])[:, 1]
    loss = log_loss(val["is_installed"], y_pred)
    print(f"loss: {loss}")

    submission = pd.DataFrame(columns=["row_id", "is_clicked", "is_installed"])

    # predict
    submission = test[["f_0"]].copy().rename(columns={"f_0": "row_id"})
    submission["is_clicked"] = 0
    submission["is_installed"] = clf.predict_proba(test[columns])[:, 1]

    submission[["row_id", "is_clicked", "is_installed"]].to_csv(
        "lgb27.csv",
        index=False,
        sep="\t",
    )


if __name__ == "__main__":
    main()
