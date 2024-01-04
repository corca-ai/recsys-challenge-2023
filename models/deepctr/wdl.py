import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from deepctr_torch.inputs import DenseFeat, SparseFeat, get_feature_names
from deepctr_torch.models import WDL
from pytorch_optimizer import MADGRAD
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")


def seed_everything(seed=42):
    import os
    import random

    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def generate_cross_column(
    df: pd.DataFrame,
    cols: List[str],
    col_name: str = None,
    concat_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    series = df[cols[0]].astype(str)
    for col in cols[1:]:
        series = series + "_" + df[col].astype(str)

    if col_name is None:
        col_num = [col.split("_")[-1] for col in cols]
        col_name = "f" + "_" + "_".join(col_num)

    df[col_name] = series

    # df[col_name] = df[col_name].astype("category")

    if concat_features is not None:
        if col_name not in concat_features:
            concat_features.append(col_name)
        print(f"Concat features: {concat_features}")
    print(f"Generated {col_name}")
    return df, col_name


def target_encoder(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: List[str],
    prefix_name: str = "TE",
    target_col: str = "is_clicked",
    alpha: float = 5.0,
    slice_recent_days: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, List[str]]:

    cut_day = (
        train.f_1.min()
        if slice_recent_days is None
        else train.f_1.max() - slice_recent_days
    )
    train_2 = train[train.f_1 < cut_day]
    train_1 = train[train.f_1 >= cut_day]
    print(f"Columns to target encode on {target_col} : {cols}")

    feat_list = []
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

        smooth_dict = smooth.to_dict()

        smooth_dict.setdefault("-1", global_mean)

        te_maps[col] = smooth_dict
        feat_list.append(feat_name)

    return train, test, te_maps, feat_list


def frequency_encoder(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: List[str],
    prefix_name: str = "FREQ",
    plot: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, List[str]]:
    fe_maps = {}
    feat_list = []
    for col in tqdm(cols):
        # fe = train[col].value_counts() / len(train)
        fe = np.log1p(len(train) / train[col].value_counts())
        feat_name = f"{prefix_name}-{col}"
        train.loc[:, feat_name] = train[col].map(fe)
        test.loc[:, feat_name] = test[col].map(fe)
        test.loc[test[feat_name].isna(), feat_name] = 0

        fe_dict = fe.to_dict()
        fe_dict.setdefault("-1", 0)
        fe_maps[col] = fe_dict
        feat_list.append(feat_name)

    if plot:
        fe.plot.bar(stacked=True)
        train.head(10)

    return train, test, fe_maps, feat_list


def loo_target_encoder(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: List[str],
    prefix_name: str = "TE",
    target_col: str = "is_clicked",
    alpha: float = 5.0,
    slice_recent_days: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, List[str]]:
    cut_day = (
        train.f_1.min()
        if slice_recent_days is None
        else train.f_1.max() - slice_recent_days
    )
    train_2 = train[train.f_1 < cut_day]
    train_1 = train[train.f_1 >= cut_day]
    print(f"Columns to target encode on {target_col} : {cols}")

    feat_list = []
    te_maps = {}
    global_mean = train[target_col].mean()
    # use tqdm
    for col in tqdm(cols):
        feat_name = f"{prefix_name}-{col}-{target_col}"
        grouped = train_1[[col, target_col]].groupby(col)[target_col]
        counts = grouped.transform("count")
        sums = grouped.transform("sum")

        train_1.loc[:, feat_name] = (sums - train_1[target_col]) / (counts - 1)

        smooth = grouped.agg("mean")

        if len(train_2) > 0:
            train_2.loc[:, feat_name] = train_2[col].map(smooth)
            train_2.loc[train_2[feat_name].isna(), feat_name] = global_mean
            train = pd.concat([train_2, train_1], axis=0)
        else:
            train = train_1

        test.loc[:, feat_name] = test[col].map(smooth)
        test.loc[test[feat_name].isna(), feat_name] = global_mean

        smooth_dict = smooth.to_dict()

        smooth_dict.setdefault("-1", global_mean)

        te_maps[col] = smooth_dict
        feat_list.append(feat_name)

    return train, test, te_maps, feat_list


def preprocess():
    ## Load Data
    train = pd.read_parquet(os.path.join(DATA_PATH, "train.parquet"))
    test = pd.read_parquet(os.path.join(DATA_PATH, "test.parquet"))
    # train.loc[train["is_installed"] == 1, "is_clicked"] = 1

    f_51_mean = train.groupby(["f_4"])["f_51"].mean().reset_index()
    train["f_51"] = train["f_51"].fillna(
        train[["f_4"]].merge(f_51_mean, how="left", on="f_4")["f_51"]
    )
    test["f_51"] = test["f_51"].fillna(
        test[["f_4"]].merge(f_51_mean, how="left", on="f_4")["f_51"]
    )
    f_67_mean = train["f_67"].mean()
    train["f_67"] = train["f_67"].fillna(f_67_mean)
    test["f_67"] = test["f_67"].fillna(f_67_mean)
    for idx in [43, 51, 58, 59, 64, 65, 66, 67, 68, 69, 70]:
        train[f"f_{idx}"] = train[f"f_{idx}"].fillna(0)
        test[f"f_{idx}"] = test[f"f_{idx}"].fillna(0)

    for col in [f"f_{i}" for i in range(2, 42)]:
        less_f_6 = train[col].value_counts()[train[col].value_counts() < 10].index
        train.loc[train[col].isin(less_f_6), col] = -999
        test.loc[test[col].isin(less_f_6), col] = -999

    sparse_features = [
        f"f_{i}" for i in [5, 10, 14, 16, 20, 21, 22, 23, 25, 32, 37, 38, 39, 40, 41]
    ]
    dense_features = [
        f"f_{i}"
        for i in [
            1,
            42,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            65,
            67,
            68,
            69,
            70,
        ]
    ]

    remove_columns = ["f_7", "f_24", "f_26", "f_27", "f_28", "f_29"]
    sparse_features = [feat for feat in sparse_features if feat not in remove_columns]
    dense_features = [feat for feat in dense_features if feat not in remove_columns]
    null_columns = [f"f_{i}" for i in [43, 51, 58, 59, 66, 68, 69, 70, 64, 65, 67]]

    for df in [train, test]:
        df[null_columns] = np.log(df[null_columns] + 1)
        df["f_30"] = df["f_30"].fillna(2).astype(int)
        df["f_31"] = df["f_31"].fillna(2).astype(int)
        df["f_42"] = np.log(df["f_42"] / 0.0385640684536896 + 1)
        f_list = [f"f_{idx}" for idx in range(44, 51)]
        df[f_list] = np.log(df[f_list] / 0.5711214712545996 + 1)
        f_list = [f"f_{idx}" for idx in range(52, 58)]
        df[f_list] = np.log(df[f_list] / 0.0385640684536896 + 1)
        df["f_60"] = np.log(df["f_60"] / 8.07946038858253 + 1)
        df["f_61"] = np.log(df["f_61"] / 0.1478508992888889 + 1)
        df["f_62"] = np.log(df["f_62"] / 0.1292997091990755 + 1)
        df["f_63"] = np.log(df["f_63"] / 0.3552210926047521 + 1)

        df[["f_71_cat", "f_72_cat", "f_73_cat"]] = (
            df[["f_71", "f_72", "f_73"]] / 0.5711214712545996
        ).astype(int)
        df[["f_74_cat", "f_75_cat", "f_76_cat"]] = (
            df[["f_74", "f_75", "f_76"]] / 0.0385640684536896
        ).astype(int)
        df[["f_77_cat", "f_78_cat", "f_79_cat"]] = (
            df[["f_77", "f_78", "f_79"]] / 37.38457512430372
        ).astype(int)

        df[["f_71", "f_72", "f_73"]] = np.log(
            df[["f_71", "f_72", "f_73"]] / 0.5711214712545996 + 1
        )
        df[["f_74", "f_75", "f_76"]] = np.log(
            df[["f_74", "f_75", "f_76"]] / 0.0385640684536896 + 1
        )
        df[["f_77", "f_78", "f_79"]] = np.log(
            df[["f_77", "f_78", "f_79"]] / 37.38457512430372 + 1
        )

    # concat feature
    for column_list in [
        ["f_3", "f_4"],
        ["f_3", "f_20", "f_43", "f_66", "f_70"],
    ]:
        train, col_name = generate_cross_column(train, column_list)
        test, col_name = generate_cross_column(test, column_list)
        sparse_features.append(col_name)

    # cat_features = [f"f_{i}" for i in [2, 4, 6, 10, 12, 19, 15, 14, 42, 74, 75, 76]] + ["f_3_4"]
    target_encode_cat_features = (
        [f"f_{i}" for i in [2, 4, 6, 15, 19, 42]]
        + ["f_71_cat", "f_73_cat", "f_72_cat"]
        + ["f_74_cat", "f_76_cat", "f_75_cat"]
        + ["f_77_cat", "f_79_cat", "f_78_cat"]
    )

    target_encode_cat_features = [
        feat for feat in target_encode_cat_features if feat not in remove_columns
    ]
    train, test, _, feat_list = target_encoder(
        train,
        test,
        cols=target_encode_cat_features,
        target_col="is_installed",
        slice_recent_days=10,
        alpha=20,
    )
    dense_features += feat_list

    frequency_encode_cat_features = (
        [f"f_{i}" for i in [2, 4, 6, 15, 19, 42]]
        + ["f_71_cat", "f_73_cat", "f_72_cat"]
        + ["f_74_cat", "f_76_cat", "f_75_cat"]
        + ["f_77_cat", "f_79_cat", "f_78_cat"]
    )
    frequency_encode_cat_features = [
        feat for feat in frequency_encode_cat_features if feat not in remove_columns
    ]
    train, test, _, feat_list = frequency_encoder(
        train, test, cols=frequency_encode_cat_features
    )
    dense_features += feat_list

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        train[feat] = train[feat].astype(str)
        test[feat] = test[feat].astype(str)

        lbe.fit(train[feat])
        lbe_name_mapping = dict(zip(lbe.classes_, lbe.transform(lbe.classes_)))
        lbe_name_mapping.setdefault("-1", len(lbe.classes_))

        train[feat] = train[feat].map(lbe_name_mapping)
        test[feat] = test[feat].map(lbe_name_mapping)

    mms = MinMaxScaler(feature_range=(0, 1))
    mms.fit(train[dense_features])

    train[dense_features] = mms.transform(train[dense_features])
    test[dense_features] = mms.transform(test[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [
        SparseFeat(feat, train[feat].nunique() + 1) for feat in sparse_features
    ] + [
        DenseFeat(
            feat,
            1,
        )
        for feat in dense_features
    ]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    # 3.generate input data for model

    # train, test = train_test_split(data, test_size=0.2)

    return (
        train,
        test,
        linear_feature_columns,
        dnn_feature_columns,
    )


def fit_and_predict(
    train,
    test,
    linear_feature_columns,
    dnn_feature_columns,
    mode="train",
):
    test["f_0"] = test["f_0"] + 4000000
    target = ["is_installed"]
    device = "cpu"
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print("cuda ready...")
        device = "cuda:1"
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    kf = KFold(n_splits=20, shuffle=True, random_state=10)

    min_val_losses = []

    for i, (train_index, valid_index) in enumerate(kf.split(train)):
        train_fold = train.iloc[train_index]
        valid_fold = train.iloc[valid_index]

        train_model_input = {name: train_fold[name] for name in feature_names}
        valid_model_input = {name: valid_fold[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}

        model = WDL(
            dnn_hidden_units=[16, 4],
            # cross_parameterization="matrix",
            linear_feature_columns=linear_feature_columns,
            dnn_feature_columns=dnn_feature_columns,
            device=device,
            dnn_activation="prelu",
            dnn_use_bn=True,
            l2_reg_dnn=0,
            l2_reg_embedding=0,
            dnn_dropout=0,
            seed=10,
            # cross_num=3,
        )
        optim = MADGRAD(model.parameters(), lr=0.001)
        model.compile(
            optim,
            "binary_crossentropy",
            metrics=["binary_crossentropy", "auc"],
        )
        es = EarlyStopping(
            monitor="val_binary_crossentropy",
            min_delta=0,
            verbose=1,
            patience=2,
            mode="min",
        )
        filepath = f"submissions/{run_id}/model-{run_id}-{i}.ckpt"
        mdckpt = ModelCheckpoint(
            filepath=filepath,
            monitor="val_binary_crossentropy",
            mode="min",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        )

        history = model.fit(
            train_model_input,
            train_fold[target].values,
            batch_size=4096,
            epochs=20,
            verbose=1,
            validation_data=(valid_model_input, valid_fold[target].values),
            callbacks=[es, mdckpt],
        )

        min_val_losses.append(min(history.history["val_binary_crossentropy"]))

        model.load_state_dict(torch.load(filepath))

        total = pd.DataFrame(columns=["f_0", "y_pred"])

        for xx in (train, test):
            pred_ans = model.predict({name: xx[name] for name in feature_names}, 256)
            df = xx[["f_0"]].copy()
            df["y_pred"] = pred_ans
            total = pd.concat((total, df))
        total.to_parquet(f"submissions/{run_id}/preds-{run_id}-{i}.parquet")

        submission = pd.DataFrame(columns=["row_id", "is_clicked", "is_installed"])

        df = test[["f_0"]].copy().rename(columns={"f_0": "row_id"})
        df["is_clicked"] = 0
        df[target] = model.predict(test_model_input, 256)
        submission = pd.concat((submission, df))

        submission["row_id"] = submission["row_id"] - 4000000
        # evaluate
        submission[["row_id", "is_clicked", "is_installed"]].to_csv(
            "wdl.csv",
            index=False,
            sep="\t",
        )

    print(run_id)
    for min_val_loss in min_val_losses:
        print(min_val_loss)


if __name__ == "__main__":
    import git

    repo = git.Repo(search_parent_directories=True)
    global run_id
    run_id = str(repo.head.object.hexsha)[:4]

    mode = "test"
    train, test, linear_feature_columns, dnn_feature_columns = preprocess()
    print(train)
    print(test)
    import os

    os.makedirs(f"submissions/{run_id}", exist_ok=True)

    fit_and_predict(train, test, linear_feature_columns, dnn_feature_columns, mode=mode)
