import os
from typing import Dict, List, Optional, Tuple

import category_encoders as ce
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from deepctr_torch.inputs import DenseFeat, SparseFeat, get_feature_names
from deepctr_torch.models import AutoInt
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

    df[col_name] = df[col_name].astype("category")

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
    slice_recent_days: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, List[str]]:
    cut_day = (
        train.f_1.min()
        if slice_recent_days is None
        else train.f_1.max() - slice_recent_days
    )
    train_1 = train[train.f_1 >= cut_day]
    print(f"Columns to target encode on {target_col} : {cols}")

    feat_list = []

    encoder = ce.CatBoostEncoder(cols=cols, verbose=1, a=5)
    encoder = encoder.fit(train_1[cols], train_1[target_col])

    train_transformed = (
        encoder.transform(train[cols])
        .add_prefix(prefix_name)
        .add_suffix(f"_{target_col}")
    )
    train = pd.concat([train, train_transformed], axis=1)

    test_transformed = (
        encoder.transform(test[cols])
        .add_prefix(prefix_name)
        .add_suffix(f"_{target_col}")
    )
    test = pd.concat(
        [
            test,
            test_transformed,
        ],
        axis=1,
    )

    feat_list = test_transformed.columns.tolist()

    return train, test, {}, feat_list


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


def preprocess():
    ## Load Data
    train = pd.read_parquet(os.path.join(DATA_PATH, "train.parquet"))
    test = pd.read_parquet(os.path.join(DATA_PATH, "test.parquet"))

    train.loc[train["is_installed"] == 1, "is_clicked"] = 1

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

    for df in [train, test]:
        df[["f_74_cat", "f_75_cat", "f_76_cat"]] = (
            df[["f_74", "f_75", "f_76"]] / 0.0385640684536896
        ).astype(int)

    # data filtering
    for col in [
        f"f_{i}"
        for i in [
            42,
            43,
            51,
            52,
            54,
            55,
            56,
            57,
            58,
            59,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            72,
            73,
        ]
    ]:
        # Catch IQR outliers
        q1 = train[col].quantile(0.25)
        q3 = train[col].quantile(0.75)
        iqr = q3 - q1
        train.loc[(train[col] < q1 - 1.5 * iqr), col] = q1 - 1.5 * iqr
        test.loc[(test[col] < q1 - 1.5 * iqr), col] = q1 - 1.5 * iqr
        train.loc[(train[col] > q3 + 1.5 * iqr), col] = q3 + 1.5 * iqr
        test.loc[(test[col] > q3 + 1.5 * iqr), col] = q3 + 1.5 * iqr

    for col in [f"f_{i}" for i in [44, 45, 47, 48, 49, 50, 53, 60, 71, 78]]:
        # Catch IQR outliers
        q1 = 0
        q3 = train[col].quantile(0.99)
        iqr = q3 - q1
        train.loc[(train[col] > q3 + iqr * 1.5), col] = q3 + iqr * 1.5
        test.loc[(test[col] > q3 + iqr * 1.5), col] = q3 + iqr * 1.5

    for col in [f"f_{i}" for i in range(2, 42)]:
        less_f_6 = train[col].value_counts()[train[col].value_counts() < 10].index
        train.loc[train[col].isin(less_f_6), col] = -999
        test.loc[test[col].isin(less_f_6), col] = -999

    sparse_features = [
        f"f_{i}" for i in [2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19, 42]
    ] + [
        "f_74_cat",
        "f_75_cat",
        "f_76_cat",
    ]

    dense_features = ["f_" + str(i) for i in range(42, 80) if i not in [74, 75, 76]]
    dense_features.append("f_1")
    remove_columns = ["f_30", "f_31"]
    sparse_features = [feat for feat in sparse_features if feat not in remove_columns]
    dense_features = [feat for feat in dense_features if feat not in remove_columns]
    null_columns = [f"f_{i}" for i in [43, 51, 58, 59, 66, 68, 69, 70, 64, 65, 67]]

    for df in [train, test]:
        df[null_columns] = np.log(df[null_columns] + 1)
        df["f_42"] = np.log(df["f_42"] / 0.0385640684536896 + 1)
        f_list = [f"f_{idx}" for idx in range(44, 51)]
        df[f_list] = np.log(df[f_list] / 0.5711214712545996 + 1)
        f_list = [f"f_{idx}" for idx in range(52, 58)]
        df[f_list] = np.log(df[f_list] / 0.0385640684536896 + 1)
        df["f_60"] = np.log(df["f_60"] / 8.07946038858253 + 1)
        df["f_61"] = np.log(df["f_61"] / 0.1478508992888889 + 1)
        df["f_62"] = np.log(df["f_62"] / 0.1292997091990755 + 1)
        df["f_63"] = np.log(df["f_63"] / 0.3552210926047521 + 1)

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
        # ["f_71", "f_73", "f_72"],
        # ["f_74", "f_76", "f_75"],
        # ["f_3", "f_20", "f_43", "f_66", "f_70"],
        ["f_3", "f_4"],
    ]:
        train, col_name = generate_cross_column(train, column_list)
        test, col_name = generate_cross_column(test, column_list)
        sparse_features.append(col_name)

    f_2_day = train.groupby("f_2")["f_1"].min().to_dict()

    train["f_2_day"] = train["f_1"] - train["f_2"].map(f_2_day)
    test["f_2_day"] = test["f_1"] - test["f_2"].map(f_2_day).fillna(
        train["f_1"].max() + 1
    )

    train["f_2_day_more_than_10"] = (train["f_2_day"] > 10).astype(int)
    test["f_2_day_more_than_10"] = (test["f_2_day"] > 10).astype(int)
    # sparse_features.append("f_2_day_more_than_10")

    cat_features = (
        [f"f_{i}" for i in [2, 4, 5, 6, 10, 12, 14, 15, 42, 74]]
        + ["f_3_4"]
        + ["f_2_day"]
    )

    cat_features = [feat for feat in cat_features if feat not in remove_columns]
    train, test, _, feat_list = target_encoder(
        train, test, cols=cat_features, target_col="is_installed"
    )
    dense_features += feat_list
    train, test, _, feat_list = target_encoder(
        train, test, cols=cat_features, target_col="is_clicked"
    )
    dense_features += feat_list

    freq_features = [f"f_{i}" for i in [2, 4, 6, 19, 42]] + ["f_3_4"]
    freq_features = [feat for feat in freq_features if feat not in remove_columns]

    train, test, _, feat_list = frequency_encoder(train, test, cols=freq_features)
    dense_features += feat_list

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        train[feat] = train[feat].astype(str)
        test[feat] = test[feat].astype(str)

        lbe.fit(train[feat].astype(str))
        lbe_name_mapping = dict(zip(lbe.classes_, lbe.transform(lbe.classes_)))

        train[feat] = train[feat].map(lbe_name_mapping)
        test[feat] = test[feat].map(lbe_name_mapping).fillna(len(lbe.classes_))

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


def normalized_binary_cross_entropy_loss(
    y_pred: torch.FloatTensor, y_true: torch.LongTensor, reduction="mean"
):
    loss = F.binary_cross_entropy(y_pred, y_true.float(), reduction=reduction)
    p = y_true.sum().float() / len(y_true)
    loss = -loss / (p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return loss


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
        device = "cuda:2"
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    min_val_losses = []

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    filepaths = []
    days = train.f_1.unique()
    # for i, (train_index, valid_index) in enumerate(kf.split(train)):
    for i, d in enumerate(days):
        # train_fold = train.iloc[train_index]
        # valid_fold = train.iloc[valid_index]
        train_fold = train[train.f_1 != d]
        valid_fold = train[train.f_1 == d]

        train_model_input = {name: train_fold[name] for name in feature_names}
        valid_model_input = {name: valid_fold[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}
        model = AutoInt(
            dnn_feature_columns=dnn_feature_columns,
            linear_feature_columns=linear_feature_columns,
            device=device,
            dnn_activation="prelu",
            dnn_use_bn=True,
            l2_reg_dnn=0,
            l2_reg_embedding=0,
        )

        model.compile(
            MADGRAD(model.parameters(), lr=0.0001),
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
        filepath = f"./notebooks/jhkim/submission/ai-{i+45}.csv"
        filepaths.append(filepath)
        submission[["row_id", "is_clicked", "is_installed"]].to_csv(
            "autoint.csv", index=False, sep="\t"
        )

    print(run_id)
    for min_val_loss in min_val_losses:
        print(min_val_loss)


if __name__ == "__main__":
    global seed

    for i in range(11, 12):
        seed = i

        seed_everything(seed)

        import git

        repo = git.Repo(search_parent_directories=True)
        global run_id
        run_id = str(repo.head.object.hexsha)[:6] + "-" + str(seed)

        mode = "test"
        train, test, linear_feature_columns, dnn_feature_columns = preprocess()

        import os

        os.makedirs(f"submissions/{run_id}", exist_ok=True)

        # train = pd.read_parquet("featured_train.parquet")
        # test = pd.read_parquet("featured_test.parquet")

        fit_and_predict(
            train, test, linear_feature_columns, dnn_feature_columns, mode=mode
        )
