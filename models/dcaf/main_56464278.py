import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from common_denominator import non_null_con_dict
from model import DCAF


def frequency_encoder(
    train: pd.DataFrame,
    val,
    test: pd.DataFrame,
    cols,
    prefix_name: str = "FREQ",
):
    feat_list = []
    for col in tqdm(cols):
        # fe = train[col].value_counts() / len(train)
        fe = np.log1p(len(train) / train[col].value_counts())
        feat_name = f"{prefix_name}-{col}"
        train.loc[:, feat_name] = train[col].map(fe)
        test.loc[:, feat_name] = test[col].map(fe).fillna(0)
        val.loc[:, feat_name] = val[col].map(fe).fillna(0)
        feat_list.append(feat_name)

    return train, val, test, feat_list


def target_encoder(
    train: pd.DataFrame,
    val,
    test: pd.DataFrame,
    cols,
    prefix_name: str = "TE",
    target_col: str = "is_clicked",
    alpha: float = 5.0,
    slice_recent_days: int = None,
):
    cut_day = (
        train.f_1.min()
        if slice_recent_days is None
        else train.f_1.max() - slice_recent_days
    )
    train_2 = train[train.f_1 < cut_day].copy()
    train_1 = train[train.f_1 >= cut_day].copy()
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

        val.loc[:, feat_name] = val[col].map(smooth)
        val.loc[val[feat_name].isna(), feat_name] = global_mean

        test.loc[:, feat_name] = test[col].map(smooth)
        test.loc[test[feat_name].isna(), feat_name] = global_mean

        smooth_dict = smooth.to_dict()

        smooth_dict.setdefault("-1", global_mean)

        te_maps[col] = smooth_dict
        feat_list.append(feat_name)

    return train, val, test, te_maps, feat_list


def seed_everything(seed=42):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()
df = pd.read_parquet("/home/ubuntu/data/df_all.parquet")
df.head()


# categorical variables
# f_2 -> 200개 이하는 모두 동일하게 묶어준다.
# f_3 -> Good.
# f_4 -> 260개 이하는 모두 동일하게 묶어준다.
# f_5 -> Good.
# f_6 -> 260개 이하는 모두 동일하게 묶어준다.
# f_7 -> 넌 나가라~
# f_8, f_9, f_10, f_11 -> GOOD.
# f_12 -> 100개 이하는 모두 동일하게 묶어준다.
# f_13 -> 10개 이하는 모두 동일하게 묶어준다.
# f_14 -> 100개 이하는 모두 동일하게 묶어준다.
# f_15 -> 10개 이하는 모두 동일하게 묶어준다.
# f_16 -> 3000개 이하는 모두 동일하게 묶어준다.
# f_17 -> 100개 이하는 모두 동일하게 묶어준다.
# f_18 -> 10개 이하는 모두 동일하게 묶어준다.
# f_19 -> 100개 이하는 모두 동일하게 묶어준다.
# f_20 -> 200개 이하는 모두 동일하게 묶어준다.
# f_21 -> 100개 이하는 모두 동일하게 묶어준다.
# f_22 -> 100개 이하는 모두 동일하게 묶어준다.
# f_23 -> Good.
# f_24 -> f_23 과 중복. 넌 나가라!
# f_25 -> Good.
# f_26, f_27, f_28, f_29 -> f_23 에 이미 다 있다. 넌 나가라!
# f_30, f_31 -> NaN을 2로 채워준다.
# f_32 -> Good.

for col, threshold in [
    ("f_2", 200),
    ("f_4", 260),
    ("f_6", 200),
    ("f_12", 100),
    ("f_13", 10),
    ("f_14", 100),
    ("f_15", 10),
    ("f_16", 3000),
    ("f_17", 100),
    ("f_18", 10),
    ("f_19", 100),
    ("f_20", 200),
    ("f_21", 100),
    ("f_22", 100),
]:
    # 이거 leaking 아니에요. test data 없어도 이렇게 짤 수 있음.
    cnt = df[col].value_counts()[df[col].value_counts() <= threshold]
    less = cnt[cnt <= threshold].index
    df.loc[df[col].isin(less), col] = 123456789

    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

for col in ["f_3", "f_5", "f_8", "f_9", "f_10", "f_11", "f_23", "f_25", "f_32"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df[["f_74_cat", "f_75_cat", "f_76_cat"]] = (
    df[["f_74", "f_75", "f_76"]] / 0.0385640684536896
).astype(int)
for col in ["f_74_cat", "f_75_cat", "f_76_cat"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df["f_30"] = df["f_30"].fillna(2).astype(int)
df["f_31"] = df["f_31"].fillna(2).astype(int)

df.drop(["f_7", "f_24", "f_26", "f_27", "f_28", "f_29"], axis=1, inplace=True)

cat_features = [
    "f_2",
    "f_3",
    "f_4",
    "f_5",
    "f_6",
    "f_8",
    "f_10",
    "f_12",
    "f_13",
    "f_14",
    "f_15",
    "f_16",
    "f_17",
    "f_18",
    "f_19",
    "f_20",
    "f_21",
    "f_22",
    "f_23",
    "f_25",
    "f_32",
    "f_74_cat",
    "f_75_cat",
    "f_76_cat",
]


# continual variables (f_42 ~ f_79)
for col, val in non_null_con_dict.items():
    df[col] = (df[col] / val).astype(int)


# all min max scaling.
# f_42, f_44 ~ f_50, f_52 ~ f_57, f_60 ~ f_63, f_71 ~ f_79  -> log 취해서
# f_43, f_51, f_58, f_59, f_64 ~ f_70 -> NaN을 0으로 채워준다. f_64는 log 취해서.

f_51_mean = df[df.f_1 != 67].groupby(["f_4"])["f_51"].mean().reset_index()
df["f_51"] = df["f_51"].fillna(
    df[["f_4"]].merge(f_51_mean, how="left", on="f_4")["f_51"]
)
f_67_mean = df[df.f_1 != 67]["f_67"].mean()
df["f_67"] = df["f_67"].fillna(f_67_mean)

for col in [
    "f_42",
    "f_44",
    "f_45",
    "f_46",
    "f_47",
    "f_48",
    "f_49",
    "f_50",
    "f_52",
    "f_53",
    "f_54",
    "f_55",
    "f_56",
    "f_57",
    "f_60",
    "f_61",
    "f_62",
    "f_63",
    "f_64",
    "f_71",
    "f_72",
    "f_73",
    "f_74",
    "f_75",
    "f_76",
    "f_77",
    "f_78",
    "f_79",
]:
    df[col] = np.log1p(df[col])

for col in [
    "f_43",
    "f_51",
    "f_58",
    "f_59",
    "f_64",
    "f_65",
    "f_66",
    "f_67",
    "f_68",
    "f_69",
    "f_70",
]:
    df[col] = df[col].fillna(0)

for col in [f"f_{idx}" for idx in range(42, 80)]:
    MAX = df[df.f_1 != 67][col].max()
    MIN = df[df.f_1 != 67][col].min()
    df[col] = (df[col] - MIN) / (MAX - MIN)

con_features = [
    "f_42",
    "f_43",
    "f_44",
    "f_45",
    "f_46",
    "f_47",
    "f_48",
    "f_49",
    "f_50",
    "f_51",
    "f_52",
    "f_53",
    "f_54",
    "f_55",
    "f_56",
    "f_57",
    "f_58",
    "f_59",
    "f_60",
    "f_61",
    "f_62",
    "f_63",
    "f_64",
    "f_65",
    "f_66",
    "f_67",
    "f_68",
    "f_69",
    "f_70",
    "f_71",
    "f_72",
    "f_73",
    "f_74",
    "f_75",
    "f_76",
    "f_77",
    "f_78",
    "f_79",
    "TE-f_2-is_installed",
    "TE-f_4-is_installed",
    "TE-f_6-is_installed",
    "TE-f_15-is_installed",
    "TE-f_16-is_installed",
    "TE-f_74-is_installed",
    "TE-f_75-is_installed",
    "TE-f_76-is_installed",
    "FREQ-f_2",
    "FREQ-f_4",
    "FREQ-f_6",
    "FREQ-f_15",
    "FREQ-f_16",
    "FREQ-f_74",
    "FREQ-f_75",
    "FREQ-f_76",
]


from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    def __init__(self, data, cat_cols=None, output_col=None):
        self.n = data.shape[0]

        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = 0

        self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = [
            col for col in data.columns if col not in self.cat_cols + [output_col]
        ]

        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.__len__(), 1))

        if self.cat_cols:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.__len__(), 1))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]


for seed in range(50, 100):
    seed_everything(seed)
    asdf = df[df.f_1 != 67].copy()
    # train = df[df.f_1 < 65].copy()
    # val = df[df.f_1.isin([65, 66])].copy()
    test = df[df.f_1 == 67].copy()

    # train, val = train_test_split(asdf, test_size=0.2, random_state=seed)
    train = asdf.copy()
    val = asdf[asdf.f_1.isin([65, 66])].copy()
    del asdf
    import gc

    _ = gc.collect()

    train, val, test, _, feat_list = target_encoder(
        train,
        val,
        test,
        cols=["f_2", "f_4", "f_6", "f_15", "f_16", "f_74", "f_75", "f_76"],
        target_col="is_installed",
        slice_recent_days=7,
        alpha=10,
    )
    train, val, test, feat_list = frequency_encoder(
        train,
        val,
        test,
        cols=["f_2", "f_4", "f_6", "f_15", "f_16", "f_74", "f_75", "f_76"],
    )
    # already added

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    cat_dims = [int(df[col].nunique()) + 1 for col in cat_features]
    emb_dims = [(x, 6) for x in cat_dims]
    no_of_cont = len(con_features)

    model = DCAF(emb_dims, no_of_cont, [256, 256], 1, 0.06, [0.083, 0.083], [1, 1]).to(
        device
    )

    # Define loss function and optimizer

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.00001)

    # Train model

    train_dataloader = DataLoader(
        TabularDataset(
            data=train[cat_features + con_features + ["is_installed"]],
            cat_cols=cat_features,
            output_col="is_installed",
        ),
        batch_size=4096,
        shuffle=True,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        TabularDataset(
            data=val[cat_features + con_features + ["is_installed"]],
            cat_cols=cat_features,
            output_col="is_installed",
        ),
        batch_size=4096,
        shuffle=True,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        TabularDataset(
            data=test[cat_features + con_features + ["is_installed"]],
            cat_cols=cat_features,
            output_col="is_installed",
        ),
        batch_size=4096,
        shuffle=False,
        num_workers=4,
    )

    epochs = 15
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for y, cont_x, cat_x in tqdm(train_dataloader):
            y = y.to(device)
            cont_x = cont_x.to(device)
            cat_x = cat_x.to(device)
            optimizer.zero_grad()
            out = model(cont_x, cat_x)
            loss = criterion(out, y)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = []
            for y, cont_x, cat_x in tqdm(val_dataloader):
                y = y.to(device)
                cont_x = cont_x.to(device)
                cat_x = cat_x.to(device)
                out = model(cont_x, cat_x)
                loss = criterion(out, y)
                val_loss.append(loss.item())
            p = val["is_installed"].mean()
            val_loss = np.mean(val_loss) / -(p * np.log(p) + (1 - p) * np.log(1 - p))
            p = train["is_installed"].mean()
            train_loss = np.mean(train_loss) / -(
                p * np.log(p) + (1 - p) * np.log(1 - p)
            )
            print(
                "Epoch: %d, Train Loss: %.4f, Val Loss: %.4f"
                % (epoch, train_loss, val_loss)
            )

            if epoch >= 2:
                preds = []
                for y, cont_x, cat_x in tqdm(test_dataloader):
                    cont_x = cont_x.to(device)
                    cat_x = cat_x.to(device)
                    out = model(cont_x, cat_x)
                    preds.append(1 / (1 + np.exp(-out.cpu().numpy())))
                preds = np.concatenate(preds)
                asdf = test[["f_0"]].rename(columns={"f_0": "row_id"})
                asdf["is_clicked"] = 0
                asdf["is_installed"] = preds
                import os

                dir = os.path.dirname(__file__).split("/")[-1]
                asdf.to_csv(
                    f"./{dir}-dcaf-{seed}-{epoch}.csv",
                    sep="\t",
                    index=False,
                )
