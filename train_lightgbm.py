import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

RUSH = [7, 8, 9, 16, 17, 18, 19]
BASE = ["distance_km", "log_dist", "hav_km", "hour", "weekday", "is_rush",
        "is_weekend", "pickup_lat", "pickup_long", "dropoff_lat", "dropoff_long"]
TE = [("puk", 40), ("dok", 40), ("route", 30)]
SEEDS = [42, 7, 2024]

PARAMS = dict(objective="regression_l1", metric="mae", learning_rate=0.06,
              num_leaves=127, min_child_samples=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
              force_row_wise=True, verbose=-1)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = np.pi / 180
    a = (0.5 - np.cos((lat2 - lat1) * p) / 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def add_features(df):
    df = df.copy()
    df["hav_km"] = haversine(df.pickup_lat, df.pickup_long, df.dropoff_lat, df.dropoff_long)
    df["log_dist"] = np.log1p(df.distance_km)
    df["is_rush"] = df.hour.isin(RUSH).astype(int)
    df["is_weekend"] = (df.weekday >= 5).astype(int)
    df["puk"] = df.pickup_lat.round(4).astype(str) + "_" + df.pickup_long.round(4).astype(str)
    df["dok"] = df.dropoff_lat.round(4).astype(str) + "_" + df.dropoff_long.round(4).astype(str)
    df["route"] = df.puk + "|" + df.dok
    return df


def smooth_encoding(keys, target, smoothing):
    prior = target.mean()
    agg = target.groupby(keys).agg(["mean", "count"])
    enc = (agg["mean"] * agg["count"] + prior * smoothing) / (agg["count"] + smoothing)
    return enc, prior


def build_matrix(train, test, y):
    Xtr = pd.DataFrame({c: train[c].values for c in BASE})
    Xte = pd.DataFrame({c: test[c].values for c in BASE})
    folds = list(KFold(5, shuffle=True, random_state=42).split(train))
    for col, sm in TE:
        oof = np.zeros(len(train))
        for tr, va in folds:
            enc, prior = smooth_encoding(train[col].iloc[tr], y.iloc[tr], sm)
            oof[va] = train[col].iloc[va].map(enc).fillna(prior).values
        Xtr["te_" + col] = oof
        enc, prior = smooth_encoding(train[col], y, sm)
        Xte["te_" + col] = test[col].map(enc).fillna(prior).values
    Xtr["t"] = np.linspace(0, 1, len(train))
    Xte["t"] = 1.0
    return Xtr, Xte


def main():
    train = add_features(pd.read_csv("data/train.csv"))
    test = add_features(pd.read_csv("data/test.csv"))
    y = train["trip_time"]
    Xtr, Xte = build_matrix(train, test, y)

    cut = int(len(train) * 0.9)
    val = lgb.train(dict(PARAMS, seed=42), lgb.Dataset(Xtr.iloc[:cut], y.iloc[:cut]),
                    num_boost_round=4000,
                    valid_sets=[lgb.Dataset(Xtr.iloc[cut:], y.iloc[cut:])],
                    callbacks=[lgb.early_stopping(120), lgb.log_evaluation(200)])
    rounds = int(val.best_iteration / 0.9)

    pred = np.zeros(len(test))
    for s in SEEDS:
        model = lgb.train(dict(PARAMS, seed=s), lgb.Dataset(Xtr, y), num_boost_round=rounds)
        pred += model.predict(Xte) / len(SEEDS)
    pred = np.clip(pred, 0, 96)
    pd.DataFrame({"id": test["id"], "trip_time": pred}).to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
