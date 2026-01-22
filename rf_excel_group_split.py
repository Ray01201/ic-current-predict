import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# =========================
# USER SETTINGS
# =========================
EXCEL_PATH = r"C:\Users\jessi\OneDrive\圖片\桌面\ml\CDM_group_by_chip_ML_ready.csv"
TARGET_COL = "y_Ipeak_abs"

CATEGORICAL_COLS = ["特徵值1", "特徵值3", "特徵值4", "特徵值7", "特徵值8", "特徵值24"]
NUMERIC_COLS = [
    "特徵值2","特徵值5","特徵值6","特徵值9","特徵值10","特徵值11","特徵值12",
    "特徵值13","特徵值14","特徵值15","特徵值16","特徵值17","特徵值18",
    "特徵值19","特徵值20","特徵值21","特徵值22","特徵值23"
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

TEST_SIZE = 0.2
RANDOM_STATE = 36
N_ESTIMATORS = 500
TOP_K_IMPORTANCE = 24


def main():
    # 1) Read data
    df = pd.read_csv(EXCEL_PATH, encoding="utf-8-sig")

    # 2) y
    y = df[TARGET_COL].abs()

    # 3) X (only selected 24 features)
    X_raw = df[FEATURE_COLS].copy()

    # ✅ 關鍵：用 24 特徵「組合」當 group
    groups = X_raw.astype(str).agg("|".join, axis=1)

    # 4) One-hot categorical features
    X = pd.get_dummies(X_raw, columns=CATEGORICAL_COLS, drop_first=False)

    # 5) Group split by feature-combination
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print("=== Group Split by 24-feature design condition ===")
    print(f"Train samples: {len(train_idx)}  Test samples: {len(test_idx)}")
    print(f"#unique design conditions train/test: "
          f"{groups.iloc[train_idx].nunique()} / {groups.iloc[test_idx].nunique()}")

    # 6) Train RF
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 7) Predict & metrics
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)

    print("\n=== Metrics ===")
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R^2  : {r2:.6f}")

    # 8) Feature importance (aggregate back to 24 features)
    fi = pd.Series(model.feature_importances_, index=X.columns)

    def base_feature(col):
        for c in CATEGORICAL_COLS:
            if col.startswith(c + "_"):
                return c
        return col

    fi_agg = fi.groupby(fi.index.map(base_feature)).sum().sort_values(ascending=False)
    fi_pct = fi_agg / fi_agg.sum() * 100

    out = pd.DataFrame({
        "feature": fi_agg.index,
        "importance": fi_agg.values,
        "percentage_%": fi_pct.values
    })

    print("\n=== Feature Importance (24 features) ===")
    print(out.head(TOP_K_IMPORTANCE).to_string(index=False,
          formatters={"importance":"{:.6f}".format,"percentage_%":"{:.2f}".format}))

    out.to_csv("rf_feature_importance_agg24.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
