import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# =========================
# USER SETTINGS
# =========================
EXCEL_PATH = r"C:\Users\jessi\OneDrive\圖片\桌面\5fold\raw data.csv"
TARGET_COL = "Output 1"

CATEGORICAL_COLS = ["特徵值1","特徵值2","特徵值3", "特徵值4", "特徵值7", "特徵值8", "特徵值24"]
NUMERIC_COLS = [
    "特徵值5","特徵值6","特徵值9","特徵值10","特徵值11","特徵值12",
    "特徵值13","特徵值14","特徵值15","特徵值16","特徵值17","特徵值18",
    "特徵值19","特徵值20","特徵值21","特徵值22","特徵值23"
]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

RANDOM_STATE = 36
N_ESTIMATORS = 300
TOP_K_IMPORTANCE = 24
N_SPLITS = 5

LOW_TH = 50.0


def main():
    # 1) Read data
    df = pd.read_csv(EXCEL_PATH, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()

    # 2) 清理：特徵欄位中出現 % 的值（"40%" -> 0.40）
    def to_ratio_series(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip()
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
        has_pct = s.str.contains("%", na=False)
        s_no_pct = s.str.replace("%", "", regex=False).str.strip()
        num = pd.to_numeric(s_no_pct, errors="coerce")
        num.loc[has_pct] = num.loc[has_pct] / 100.0
        return num

    for col in FEATURE_COLS:
        if col in df.columns and df[col].astype(str).str.contains("%", na=False).any():
            df[col] = to_ratio_series(df[col])

    # 3) 數值欄強制轉數字
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ## 4) y：處理括號負號 + 去逗號 + 去單位
    y_str = df[TARGET_COL].astype(str).str.strip()

    # (0.35) → -0.35
    y_str = y_str.str.replace(r"\((.*?)\)", r"-\1", regex=True)

    # 去掉逗號
    y_str = y_str.str.replace(",", "", regex=False)

    # 去掉單位 A（若有）
    y_str = y_str.str.replace("A", "", regex=False)

    # 轉成數字
    y = pd.to_numeric(y_str, errors="coerce")

    # 如果你還是想要 abs：
    y = y.abs()

    bad_y = y.isna().sum()
    if bad_y > 0:
        print(f"⚠️ y 有 {bad_y} 筆無法轉成數字，已自動刪除這些列")
        keep = ~y.isna()
        df = df.loc[keep].copy()
        y = y.loc[keep].copy()

    # 5) X (24 features) + 缺值處理
    #   注意：要在刪 y 後再取 X，避免 index 對不上
    X_raw = df[FEATURE_COLS].copy()
    X_raw[NUMERIC_COLS] = X_raw[NUMERIC_COLS].fillna(X_raw[NUMERIC_COLS].median())
    X_raw[CATEGORICAL_COLS] = X_raw[CATEGORICAL_COLS].fillna("<NA>")

    # ✅ 用 24 特徵組合當 group
    groups = X_raw.fillna("<NA>").astype(str).agg("|".join, axis=1)

    # 6) One-hot categorical features
    X = pd.get_dummies(X_raw, columns=CATEGORICAL_COLS, drop_first=False)

    print("=== Group 5-Fold CV by 24-feature design condition ===")
    print(f"Total samples: {len(df)}")
    print(f"Unique design groups: {groups.nunique()}")
    print(f"X shape after one-hot: {X.shape}")

    # 7) GroupKFold
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_metrics = []
    fold_importance_pct = []

    def base_feature(colname: str) -> str:
        for c in CATEGORICAL_COLS:
            if colname.startswith(c + "_"):
                return c
        return colname

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 同 group 不得交叉
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        assert train_groups.isdisjoint(test_groups), "Group leakage: train/test 有重疊 group！"

        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        r2 = r2_score(y_test, y_pred)

        fold_metrics.append({
            "fold": fold,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "MSE": mse,
            "RMSE": rmse,
            "MAPE_%": mape,
            "R2": r2
        })

        # importance 聚合回 24 features
        fi = pd.Series(model.feature_importances_, index=X.columns)
        fi_agg = fi.groupby(fi.index.map(base_feature)).sum().sort_values(ascending=False)
        fi_pct = fi_agg / fi_agg.sum() * 100
        fold_importance_pct.append(fi_pct.reindex(FEATURE_COLS).fillna(0.0))

        high_ge_80 = fi_pct[fi_pct >= LOW_TH]

        print(f"\n--- Fold {fold} ---")
        print(f"Train samples/groups: {len(train_idx)} / {len(train_groups)}")
        print(f"Test  samples/groups: {len(test_idx)} / {len(test_groups)}")
        print(f"MSE={mse:.6f}  RMSE={rmse:.6f}  MAPE={mape:.2f}%  R2={r2:.6f}")

        print("\nTop 10 Feature Importance (aggregated to 24):")
        for k, v in fi_pct.sort_values(ascending=False).head(10).items():
            print(f"  {k}: {v:.2f}%")

        print(f"\nFeatures >= {LOW_TH:.0f}% (this fold):")
        if len(high_ge_80) == 0:
            print("  None")
        else:
            for k, v in high_ge_80.items():
                print(f"  {k}: {v:.2f}%")

    # 8) Summary
    metrics_df = pd.DataFrame(fold_metrics)
    print("\n===== 5-Fold CV Metrics Summary =====")
    print(metrics_df.to_string(index=False, formatters={
        "MSE":"{:.6f}".format,
        "RMSE":"{:.6f}".format,
        "MAPE_%":"{:.2f}".format,
        "R2":"{:.6f}".format,
    }))

    print("\nAverages:")
    print(f"Avg R2   = {metrics_df['R2'].mean():.6f} ± {metrics_df['R2'].std():.6f}")
    print(f"Avg RMSE = {metrics_df['RMSE'].mean():.6f} ± {metrics_df['RMSE'].std():.6f}")
    print(f"Avg MAPE = {metrics_df['MAPE_%'].mean():.2f}% ± {metrics_df['MAPE_%'].std():.2f}%")

    imp_mat = pd.concat(fold_importance_pct, axis=1)
    imp_mat.columns = [f"fold{f}" for f in range(1, N_SPLITS + 1)]

    imp_summary = pd.DataFrame({
        "feature": imp_mat.index,
        "importance_mean_%": imp_mat.mean(axis=1),
        "importance_std_%": imp_mat.std(axis=1),
        "folds_ge_80": (imp_mat >= LOW_TH).sum(axis=1)
    }).sort_values("importance_mean_%", ascending=False)

    print("\n===== Mean Feature Importance Across 5 Folds (aggregated to 24) =====")
    print(imp_summary.head(TOP_K_IMPORTANCE).to_string(index=False, formatters={
        "importance_mean_%":"{:.2f}".format,
        "importance_std_%":"{:.2f}".format,
    }))


    print(f"\n===== Features with mean importance >= {LOW_TH:.0f}% =====")
    mean_ge_80 = imp_summary[imp_summary["importance_mean_%"] >= LOW_TH]
    print("None" if len(mean_ge_80) == 0 else mean_ge_80.to_string(index=False, formatters={
        "importance_mean_%":"{:.2f}".format,
        "importance_std_%":"{:.2f}".format,
    }))

    # 9) Save
    metrics_df.to_csv("cv5_metrics.csv", index=False, encoding="utf-8-sig")
    imp_summary.to_csv("cv5_feature_importance_agg24_summary.csv", index=False, encoding="utf-8-sig")
    imp_mat.reset_index().rename(columns={"index": "feature"}).to_csv(
        "cv5_feature_importance_agg24_eachfold.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved:")
    print(" - cv5_metrics.csv")
    print(" - cv5_feature_importance_agg24_summary.csv")
    print(" - cv5_feature_importance_agg24_eachfold.csv")
    
        # ==========================================================
    # 10) 分析：importance >= 50% 的特徵 與 Y 的對應關係（含趨勢線/方程式/r）
    # ==========================================================
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    print(f"\n===== Inspect Features with importance >= {LOW_TH:.0f}% =====")

    important_feats = imp_summary[imp_summary["importance_mean_%"] >= LOW_TH]["feature"].tolist()
    if len(important_feats) == 0:
        print(f"No feature >= {LOW_TH:.0f}% importance.")
        return
    else:
        print("Features:", important_feats)

    for feat in important_feats:
        print(f"\n--- Inspecting {feat} ---")

        # 取數值（注意：feat 可能是類別欄，若轉不成數字會全 NaN）
        x = pd.to_numeric(df[feat], errors="coerce")
        keep = ~x.isna()
        x = x[keep]
        y_plot = y[keep]

        if len(x) < 10:
            print("Not enough numeric samples to plot/regress.")
            continue

        # ===== Pearson r =====
        pearson = np.corrcoef(x, y_plot)[0, 1]

        # ===== 線性回歸 y = a x + b =====
        X1 = x.values.reshape(-1, 1)
        lr = LinearRegression().fit(X1, y_plot.values)
        y_hat = lr.predict(X1)
        a = float(lr.coef_[0])
        b = float(lr.intercept_)
        r2_lin = r2_score(y_plot.values, y_hat)

        print("Pearson r =", round(pearson, 6))
        print("Linear R2  =", round(r2_lin, 6))
        print(f"Linear eq  : y = {a:.6g} * x + {b:.6g}")

        # ===== 分箱趨勢（binned trend）=====
        tmp = pd.DataFrame({"x": x, "y": y_plot})
        tmp["bin"] = pd.qcut(tmp["x"], q=30, duplicates="drop")
        g = tmp.groupby("bin").agg(
            x_mean=("x", "mean"),
            y_mean=("y", "mean")
        ).reset_index()

        # ===== 畫圖：scatter + 分箱線 + 線性趨勢線 =====
        plt.figure()
        plt.scatter(tmp["x"], tmp["y"], s=6, alpha=0.25)          # scatter
        plt.plot(g["x_mean"], g["y_mean"])                        # binned trend

        # 線性趨勢線：用 x 範圍畫一條直線
        x_line = np.linspace(tmp["x"].min(), tmp["x"].max(), 200)
        y_line = a * x_line + b
        plt.plot(x_line, y_line)                                  # linear trend line

        plt.xlabel(feat)
        plt.ylabel(TARGET_COL)
        plt.title(f"{feat} vs {TARGET_COL}")

        # 在圖上寫：r、R²、方程式
        textstr = (
            f"Pearson r = {pearson:.3f}\n"
            f"Linear $R^2$ = {r2_lin:.3f}\n"
            f"y = {a:.3g}x + {b:.3g}"
        )
        plt.text(
            0.05, 0.95,
            textstr,
            transform=plt.gca().transAxes,
            verticalalignment="top"
        )

        plt.show()

if __name__ == "__main__":
    main()