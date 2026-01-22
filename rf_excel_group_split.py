
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

"""
Random Forest regression with:
- Excel input
- Categorical columns -> one-hot (pd.get_dummies)
- Group split by a group column (e.g., chip_id) so the same chip never appears in both train & test
- Feature importance + percentage

HOW TO RUN (PowerShell / CMD):
    python rf_excel_group_split.py

You can edit the parameters in the "USER SETTINGS" section below.
"""

# =========================
# USER SETTINGS (edit here)
# =========================
EXCEL_PATH = "C:\\Users\\jessi\\Downloads\\CDM_group_by_chip_ML_ready.csv"          # <-- 改成你的 Excel 路徑
TARGET_COL = "y_Ipeak"        # <-- 改成你的 y 欄位（例如 I_peak）
GROUP_COL  = "chip_id"            # <-- 用來做 group split 的欄位（不會當特徵）
CATEGORICAL_COLS = ["特徵值1", "特徵值3", "特徵值4","特徵值7", "特徵值8", "特徵值24"]  # <-- 改成你要 one-hot 的類別欄位清單

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 300

# (可選) 只列出前幾名重要特徵；None 表示全部列出
TOP_K_IMPORTANCE = 20


def main():
    # 1) 讀取 Excel
    df = pd.read_excel(EXCEL_PATH)

    # 基本檢查
    for col in [TARGET_COL, GROUP_COL]:
        if col not in df.columns:
            raise ValueError(f"找不到欄位 '{col}'。目前欄位有：{list(df.columns)}")

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            raise ValueError(f"你指定的類別欄位 '{col}' 不在 Excel 中。")

    # 2) 取出 y 與 group（group 只拿來切資料，不會進入 X）
    y = df[TARGET_COL]
    groups = df[GROUP_COL].astype(str)

    # 3) 組 X：把 target 與 group 拿掉，其餘都是特徵
    X_raw = df.drop(columns=[TARGET_COL, GROUP_COL])

    # 4) one-hot 類別欄位
    X = pd.get_dummies(X_raw, columns=CATEGORICAL_COLS, drop_first=False)

    # 5) Group split（同一個 chip 不會同時出現在 train/test）
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print("=== Group Split by chip (chip_id NOT used as feature) ===")
    print(f"Train samples: {len(train_idx):,}  Test samples: {len(test_idx):,}")
    print(f"#chips train/test: {groups.iloc[train_idx].nunique()}/{groups.iloc[test_idx].nunique()}")

    # 6) 訓練 Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 7) 預測與評估
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100.0
    r2 = r2_score(y_test, y_pred)

    print("\n=== Metrics (Regression) ===")
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R^2  : {r2:.6f}")

    # 8) Feature Importance（貢獻度）+ Percentage
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fi_pct = (fi / fi.sum()) * 100.0
    out = pd.DataFrame({
        "feature": fi.index,
        "importance": fi.values,
        "percentage_%": fi_pct.values
    })

    out_show = out.head(TOP_K_IMPORTANCE) if TOP_K_IMPORTANCE is not None else out

    print("\n=== Feature Importance (Top) ===")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 10, "display.width", 160):
        print(out_show.to_string(index=False, formatters={
            "importance": "{:.6f}".format,
            "percentage_%": "{:.2f}".format
        }))

    # 9) 存檔
    out.to_csv("rf_feature_importance_with_pct.csv", index=False, encoding="utf-8-sig")

    pred_df = df.iloc[test_idx].copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = y_pred
    pred_df.to_csv("rf_test_predictions.csv", index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print(" - rf_feature_importance_with_pct.csv")
    print(" - rf_test_predictions.csv")


if __name__ == "__main__":
    main()
