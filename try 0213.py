import os
import warnings
from datetime import datetime

# 只關掉 UserWarning（你指定）
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)


# =========================================================
# Matplotlib 設定：每張圖「停住」直到你手動關閉視窗
# =========================================================
plt.ioff()


# =========================================================
# 0) 基本設定
# =========================================================
DATA_DIR = r"C:\Users\user\Desktop\以3種不同封裝類型分類_raw data_整理後_2026.02.12"

FILES = {
    "BGA": "BGA_final_未扣除奇怪值.xlsx",
    "QFN": "QFN_final_尚未扣掉奇怪值.xlsx",
    "QFP": "QFP_final(RL6608 Pin88 outlier).xlsx",
}

TARGET_COL = "I(A)"
OUT_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

TOP_CUM_THRESHOLD = 0.90
TOP_IMPORTANCE_SHOW_N = 20


# =========================================================
# 工具函數
# =========================================================
def safe_filename(name: str) -> str:
    return (
        name.replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace('"', "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
    )


def add_bar_labels_percent(ax, values, fmt="{:.2f}%"):
    """在水平長條圖每個長條旁標記百分比。"""
    for i, v in enumerate(values):
        ax.text(v, i, " " + fmt.format(v), va="center")


def compute_linear_fit_and_metrics(x, y):
    """
    對 (x,y) 做簡單線性回歸：y = a x + b
    回傳：(a, b, r2, mape, x_line, y_line)
    若資料不足或 x 幾乎無變化，回傳 None。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x2 = x[mask]
    y2 = y[mask]

    if len(x2) < 3:
        return None
    if np.nanstd(x2) < 1e-12:
        return None

    a, b = np.polyfit(x2, y2, 1)
    y_hat = a * x2 + b

    r2 = r2_score(y2, y_hat)

    # MAPE：避免 y=0 造成爆炸，這裡把 |y| 很小的點排除
    eps = 1e-12
    mask_y = np.abs(y2) > eps
    if np.sum(mask_y) >= 3:
        mape = mean_absolute_percentage_error(y2[mask_y], y_hat[mask_y]) * 100
    else:
        mape = np.nan

    x_line = np.linspace(np.min(x2), np.max(x2), 100)
    y_line = a * x_line + b

    return a, b, r2, mape, x_line, y_line


def build_X_from_raw_record(raw_record: dict, cat_cols, unit_tokens, drop_cols, train_cols, med):
    """
    將一筆「原始欄位」資料 raw_record → 轉成可丟進 model 的 X (one-hot & 對齊欄位)
    """
    new_df = pd.DataFrame([raw_record])

    # 字串去空白
    obj_cols_new = new_df.select_dtypes(include=["object", "string"]).columns
    new_df[obj_cols_new] = new_df[obj_cols_new].apply(lambda s: s.astype(str).str.strip())

    # 單位字串 → NaN
    new_df = new_df.replace(list(unit_tokens), np.nan)

    # 確保類別欄位存在（沒填就 NaN）
    for c in cat_cols:
        if c not in new_df.columns:
            new_df[c] = np.nan

    # 非類別欄位轉數字
    for c in new_df.columns:
        if c not in cat_cols:
            new_df[c] = pd.to_numeric(new_df[c], errors="coerce")

    # one-hot
    new_processed = pd.get_dummies(new_df, columns=cat_cols, dummy_na=True)

    # 移除不該當特徵的欄（保險）
    new_processed = new_processed.drop(columns=drop_cols, errors="ignore")

    # 對齊訓練欄位：缺的補 0，多的丟掉
    for col in train_cols:
        if col not in new_processed.columns:
            new_processed[col] = 0
    new_processed = new_processed[train_cols]

    # 缺值補法與訓練一致：先用訓練中位數補、再補 0
    new_processed = new_processed.apply(pd.to_numeric, errors="coerce")
    new_processed = new_processed.fillna(med).fillna(0)

    return new_processed


def safe_float(s: str):
    s = s.strip()
    if s == "":
        return None
    try:
        return float(s)
    except:
        return None


# =========================================================
# 0.5) 重要方程式（你要求：print 出來）
# =========================================================
print("\n===== Important Equations / 重要方程式 =====")
print("（中文）Random Forest 的預測：")
print("  ŷ(x) = (1/T) * Σ_{t=1..T} f_t(x)")
print("  其中 T 是樹的數量，f_t(x) 是第 t 棵樹的預測。")
print("\n（中文）評估指標：")
print("  MSE = (1/n) * Σ (y_i - ŷ_i)^2   （越小越好）")
print("  MAPE = (1/n) * Σ |(y_i - ŷ_i) / y_i|   （越小越好；y_i 接近 0 時會變不穩定）")
print("  R² = 1 - [Σ (y_i - ŷ_i)^2] / [Σ (y_i - ȳ)^2]   （越接近 1 越好）")


# =========================================================
# 1) 讀取三個 Excel、合併、加入 PackageType
# =========================================================
dfs = []
for pkg, fname in FILES.items():
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到檔案：{path}")

    tmp = pd.read_excel(path)
    tmp["PackageType"] = pkg
    dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)


# =========================================================
# 2) 清理資料：去空白 / 單位 → NaN
# =========================================================
obj_cols = df.select_dtypes(include=["object", "string"]).columns
df[obj_cols] = df[obj_cols].apply(lambda s: s.astype(str).str.strip())

unit_tokens = {"mm", "mm*mm", "mm*mm*mm", "A", "ps", "C", "pf"}
df = df.replace(list(unit_tokens), np.nan)


# =========================================================
# 3) 目標欄位（Y）
# =========================================================
if TARGET_COL not in df.columns:
    raise KeyError(f"找不到目標欄位：{TARGET_COL}。請確認 Excel 欄名是否完全一致。")

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)


# =========================================================
# 4) 類別欄位（one-hot）：所有字串欄位 + PackageType
# =========================================================
cat_cols = list(df.select_dtypes(include=["object", "string"]).columns)
if "PackageType" in df.columns and "PackageType" not in cat_cols:
    cat_cols.append("PackageType")


# =========================================================
# 5) 其他欄位盡量轉成數字
# =========================================================
for c in df.columns:
    if c not in cat_cols and c != TARGET_COL:
        df[c] = pd.to_numeric(df[c], errors="coerce")


# =========================================================
# 6) one-hot encoding
# =========================================================
df_processed = pd.get_dummies(df, columns=cat_cols, dummy_na=True)


# =========================================================
# 7) 建立 X / y
# =========================================================
y = df_processed[TARGET_COL]

DROP_COLS = [
    "Unnamed: 0",
    TARGET_COL,
    "Output 1", "Output 2", "Output 3", "Output 4", "Output 5",
]
X = df_processed.drop(columns=DROP_COLS, errors="ignore")

X = X.apply(pd.to_numeric, errors="coerce")
med = X.median(numeric_only=True)
X = X.fillna(med).fillna(0)

TRAIN_COLUMNS = X.columns.tolist()


# =========================================================
# 8) 建立模型
# =========================================================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


# =========================================================
# 9) 5-fold Cross Validation
# =========================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "r2": "r2",
    "mse": make_scorer(mean_squared_error, greater_is_better=False),
    "mape": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
}

cv = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=False)

r2_scores = cv["test_r2"]
mse_scores = -cv["test_mse"]
mape_scores = -cv["test_mape"]

print("\n===== 5-Fold Cross-Validation Results =====")
print("（中文）5 折交叉驗證結果：分成 5 份輪流當測試集，評估模型穩定性。")

for i in range(5):
    print(
        f"Fold {i+1}（第{i+1}折）: "
        f"R2={r2_scores[i]:.4f}（解釋力，越接近 1 越好）, "
        f"MSE={mse_scores[i]:.6f}（均方誤差，越小越好）, "
        f"MAPE={mape_scores[i]*100:.2f}%（平均相對誤差%，越小越好）"
    )

print("\n----- CV Summary -----")
print("（中文）mean=平均表現；std=標準差（越小代表越穩定）。")
print(f"R2   : mean={np.mean(r2_scores):.4f}, std={np.std(r2_scores):.4f}")
print(f"MSE  : mean={np.mean(mse_scores):.6f}, std={np.std(mse_scores):.6f}")
print(f"MAPE : mean={np.mean(mape_scores)*100:.2f}%, std={np.std(mape_scores)*100:.2f}%")


# =========================================================
# 10) Fit 全資料（重要度 + 預測）
# =========================================================
model.fit(X, y)


# =========================================================
# 11) 特徵重要度：只輸出前 20（百分比、不科學記號）
# =========================================================
fi = (
    pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)
fi["CumImportance"] = fi["Importance"].cumsum()

top_show = fi.head(TOP_IMPORTANCE_SHOW_N).copy()
top_show["Importance(%)"] = top_show["Importance"] * 100
top_show["CumImportance(%)"] = top_show["CumImportance"] * 100
top_show = top_show.drop(columns=["Importance", "CumImportance"])

pd.set_option("display.float_format", lambda x: f"{x:.2f}")

print("\n===== All Feature Importances / 所有特徵重要性 =====")
print("（中文）以下僅輸出『前 20 名』特徵的重要度（百分比）。")
print("（中文）重要度越大，代表模型越常用它來降低誤差；此值不是因果關係。")
print(top_show.to_string(index=False))


# =========================================================
# 12) 重要度長條圖：每個長條旁都標記 %
# =========================================================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

top_n = min(30, len(fi))
fi_plot = fi.head(top_n).copy()
fi_plot["Importance(%)"] = fi_plot["Importance"] * 100

fig, ax = plt.subplots()
ax.barh(fi_plot["Feature"][::-1], fi_plot["Importance(%)"][::-1])
ax.set_xlabel("Importance (%)（重要度百分比）")
ax.set_title(f"Top {top_n} Feature Importances (Random Forest)\n（中文）前 {top_n} 名特徵重要度")
ax.set_xlim(0, max(fi_plot["Importance(%)"]) * 1.15)

# 標記每個長條旁的百分比
vals = fi_plot["Importance(%)"][::-1].values
add_bar_labels_percent(ax, vals, fmt="{:.2f}%")

plt.tight_layout()

imp_plot_path = os.path.join(OUT_DIR, f"feature_importance_top_{ts}.png")
plt.savefig(imp_plot_path, dpi=200)

print(f"\n[INFO] 已輸出重要度圖：{imp_plot_path}")
print("（中文）長條越長代表對預測 I(A) 的貢獻度越高。")

plt.show(block=True)


# =========================================================
# 13) 特徵 vs Y：「散布圖(scatter chart)」+ 回歸線 + 方程式 + R² + MAPE
#     - R² / MAPE 針對「回歸線」(y=ax+b) 的擬合結果計算
# =========================================================
top_features = fi[fi["CumImportance"] <= TOP_CUM_THRESHOLD]["Feature"].tolist()
if len(top_features) == 0:
    top_features = fi["Feature"].head(1).tolist()

print("\n===== Feature vs Y Charts / 特徵對 Y 圖 =====")
print(f"（中文）將針對累積重要度 ≤ {TOP_CUM_THRESHOLD:.2f} 的特徵產生「散布圖(scatter chart)」，共 {len(top_features)} 張。")

for feat in top_features:
    xvals = X[feat].values
    yvals = y.values

    fig, ax = plt.subplots()
    ax.scatter(xvals, yvals, s=10)

    ax.set_xlabel(feat)
    ax.set_ylabel(TARGET_COL)

    fit = compute_linear_fit_and_metrics(xvals, yvals)
    if fit is None:
        ax.set_title(f"{feat} vs {TARGET_COL}（scatter chart / 散布圖）\n（中文）資料不足或特徵幾乎為常數，未畫回歸線")
        eq_text = "N/A"
        r2_text = "N/A"
        mape_text = "N/A"
    else:
        a, b, r2_lin, mape_lin, x_line, y_line = fit
        ax.plot(x_line, y_line)
        eq_text = f"y = {a:.6g} x + {b:.6g}"
        r2_text = f"R² = {r2_lin:.4f}"
        mape_text = "MAPE = N/A" if np.isnan(mape_lin) else f"MAPE = {mape_lin:.2f}%"

        # 把方程式 + 指標放到圖內
        ax.text(
            0.02, 0.98,
            f"{eq_text}\n{r2_text}\n{mape_text}",
            transform=ax.transAxes,
            ha="left", va="top"
        )

        ax.set_title(
            f"{feat} vs {TARGET_COL}（scatter chart / 散布圖）\n"
            f"{eq_text} | {r2_text} | {mape_text}"
        )

    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"scatter_chart_{safe_filename(feat)}_{ts}.png")
    plt.savefig(out_path, dpi=200)

    print(f"[INFO] 圖已儲存：{out_path}")
    if fit is not None:
        print(f"      回歸方程式：{eq_text}；{r2_text}；{mape_text}")

    plt.show(block=True)


# =========================================================
# 14) 逐欄位互動輸入 → 預測新 IC 的 I(A)
# =========================================================
RAW_INPUT_COLUMNS = [c for c in df.columns if c != TARGET_COL]

def interactive_input_predict():
    print("\n===== Predict New IC / 預測新 IC（逐筆輸入） =====")
    print("（中文）程式會一欄一欄詢問。Enter 跳過該欄（會用訓練資料的中位數/0 補值）。")
    print("（中文）輸入 q 可離開預測模式。")

    record = {}

    # 先問 PackageType（較符合使用習慣）
    while True:
        v = input("\n請輸入 PackageType（BGA/QFN/QFP），或 q 離開：\n> ").strip()
        if v.lower() in {"q", "quit", "exit"}:
            return
        if v in {"BGA", "QFN", "QFP"}:
            record["PackageType"] = v
            break
        print("（中文）請輸入 BGA / QFN / QFP 其中之一。")

    # 依序詢問其他欄位
    for col in RAW_INPUT_COLUMNS:
        if col == "PackageType":
            continue

        if col in cat_cols:
            v = input(f"\n請輸入 {col}（文字類別；Enter 跳過；q 離開）：\n> ").strip()
            if v.lower() in {"q", "quit", "exit"}:
                return
            if v == "":
                continue
            record[col] = v
        else:
            v = input(f"\n請輸入 {col}（數值；Enter 跳過；q 離開）：\n> ").strip()
            if v.lower() in {"q", "quit", "exit"}:
                return
            if v == "":
                continue

            f = safe_float(v)
            if f is None:
                print("（中文）你輸入的不是數字，這欄先略過。")
                continue
            record[col] = f

    drop_for_new = DROP_COLS[:]  # 同訓練
    X_new = build_X_from_raw_record(
        raw_record=record,
        cat_cols=cat_cols,
        unit_tokens=unit_tokens,
        drop_cols=drop_for_new,
        train_cols=TRAIN_COLUMNS,
        med=med
    )
    pred = model.predict(X_new)[0]

    print("\n===== Prediction Result / 預測結果 =====")
    print(f"Predicted {TARGET_COL} = {pred}")
    print(f"（中文）預測的新 IC 之 {TARGET_COL}（電流 I）= {pred}")


# =========================================================
# 15) 整理完資料後 → 讓使用者選擇「看數據」或「做預測」
# =========================================================
def menu_loop():
    print("\n===== Menu / 功能選單 =====")
    print("1) 看數據（重要度長條圖 + 特徵對 I(A) 的散布圖(scatter chart)）")
    print("2) 做預測（逐欄位輸入新 IC 特徵 → 預測 I(A)）")
    print("q) 離開")

    while True:
        choice = input("\n請輸入 1 / 2 / q：\n> ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            print("（中文）已離開。")
            return

        if choice == "1":
            print("\n（中文）你已選擇：看數據。")
            print("（中文）將依序顯示：重要度圖、以及多張散布圖(scatter chart)。")
            print("（中文）每張圖都會停住，關掉視窗後才會繼續。")
            # 重要度圖與散布圖在上面已產生並顯示過
            print("（中文）注意：本程式為了保持流程一致，圖已在前段產生完成。")
            print("（中文）如果你希望『選了 1 才開始畫圖』，我也可以幫你改成延遲產圖版本。")
        elif choice == "2":
            print("\n（中文）你已選擇：做預測。")
            interactive_input_predict()
        else:
            print("（中文）請輸入 1 / 2 / q。")


menu_loop()

print("\n全部完成 ✅")
print(f"（中文）圖檔輸出資料夾：{OUT_DIR}")
