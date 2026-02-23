import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# 1. 讀取與基本清理
# ---------------------------------------------------------
df = pd.read_excel(r"C:\Users\Joanne\Desktop\0223test\0223data.xlsx")

features_x = [
    'wafer fab', 'wafer process', 'PKG type', 'PKG tech.', 'Pin amount', 'PIN Pitch', 
    'PIN function(RF/IO/VDD/GND)', 'Stress(+/-)',  'Stressed Voltage',  'PKG.Area', 'PKG. thickness', 'PKG.volume', 'die amount', 'Die width', 
    'Die long', 'Die Area', 'Die area coverage (%)', 'Pin/EPad/Ball width', 
    '(BD)Pin/EPad/Ball long', '(BD)Pin/EPad/Ball area', '(BD)Pin/EPad/Ball coverage%', 
    'PKG. Cover Material(Insulator or Conductor)'
]

cat_features = [
    'wafer fab', 'wafer process', 'PKG type', 'PKG tech.', 'PIN function(RF/IO/VDD/GND)',
    '(BD)Pin/EPad/Ball long', 'PKG. Cover Material(Insulator or Conductor)'
]

# 需求 A：電壓（目標值）取絕對值
target_y = 'I(A)'
df[target_y] = pd.to_numeric(df[target_y], errors='coerce').abs() 
df = df.dropna(subset=[target_y]) # 剔除無效目標值

# 預先處理特徵缺失值，避免 group_id 出現 NaN
for col in features_x:
    if col in cat_features:
        df[col] = df[col].astype(str).replace('nan', 'Unknown').fillna("Unknown")
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# 需求 B：定義組別 (特徵完全相同者歸為一組)
df['group_id'] = df.groupby(features_x).ngroup()

# ---------------------------------------------------------
# 2. 資料提取與編碼
# ---------------------------------------------------------
X = df[features_x].copy()
y = df[target_y].copy()
groups = df['group_id'].copy()

for col in cat_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# ---------------------------------------------------------
# 3. 確保相同特徵不跨集 (Group-based Split)
# ---------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

# ---------------------------------------------------------
# 4. 5-Fold Group Cross Validation
# ---------------------------------------------------------
gkf = GroupKFold(n_splits=5)
cv_results = []

print(f"\n{'Fold':^6} | {'RMSE':^10} | {'R2':^10} | {'MAPE (%)':^10}")
print("-" * 55)

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_full, y_train_full, groups=groups_train)):
    X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
    
    # 新版 XGBoost 將 early_stopping_rounds 放進初始化
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        early_stopping_rounds=50
    )
    
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    r2 = r2_score(y_val, val_preds)
    mape = mean_absolute_percentage_error(y_val, val_preds)
    
    cv_results.append({'rmse': rmse, 'r2': r2, 'mape': mape})
    print(f"Fold {fold+1:^2} | {rmse:^10.4f} | {r2:^10.4f} | {mape*100:^10.2f}%")

# ---------------------------------------------------------
# 5. 最終測試結果與特徵貢獻度
# ---------------------------------------------------------
test_preds = model.predict(X_test)
print("-" * 55)
print(f"平均 CV R2  : {np.mean([r['r2'] for r in cv_results]):.4f}")
print(f"最終 Test R2: {r2_score(y_test, test_preds):.4f}")
print(f"最終 Test MAPE: {mean_absolute_percentage_error(y_test, test_preds)*100:.2f}%")

# 特徵貢獻度排名
importances = model.feature_importances_
fe_df = pd.DataFrame({'Feature': features_x, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\n[特徵貢獻度 Top 10]")
print(fe_df.head(10))

# 繪圖
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=fe_df)
plt.title('Feature Importance (Absolute Voltage/Current)')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7. 關鍵特徵趨勢分析：Stressed Voltage vs. I(A)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# 畫出原始數據點
sns.scatterplot(x=df['Stressed Voltage'], y=df[target_y], alpha=0.5, label='Actual Data')

# 畫出趨勢線 (使用線性或非線性回歸趨勢)
sns.regplot(x=df['Stressed Voltage'], y=df[target_y], scatter=False, color='red', label='Trend Line')

plt.title('Relationship: Stressed Voltage vs. Current I(A)', fontsize=14)
plt.xlabel('Stressed Voltage (V)', fontsize=12)
plt.ylabel('Current I(A) (Absolute Value)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.show()