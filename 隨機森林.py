import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # 注意：預測數值要用 Regressor(回歸?
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_percentage_error
import matplotlib.pyplot as plt

# 1. 讀取 Excel 檔案
df = pd.read_excel(r"C:\\Users\Asus\Downloads\\merged_file.xlsx")

# 2. 處理類別型欄位 (Categorical Data)
#將文字編碼，轉為 0 與 1
# 假設 'device_type' 和 'location' 是類別欄位
df_processed = pd.get_dummies(df, columns=['特徵值1', '特徵值2','特徵值3','特徵值4','特徵值5','特徵值7','特徵值8','特徵值15','特徵值24'])
#df_processed = pd.get_dummies(df, columns=['特徵值8'])

# 3. 定義特徵 (X) 與目標 (y)
# 假設你要預測的目標欄位名稱是 'max_current'

#X = df_processed.drop(['Unnamed: 0','Output 1','Output 2','Output 3','Output 4','Output 5','特徵值1', '特徵值2','特徵值3','特徵值4','特徵值5','特徵值6','特徵值7','特徵值10','特徵值11','特徵值12', '特徵值13','特徵值14','特徵值15','特徵值16','特徵值17','特徵值18','特徵值19','特徵值20','特徵值21','特徵值22','特徵值23','特徵值24'], axis=1) # 除了目標以外的所有欄位都是特徵          # 特徵：特徵值8 和 特徵值9
X = df_processed.drop(['Unnamed: 0','Output 1','Output 2','Output 3','Output 4','Output 5','特徵值10', '特徵值11'], axis=1) # 除了目標以外的所有欄位都是特徵          # 特徵：特徵值8 和 特徵值9
y = df_processed['Output 1']              # 目標：最大電流

# 4. 切分資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 5. 建立並訓練「隨機森林迴歸模型」
#model = RandomForestRegressor(n_estimators=100, random_state=42)
#model.fit(X_train, y_train)

# 5. 建立並訓練「隨機森林迴歸模型」
# 透過 max_depth 來限制深度，並搭配其他參數來控制模型規模
model = RandomForestRegressor(
    n_estimators=100,      # 森林中樹的數量
    max_depth=50,           # 【重點】限制每棵樹的最大深度。如果設為 None，樹會無限生長直到葉子純淨
    min_samples_split=10,  # 內部節點再劃分所需最小樣本數（預測更保守）
    min_samples_leaf=5,    # 葉子節點最少需要的樣本數（避免產生只為少數樣本服務的規則）
    random_state=42
)

model.fit(X_train, y_train)

# 6. 進行預測
y_pred = model.predict(X_test)

# 訓練集的預測與評估
y_train_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)

# 測試集的預測與評估 (你原本的代碼)
y_test_pred = model.predict(X_test)
r2_test = r2_score(y_test, y_test_pred)

print(f"訓練集 R2: {r2_train:.4f}")
print(f"測試集 R2: {r2_test:.4f}")


# 7. 評估模型 (迴歸問題使用的指標不同)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方誤差 (MSE): {mse:.2f}")
print(f"MAPE (平均絕對百分比誤差): {mape * 100:.2f}%")
print(f"R-squared 分數: {r2:.2f}") # 越接近 1 代表預測越準


importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print(feature_importance_df.head(10))


