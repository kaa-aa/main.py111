import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 데이터 준비 (1906년~2025년 연평균 기온)
np.random.seed(42)
years = np.arange(1906, 2026)
trend = 10.5 + 0.008 * (years - 1906) + 0.0001 * (years - 1906)**2
temp = trend + np.random.normal(0, 0.45, size=len(years))
df = pd.DataFrame({'Year': years, 'Temp': temp})

# 2. 데이터 세트 분할
train_100 = df[(df['Year'] >= 1906) & (df['Year'] <= 2005)]  # 최근 100년 (1906~2005)
train_50  = df[(df['Year'] >= 1956) & (df['Year'] <= 2005)]  # 최근 50년 (1956~2005)
test_20   = df[(df['Year'] >= 2006) & (df['Year'] <= 2025)]  # 공통 테스트 (2006~2025)

# 3. 모델 학습 및 예측
model_all = LinearRegression().fit(df[['Year']], df['Temp'])
pred_all  = model_all.predict(df[['Year']])

model_100 = LinearRegression().fit(train_100[['Year']], train_100['Temp'])
pred_100  = model_100.predict(test_20[['Year']])

model_50  = LinearRegression().fit(train_50[['Year']], train_50['Temp'])
pred_50   = model_50.predict(test_20[['Year']])

# 4. 성능 평가 함수
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    return mae, mse, r2

mae_all, mse_all, r2_all = evaluate(df['Temp'], pred_all)
mae_100, mse_100, r2_100 = evaluate(test_20['Temp'], pred_100)
mae_50, mse_50, r2_50   = evaluate(test_20['Temp'], pred_50)
