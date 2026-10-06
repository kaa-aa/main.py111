import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

np.random.seed(42)
years = np.arange(1906, 2026) # 1906~2025

# Realistic climate temperature model
trend = 10.5 + 0.008 * (years - 1906) + 0.0001 * (years - 1906)**2
noise = np.random.normal(0, 0.45, size=len(years))
temp = trend + noise

df = pd.DataFrame({'Year': years, 'Temp': temp})

train_100 = df[(df['Year'] >= 1906) & (df['Year'] <= 2005)]
train_50 = df[(df['Year'] >= 1956) & (df['Year'] <= 2005)]
test_20 = df[(df['Year'] >= 2006) & (df['Year'] <= 2025)]

# Models
m_all = LinearRegression().fit(df[['Year']], df['Temp'])
m_100 = LinearRegression().fit(train_100[['Year']], train_100['Temp'])
m_50 = LinearRegression().fit(train_50[['Year']], train_50['Temp'])

p_all = m_all.predict(df[['Year']])
p_100 = m_100.predict(test_20[['Year']])
p_50 = m_50.predict(test_20[['Year']])

def get_m(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)

mae_all, mse_all, r2_all = get_m(df['Temp'], p_all)
mae_100, mse_100, r2_100 = get_m(test_20['Temp'], p_100)
mae_50, mse_50, r2_50 = get_m(test_20['Temp'], p_50)

print(f"Entire: Slope={m_all.coef_[0]:.4f}, Intercept={m_all.intercept_:.4f}, MAE={mae_all:.4f}, MSE={mse_all:.4f}, R2={r2_all:.4f}")
print(f"100yr:  Slope={m_100.coef_[0]:.4f}, Intercept={m_100.intercept_:.4f}, MAE={mae_100:.4f}, MSE={mse_100:.4f}, R2={r2_100:.4f}")
print(f"50yr:   Slope={m_50.coef_[0]:.4f}, Intercept={m_50.intercept_:.4f}, MAE={mae_50:.4f}, MSE={mse_50:.4f}, R2={r2_50:.4f}")

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(df['Year'], df['Temp'], color='gray', alpha=0.5, label='Actual Data (1906-2005)')
plt.scatter(test_20['Year'], test_20['Temp'], color='red', alpha=0.8, label='Test Data (2006-2025)')

x_range = np.arange(1906, 2026).reshape(-1, 1)
plt.plot(x_range, m_all.predict(x_range), 'k--', label=f'Entire (Slope: {m_all.coef_[0]:.4f})')
plt.plot(x_range, m_100.predict(x_range), 'b-', label=f'Train 100yr (Slope: {m_100.coef_[0]:.4f})')
plt.plot(x_range, m_50.predict(x_range), 'g-', label=f'Train 50yr (Slope: {m_50.coef_[0]:.4f})')

plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.title('Annual Mean Temperature Linear Regression')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('temp_regression_plot.png', dpi=150)
plt.close()

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 데이터 생성 (1906년~2025년 연평균 기온)
np.random.seed(42)
years = np.arange(1906, 2026)
trend = 10.5 + 0.008 * (years - 1906) + 0.0001 * (years - 1906)**2
temp = trend + np.random.normal(0, 0.45, size=len(years))
df = pd.DataFrame({'Year': years, 'Temp': temp})

# 2. 데이터 세트 분할
train_100 = df[(df['Year'] >= 1906) & (df['Year'] <= 2005)]  # 최근 100년
train_50  = df[(df['Year'] >= 1956) & (df['Year'] <= 2005)]  # 최근 50년
test_20   = df[(df['Year'] >= 2006) & (df['Year'] <= 2025)]  # 공통 테스트 (최근 20년)

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
