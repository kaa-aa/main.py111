import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate realistic sample annual mean temperature data for 1906-2025 (120 years)
# Based on actual climate trends: base ~10.5C in 1906, slight warming 1906-1950, faster warming 1956-2005, accelerated warming 2006-2025
np.random.seed(42)
years = np.arange(1906, 2026) # 1906 to 2025 inclusive (120 years)

# Realistic temperature model: baseline 10.5 + non-linear warming trend + noise
trend = 10.5 + 0.008 * (years - 1906) + 0.0001 * (years - 1906)**2
noise = np.random.normal(0, 0.45, size=len(years))
temp = trend + noise

df = pd.DataFrame({'Year': years, 'Temp': temp})

# Split sets
train_100 = df[(df['Year'] >= 1906) & (df['Year'] <= 2005)] # 100 years
train_50 = df[(df['Year'] >= 1956) & (df['Year'] <= 2005)]  # 50 years
test_20 = df[(df['Year'] >= 2006) & (df['Year'] <= 2025)]   # 20 years

# Model 1: Trained on 100 years (1906-2005)
model_100 = LinearRegression()
model_100.fit(train_100[['Year']], train_100['Temp'])
pred_100_test = model_100.predict(test_20[['Year']])

# Model 2: Trained on 50 years (1956-2005)
model_50 = LinearRegression()
model_50.fit(train_50[['Year']], train_50['Temp'])
pred_50_test = model_50.predict(test_20[['Year']])

# Model Entire (1906-2025)
model_all = LinearRegression()
model_all.fit(df[['Year']], df['Temp'])
pred_all = model_all.predict(df[['Year']])

# Evaluation metrics function
def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

mae_100, mse_100, r2_100 = get_metrics(test_20['Temp'], pred_100_test)
mae_50, mse_50, r2_50 = get_metrics(test_20['Temp'], pred_50_test)
mae_all, mse_all, r2_all = get_metrics(df['Temp'], pred_all)

print(f"Entire Data (1906-2025): Slope={model_all.coef_[0]:.4f}, MAE={mae_all:.4f}, MSE={mse_all:.4f}, R2={r2_all:.4f}")
print(f"100-Year Model (1906-2005): Slope={model_100.coef_[0]:.4f}, MAE={mae_100:.4f}, MSE={mse_100:.4f}, R2={r2_100:.4f}")
print(f"50-Year Model (1956-2005): Slope={model_50.coef_[0]:.4f}, MAE={mae_50:.4f}, MSE={mse_50:.4f}, R2={r2_50:.4f}")
