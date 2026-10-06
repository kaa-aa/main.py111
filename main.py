import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("서울 기온 데이터를 기반으로 연평균 기온 변화 트렌드를 분석하고 미래 기온을 예측합니다.")

# 1. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    # 데이터 읽기 (CSV UTF-8)
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    # 열 이름 공백 제거
    df.columns = df.columns.str.strip()
    
    # 날짜 컬럼 datetime 변환 및 연도 추출
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    # 조건 1: 2025년 이하 데이터만 사용
    df = df[df['연도'] <= 2025]
    
    # 연도별 평균기온 및 관측일수 계산
    yearly_df = df.groupby('연도').agg(
        연평균기온=('평균기온', 'mean'),
        관측일수=('평균기온', 'count')
    ).reset_index()
    
    # 조건 2: 관측일수가 300일 이상인 연도만 정제
    yearly_clean = yearly_df[yearly_df['관측일수'] >= 300].copy()
    
    return yearly_clean

df_clean = load_data()

# 2. 회귀 모델 생성 (독립변수: 1908년부터 지난 연수)
df_clean['지난연수'] = df_clean['연도'] - 1908

X = df_clean[['지난연수']]
y = df_clean['연평균기온']

model = LinearRegression()
model.fit(X, y)

# 회귀선 예측값 추가
df_clean['회귀선'] = model.predict(X)

# 데이터 기본 정보 산출
num_years = len(df_clean)
start_year = int(df_clean['연도'].min())
end_year = int(df_clean['연도'].max())

# 상관계수 계산
corr = df_clean['연도'].corr(df_clean['연평균기온'])

# --- 화면 레이아웃 구성 ---

# 지표 요약 수치
st.subheader("📌 데이터 및 학습 요약")
col1, col2, col3, col4 = st.columns(4)
col1.metric("학습 데이터 연도 개수", f"{num_years}개")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("연도-기온 상관계수", f"{corr:.4f}")

st.markdown("---")

# 3. 예측 슬라이더 및 큰 글씨 표시
st.subheader("🔮 예측할 연도 선택")
selected_year = st.slider("연도를 선택하세요:", min_value=1900, max_value=2100, value=2030, step=1)

# 선택한 연도의 예측 기온 계산 (1908년 기준 지난 연수)
years_passed = selected_year - 1908
predicted_temp = model.predict([[years_passed]])[0]

# 큰 수치 카드로 표시
st.markdown(
    f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 25px;">
        <h3 style="color: #333333; margin: 0;">📅 {selected_year}년 서울 예상 연평균 기온</h3>
        <h1 style="color: #ff4b4b; font-size: 50px; margin: 10px 0 0 0;">{predicted_temp:.2f} °C</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# 4. Plotly 산점도 및 회귀선 시각화
st.subheader("📊 연평균 기온 산점도 및 회귀선")

# 전체 범위(1900~2100년) 회귀선 라인 데이터 생성
plot_years = np.arange(1900, 2101)
plot_years_passed = (plot_years - 1908).reshape(-1, 1)
plot_pred_temp = model.predict(plot_years_passed)

fig = go.Figure()

# 실제 연평균 기온 산점도
fig.add_trace(go.Scatter(
    x=df_clean['연도'],
    y=df_clean['연평균기온'],
    mode='markers',
    name='실제 연평균기온',
    marker=dict(color='#1f77b4', size=8, opacity=0.8)
))

# 전체 기간 회귀선
fig.add_trace(go.Scatter(
    x=plot_years,
    y=plot_pred_temp,
    mode='lines',
    name='선형 회귀선',
    line=dict(color='#ff4b4b', width=2)
))

# 선택된 연도 강조 표시
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode='markers+text',
    name=f'{selected_year}년 예측점',
    marker=dict(color='green', size=14, symbol='star'),
    text=[f"{selected_year}년<br>{predicted_temp:.2f}°C"],
    textposition="top center"
))

# 그래프 레이아웃 설정
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (°C)",
    hovermode="x unified",
    template="plotly_white",
    height=550,
    xaxis=dict(range=[1895, 2105], dtick=20)
)

st.plotly_chart(fig, use_container_width=True)

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
