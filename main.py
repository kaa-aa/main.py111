import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 페이지 설정
st.set_page_config(page_title="서울 기온 예측기", layout="wide")
st.title("🌡️ 서울 연평균 기온 예측기")

# 2. 데이터 로드 및 전처리
@st.cache_data
def load_and_process_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜 컬럼을 datetime형으로 변환 후 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 연도별 관측일수 및 평균기온 계산
    yearly = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 필터링: 2025년 이하 & 관측일수 300일 이상
    filtered_yearly = yearly[(yearly["연도"] <= 2025) & (yearly["관측일수"] >= 300)].copy()
    
    return filtered_yearly

df_yearly = load_and_process_data()

# 3. 회귀 모델 계산 (독립변수: 1908년부터 지난 연수)
# X = 연도 - 1908, y = 연평균기온
df_yearly["x"] = df_yearly["연도"] - 1908
X = df_yearly["x"].values
y = df_yearly["연평균기온"].values

# 선형 회귀 계수 (y = slope * x + intercept)
slope, intercept = np.polyfit(X, y, 1)

# 상관계수 계산
corr_matrix = np.corrcoef(X, y)
correlation = corr_matrix[0, 1]

# 데이터 정보 추출
num_years = len(df_yearly)
start_year = int(df_yearly["연도"].min())
end_year = int(df_yearly["연도"].max())

# 4. 요약 정보 출력
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 대상 연도 수", f"{num_years}개 해")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수 (r)", f"{correlation:.4f}")

st.divider()

# 5. 연도 선택 슬라이더 및 예상 기온 예측
selected_year = st.slider("예상 기온을 확인할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)

# 예측값 계산 (x_pred = selected_year - 1908)
x_pred = selected_year - 1908
predicted_temp = slope * x_pred + intercept

st.subheader(f"🔮 {selected_year}년 예상 연평균 기온")
st.title(f"{predicted_temp:.2f} °C")

st.divider()

# 6. Plotly 시각화
# 회귀선 표시를 위한 전체 연도 범위 (1900 ~ 2100)
line_years = np.arange(1900, 2101)
line_x = line_years - 1908
line_y = slope * line_x + intercept

fig = go.Figure()

# 관측값 산점도
fig.add_trace(go.Scatter(
    x=df_yearly["연도"],
    y=df_yearly["연평균기온"],
    mode="markers",
    name="관측 연평균기온",
    marker=dict(size=8, color="#1f77b4")
))

# 추세선 (회귀 직선)
fig.add_trace(go.Scatter(
    x=line_years,
    y=line_y,
    mode="lines",
    name="회귀 직선",
    line=dict(color="#ff7f0e", width=2)
))

# 사용자가 선택한 연도 강조 표시
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode="markers",
    name=f"선택한 해 ({selected_year}년 예측)",
    marker=dict(size=14, color="red", symbol="star")
))

fig.update_layout(
    title="서울 연도별 연평균기온 변화 및 추세선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="x unified",
    xaxis=dict(dtick=10)
)

st.plotly_chart(fig, use_container_width=True)
