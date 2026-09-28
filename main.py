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

# 3. 전체 기간 회귀 모델 (독립변수: 1908년부터 지난 연수)
df_yearly["x"] = df_yearly["연도"] - 1908
X_all = df_yearly["x"].values
y_all = df_yearly["연평균기온"].values

slope_all, intercept_all = np.polyfit(X_all, y_all, 1)

# 4. 최근 20년 회귀 모델 (기준: 데이터의 가장 최근 연도 포함 20년)
max_year = int(df_yearly["연도"].max())
min_year_recent = max_year - 19
df_recent = df_yearly[df_yearly["연도"] >= min_year_recent].copy()

X_recent = df_recent["x"].values
y_recent = df_recent["연평균기온"].values

slope_recent, intercept_recent = np.polyfit(X_recent, y_recent, 1)

# 100년당 기온 상승량 계산 (기울기 * 100)
rate_100_all = slope_all * 100
rate_100_recent = slope_recent * 100

# 5. 상승 속도 비교 카드 (상단에 크게 표시)
st.subheader("📈 100년당 기온 상승 속도 비교")
col_rate1, col_rate2 = st.columns(2)

with col_rate1:
    st.metric(
        label=f"전체 기간 ({df_yearly['연도'].min()}년 ~ {max_year}년)",
        value=f"+{rate_100_all:.2f} °C / 100년",
        help="전체 데이터를 바탕으로 계산한 100년당 기온 상승량입니다."
    )

with col_rate2:
    # 전체 기간 대비 최근 20년의 가속도 계산
    diff = rate_100_recent - rate_100_all
    st.metric(
        label=f"최근 20년 ({min_year_recent}년 ~ {max_year}년)",
        value=f"+{rate_100_recent:.2f} °C / 100년",
        delta=f"+{diff:.2f} °C 더 빠르게 상승 중" if diff > 0 else f"{diff:.2f} °C",
        help="최근 20년 데이터만으로 계산한 100년당 기온 상승량입니다."
    )

st.divider()

# 6. 기본 요약 정보
num_years = len(df_yearly)
start_year = int(df_yearly["연도"].min())

col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 대상 연도 수", f"{num_years}개 해")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{max_year}년")
col4.metric("전체 상관계수 (r)", f"{np.corrcoef(X_all, y_all)[0, 1]:.4f}")

st.divider()

# 7. 연도 선택 슬라이더 및 예측값 계산 (전체 기간 모델 기준)
selected_year = st.slider("예상 기온을 확인할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)

x_pred = selected_year - 1908
pred_all = slope_all * x_pred + intercept_all
pred_recent = slope_recent * x_pred + intercept_recent

st.subheader(f"🔮 {selected_year}년 예상 연평균 기온")
col_pred1, col_pred2 = st.columns(2)
with col_pred1:
    st.markdown(f"**전체 추세 기준:** `{pred_all:.2f} °C`")
with col_pred2:
    st.markdown(f"**최근 20년 추세 기준:** `{pred_recent:.2f} °C`")

st.divider()

# 8. Plotly 시각화 (두 개의 추세선 비교)
line_years = np.arange(1900, 2101)
line_x = line_years - 1908

line_y_all = slope_all * line_x + intercept_all
line_y_recent = slope_recent * line_x + intercept_recent

fig = go.Figure()

# 관측 데이터 (산점도)
fig.add_trace(go.Scatter(
    x=df_yearly["연도"],
    y=df_yearly["연평균기온"],
    mode="markers",
    name="관측 연평균기온",
    marker=dict(size=8, color="#1f77b4")
))

# 전체 기간 추세선
fig.add_trace(go.Scatter(
    x=line_years,
    y=line_y_all,
    mode="lines",
    name=f"전체 추세선 (+{rate_100_all:.2f}°C/100년)",
    line=dict(color="#ff7f0e", width=2)
))

# 최근 20년 추세선
fig.add_trace(go.Scatter(
    x=line_years,
    y=line_y_recent,
    mode="lines",
    name=f"최근 20년 추세선 (+{rate_100_recent:.2f}°C/100년)",
    line=dict(color="#d62728", width=2, dash="dash")
))

# 선택한 해 표시 (전체 추세선 기준)
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[pred_all],
    mode="markers",
    name=f"선택 연도 ({selected_year}년)",
    marker=dict(size=14, color="purple", symbol="star")
))

fig.update_layout(
    title="서울 연도별 연평균기온 및 추세선 비교 (전체 vs 최근 20년)",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="x unified",
    xaxis=dict(dtick=10)
)

st.plotly_chart(fig, use_container_width=True)
