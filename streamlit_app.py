# streamlit_app.py
# -*- coding: utf-8 -*-
# =========================================================
# 기온 변화와 자살률 대시보드 (Streamlit + GitHub Codespaces)
# 1) 공개 데이터 대시보드 (World Bank + NASA GISTEMP)
# 2) 사용자 입력(이미지/설명) 기반 대시보드 (스탠퍼드 그래프 유사 재현)
# ---------------------------------------------------------
# ✅ 데이터 출처(공식, 코드 내 주석 참고)
# - World Bank Suicide mortality rate, code: SH.STA.SUIC.P5
#   API 문서: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
#   예시: https://api.worldbank.org/v2/country/USA/indicator/SH.STA.SUIC.P5?format=json&per_page=20000
# - NASA GISTEMP v4 Global mean temperature anomaly (연평균, °C, 1951–1980 기준)
#   CSV: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
#   문서: https://data.giss.nasa.gov/gistemp/
# 실패 시 → 예시 데이터로 자동 대체(화면 안내) / 미래 데이터 제거 규칙 적용
# =========================================================

import io
import os
import json
import datetime as dt
from base64 import b64encode

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_echarts import st_echarts

# ------------------------------
# 기본 설정 & 폰트 주입
# ------------------------------
st.set_page_config(page_title="기온 변화와 자살률 대시보드", layout="wide")

def inject_font_css():
    """/fonts/Pretendard-Bold.ttf 존재 시 Streamlit/Plotly/HTML에 적용"""
    font_path = "/fonts/Pretendard-Bold.ttf"
    if os.path.exists(font_path):
        with open(font_path, "rb") as f:
            font_data = b64encode(f.read()).decode("utf-8")
        st.markdown(
            f"""
            <style>
            @font-face {{
                font-family: 'Pretendard';
                src: url(data:font/ttf;base64,{font_data}) format('truetype');
                font-weight: 700; font-style: normal; font-display: swap;
            }}
            html, body, [class*="css"] {{
                font-family: 'Pretend', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif !important;
            }}
            .echarts-text, .plotly, .js-plotly-plot * {{
                font-family: 'Pretendard', sans-serif !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        # Matplotlib/Seaborn용
        matplotlib.rcParams["font.family"] = "Pretendard"
        matplotlib.rcParams["axes.unicode_minus"] = False
        sns.set_theme(style="whitegrid")
    else:
        sns.set_theme(style="whitegrid")

inject_font_css()

TODAY = dt.date.today()
THIS_YEAR = TODAY.year

st.title("🌡️ 기온 변화와 자살률 대시보드")
st.caption("공개 데이터 우선 → 실패 시 예시 데이터로 대체합니다. (현지 자정 이후의 미래 데이터는 표시하지 않음)")

# ------------------------------
# 유틸 / 공통 함수
# ------------------------------
def _ensure_date_year(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    """기본 스키마(date,value,group)에 맞게 연도→date(연-01-01) 표준화"""
    out = df.copy()
    out = out[pd.to_numeric(out[year_col], errors="coerce").notna()]
    out["year"] = out[year_col].astype(int)
    out["date"] = pd.to_datetime(out["year"].astype(str) + "-01-01")
    out = out[out["date"].dt.date <= TODAY]  # 미래 제거
    return out

@st.cache_data(ttl=3600)
def fetch_worldbank_suicide(country_code: str) -> pd.DataFrame:
    """World Bank 자살률(연도별, 10만명당) 다운로드
    예시: https://api.worldbank.org/v2/country/USA/indicator/SH.STA.SUIC.P5?format=json&per_page=20000
    """
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/SH.STA.SUIC.P5"
    params = {"format": "json", "per_page": 20000}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    rows = []
    if isinstance(js, list) and len(js) > 1 and isinstance(js[1], list):
        for rec in js[1]:
            year = rec.get("date")
            val = rec.get("value")
            if year is None:
                continue
            try:
                yi = int(year)
            except:
                continue
            if yi <= THIS_YEAR and val is not None:
                rows.append({"year": yi, "suicide_rate": float(val)})
    df = pd.DataFrame(rows).sort_values("year")
    return df

@st.cache_data(ttl=3600)
def fetch_nasa_gistemp_global() -> pd.DataFrame:
    """NASA GISTEMP v4 연평균 온도 이상치(°C) 파싱
    CSV: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
    """
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    content = r.content.decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(content), skiprows=1)
    # 표준 헤더 보정
    if "Year" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Year"})
    if "J-D" not in df.columns:
        # 마지막 열을 연평균으로 간주(보수적)
        df["J-D"] = df.iloc[:, -1]
    out = df[["Year", "J-D"]].rename(columns={"Year": "year", "J-D": "temp_anomaly"})
    out = out[pd.to_numeric(out["temp_anomaly"], errors="coerce").notna()]
    out["year"] = out["year"].astype(int)
    # NASA 파일은 보통 0.01°C 단위 → °C로 변환
    out["temp_anomaly"] = out["temp_anomaly"].astype(float) / 100.0
    out = out[out["year"] <= THIS_YEAR]
    return out.sort_values("year")

def tiny_example() -> pd.DataFrame:
    years = np.arange(2000, 2012)
    rng = np.random.default_rng(7)
    temp = np.linspace(0.1, 0.9, len(years)) + rng.normal(0, 0.03, len(years))
    suic = np.linspace(16.0, 13.8, len(years)) + rng.normal(0, 0.25, len(years))
    return pd.DataFrame({"year": years, "temp_anomaly": temp, "suicide_rate": suic})

def fit_ols(x, y):
    slope, intercept, r, p, se = stats.linregress(np.asarray(x, float), np.asarray(y, float))
    return slope, intercept, r, p, se

def download_button_from_df(df: pd.DataFrame, label: str, fname: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=fname, mime="text/csv")

# =========================================================
# 사이드바
# =========================================================
st.sidebar.header("⚙️ 보기 설정")
page = st.sidebar.radio("대시보드 선택", ["① 공개 데이터", "② 사용자 입력(이미지/설명)"], index=0)

# =========================================================
# ① 공개 데이터 대시보드
# =========================================================
if page == "① 공개 데이터":
    st.subheader("① 공개 데이터 대시보드")
    st.markdown(
        "- **자살률**: World Bank `SH.STA.SUIC.P5` (연도별, 10만 명당)\n"
        "- **기온**: NASA GISTEMP v4 글로벌 연평균 온도 이상치(°C, 1951–1980 기준)"
    )

    col1, col2 = st.columns(2)
    with col1:
        country_label = st.selectbox(
            "국가 선택",
            [
                "United States (USA)",
                "Mexico (MEX)",
                "Korea, Rep. (KOR)",
                "Japan (JPN)",
                "Germany (DEU)",
            ],
            index=0,
        )
    with col2:
        st.info("서로 다른 단위/범위이므로 **동일 연도** 기준으로 단순 상관을 봅니다. 인과 아님.")

    code = country_label.split("(")[-1].replace(")", "").strip()

    # 데이터 로드
    got_api = True
    try:
        wb = fetch_worldbank_suicide(code)
        nasa = fetch_nasa_gistemp_global()
    except Exception as e:
        got_api = False
        st.warning("🔌 공개 데이터 API 연결 실패. 예시 데이터로 대체합니다.")
        st.exception(e)

    if got_api and not wb.empty and not nasa.empty:
        merged = pd.merge(wb, nasa, on="year", how="inner")
        merged = merged[merged["year"] <= THIS_YEAR].sort_values("year")
        if merged.empty:
            st.warning("교집합 연도가 없어 예시 데이터로 대체합니다.")
            merged = tiny_example()
            mode = "예시 데이터"
        else:
            mode = "공개 데이터"
    else:
        merged = tiny_example()
        mode = "예시 데이터"

    # 표준 스키마(date,value,group) 보조 테이블
    merged_std = _ensure_date_year(merged.rename(columns={"suicide_rate": "value"}), "year")
    merged_std["group"] = "자살률(10만명당)"
    merged_temp = _ensure_date_year(merged.rename(columns={"temp_anomaly": "value"}), "year")
    merged_temp["group"] = "기온 이상치(°C)"
    std_all = pd.concat([merged_std, merged_temp], ignore_index=True)

    # 연도 범위 필터
    y_min, y_max = int(merged["year"].min()), int(merged["year"].max())
    y1, y2 = st.slider("연도 범위", y_min, y_max, (max(y_min, y_max - 30), y_max))
    merged = merged[(merged["year"] >= y1) & (merged["year"] <= y2)]
    std_all = std_all[(std_all["date"].dt.year >= y1) & (std_all["date"].dt.year <= y2)]

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.line(
            merged, x="year", y="suicide_rate", markers=True,
            title=f"자살률 추이 — {country_label}  ({mode})",
            labels={"year": "연도", "suicide_rate": "자살률(10만명당)"}
        )
        fig1.update_layout(height=370)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.line(
            merged, x="year", y="temp_anomaly", markers=True,
            title="글로벌 연평균 기온 이상치 (NASA GISTEMP, °C)",
            labels={"year": "연도", "temp_anomaly": "온도 이상치(°C)"}
        )
        fig2.update_layout(height=370)
        st.plotly_chart(fig2, use_container_width=True)

    # 산점도 + OLS
    if len(merged) >= 3:
        slope, intercept, r, p, se = fit_ols(merged["temp_anomaly"], merged["suicide_rate"])
        xg = np.linspace(merged["temp_anomaly"].min(), merged["temp_anomaly"].max(), 100)
        yhat = slope * xg + intercept
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=merged["temp_anomaly"], y=merged["suicide_rate"],
            mode="markers", name="연도별 관측치"
        ))
        fig3.add_trace(go.Scatter(
            x=xg, y=yhat, mode="lines", name="선형회귀(OLS)"
        ))
        fig3.update_layout(
            title=f"온도 이상치 vs 자살률 상관 — r={r:.3f}, p={p:.3g}, slope={slope:.3f}",
            xaxis_title="온도 이상치(°C, NASA)",
            yaxis_title="자살률(10만명당, World Bank)",
            height=420,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ECharts로 동일 데이터 미니 컴보(라인 2종)
    st.markdown("#### ECharts 미니 뷰")
    opt = {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["자살률", "온도 이상치"]},
        "xAxis": {"type": "category", "data": merged["year"].astype(str).tolist()},
        "yAxis": [{"type": "value"}, {"type": "value"}],
        "series": [
            {"name": "자살률", "type": "line", "data": merged["suicide_rate"].round(2).tolist(), "yAxisIndex": 0},
            {"name": "온도 이상치", "type": "line", "data": merged["temp_anomaly"].round(3).tolist(), "yAxisIndex": 1},
        ],
    }
    st_echarts(opt, height="340px")

    # 다운로드
    st.markdown("##### 전처리된 표 다운로드")
    download_button_from_df(std_all[["date", "value", "group"]], "CSV 내려받기", f"public_processed_{code}.csv")

    with st.expander("데이터 출처(공식 링크)"):
        st.markdown(
            "- World Bank — Suicide mortality rate (`SH.STA.SUIC.P5`)\n"
            "  - https://api.worldbank.org/v2/country/USA/indicator/SH.STA.SUIC.P5?format=json&per_page=20000\n"
            "- NASA GISTEMP v4 — Global mean temperature anomaly (annual)\n"
            "  - https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
        )
        st.caption("주의: 단순 연도 매칭 상관은 인과를 의미하지 않습니다. 교란변수 통제가 필요합니다.")

# =========================================================
# ② 사용자 입력(이미지/설명) 기반 대시보드
# =========================================================
else:
    st.subheader("② 사용자 입력(이미지/설명) 기반 대시보드")
    st.caption("입력: “기온 상승과 자살률 변화의 상관관계 — 왼쪽 미국, 오른쪽 멕시코 (스탠퍼드)”")
    # 제공된 이미지(컨테이너 경로)
    img_path = "/mnt/data/74d556af-4516-426b-b19f-d445ebb10fb2.png"
    if os.path.exists(img_path):
        st.image(Image.open(img_path), caption="참고 이미지(설명 재현용)")

    st.info("아래 그래프는 원문 연구 이미지를 **설명 기반으로 유사 재현**한 예시 데이터입니다. 실제 수치와 다를 수 있습니다.")

    # 예시 데이터 생성(미국, 멕시코 — 선형 추세 + 신뢰대역 느낌)
    def synth(country="USA", n=60, x_min=-20, x_max=40, seed=0):
        rng = np.random.default_rng(seed)
        x = np.linspace(x_min, x_max, n)
        if country == "USA":
            slope, intercept, noise_sd, band = 0.5, -10, 2.2, 3.0
        else:
            slope, intercept, noise_sd, band = 1.0, -15, 3.2, 4.2
        y_hat = intercept + slope * x
        y = y_hat + rng.normal(0, noise_sd, size=n)
        # 대역: 20°C 근방 좁고 멀수록 넓어지도록
        width = (np.abs(x - 20) / (x_max - x_min) + 0.1) * band
        ci_lo = y_hat - 1.64 * width
        ci_hi = y_hat + 1.64 * width
        df = pd.DataFrame({"temp_C": x, "pct_change": y, "y_hat": y_hat, "ci_lo": ci_lo, "ci_hi": ci_hi})
        return df, slope

    us, s_us = synth("USA", seed=11)
    mx, s_mx = synth("MEX", seed=22)

    def panel(df: pd.DataFrame, title: str):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.concat([df["temp_C"], df["temp_C"][::-1]]),
            y=pd.concat([df["ci_hi"], df["ci_lo"][::-1]]),
            fill="toself", mode="lines", line=dict(width=0), name="신뢰 구간", opacity=0.25,
        ))
        fig.add_trace(go.Scatter(x=df["temp_C"], y=df["y_hat"], mode="lines", name="추정치"))
        fig.update_layout(
            title=title, height=480,
            xaxis_title="월평균 기온(°C)", yaxis_title="자살률 변화율(%)",
            yaxis=dict(range=[-40, 40])
        )
        return fig

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(panel(us, "United States"), use_container_width=True)
        st.caption(f"가정한 기울기: {s_us:+.2f} %/°C")
    with c2:
        st.plotly_chart(panel(mx, "Mexico"), use_container_width=True)
        st.caption(f"가정한 기울기: {s_mx:+.2f} %/°C")

    # 표준 스키마로 내보내기용 테이블
    us_std = us.rename(columns={"temp_C": "date", "pct_change": "value"}).copy()
    us_std["group"] = "미국(예시)"
    mx_std = mx.rename(columns={"temp_C": "date", "pct_change": "value"}).copy()
    mx_std["group"] = "멕시코(예시)"
    # x축이 '온도'라서 date 스키마 요구와 다르지만, 규칙을 유지하기 위해 수치형을 그대로 기록
    export_df = pd.concat([us_std[["date", "value", "group"]], mx_std[["date", "value", "group"]]], ignore_index=True)
    download_button_from_df(export_df, "예시 데이터 CSV 내려받기", "user_example_panels.csv")

    with st.expander("설명"):
        st.markdown(
            "- 두 패널은 이미지 설명을 토대로 **선형 증가 추세**와 **신뢰대역 형태**를 모사했습니다.\n"
            "- 실제 재현을 위해서는 **월별 지역 기온**과 **월별 자살 사망 수(인구보정)**, 그리고 고정효과 모형이 필요합니다."
        )

st.markdown("---")
st.caption("© 공개 데이터 우선 원칙. 실패 시 예시 데이터로 자동 대체. 모든 라벨/툴팁은 한국어로 제공됩니다.")
