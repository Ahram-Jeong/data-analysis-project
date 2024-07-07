import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# --- page
st.set_page_config(page_title = "Analysis | Jeong Ahram")

# 기간
start_date = "2014-09-01"
end_date = "2019-12-31"

# 종목
bitcoin = "BTC-USD"
gold = "GC=F"
usd = "EURUSD=X"
wti = "CL=F"


# 일간 티커 데이터 수집
def ticker_data(tk_name, ds, de):
    result = yf.download(tk_name, ds, de)
    result["change"] = result["Adj Close"].diff()
    result["daily_return"] = result["Adj Close"].pct_change() * 100
    result = result.dropna()  # NaN 포함 행 삭제
    return result


# 주간 티커 데이터 수집
def ticker_data_wk(tk_name, ds, de):
    result = yf.download(tk_name, ds, de, interval="1wk")
    result["weekly_return"] = result["Adj Close"].pct_change() * 100
    result = result.dropna()
    return result


# 일별 수정 종가 (Adj Close) 추출 함수
def daily_adj_close(tk_name, ds, de):
    result = yf.download(tk_name, ds, de)
    return result["Adj Close"]


# 일별 수익률 (daily_return) 추출 함수
def daily_return_rate(tk_name, ds, de):
    result = ticker_data(tk_name, ds, de)
    return result["daily_return"]


# 주간 수익률 (weekly_return) 추출 함수
def weekly_return_rate(tk_name, ds, de):
    result = ticker_data_wk(tk_name, ds, de)
    return result["weekly_return"]


# 비트코인과 티커 간의 상관성 산출
def corr_bit(tk_name, ds, de):
    result = weekly_return_rate(bitcoin, ds, de) / weekly_return_rate(tk_name, ds, de)
    result = result.dropna()
    return result


# 계산식 for 선형 회귀
def lin_cal(tk_name, ds, de):
    tk2019 = ticker_data(tk_name, ds, de)  # 제도 편입 전 기간의 티커 데이터 수집
    std2019 = tk2019[tk2019.index == "2019-11-15"]  # 기준이 되는 날짜의 행 추출
    std2019 = std2019["Adj Close"].values[0]  # 해당 행의 수정 종가
    result = (ticker_data(tk_name, ds, de)["Adj Close"] / std2019) * 100  # 티커의 수정 종가 컬럼과 연산
    return result


# 티커 생성 시 호출 함수
def ticker_call(tk_name, ds, de):
    result = ticker_data(tk_name, ds, de)  # 일간 티커 데이터 수집
    result["chg_rat_exp_2019"] = lin_cal(tk_name, ds, de)  # 선형 회귀 산출을 위해 연산된 컬럼 추가
    return result


# 선형 회귀 계수 산출
def lin_data(tk_name, ds, de):
    bit_col = ticker_call(bitcoin, ds, de)["chg_rat_exp_2019"]
    tk_col = ticker_call(tk_name, ds, de)["chg_rat_exp_2019"]

    result = pd.concat([bit_col, tk_col], axis=1)
    result.columns = ["bit", "tk"]
    result = result.dropna()

    model = stats.linregress(result["bit"], result["tk"])  # 선형 회귀 계수 객체 생성
    dict_lin = {
        "slope": model.slope,
        "intercept": model.intercept,
        "rvalue": model.rvalue,
        "pvalue": model.pvalue,
        "stderr": model.stderr,
        "intercept_stderr": model.intercept_stderr
    }
    result_lin = pd.DataFrame(list(dict_lin.values()), index=dict_lin.keys())
    return result_lin

bit_ticker = ticker_call(bitcoin, start_date, end_date)

# 폰트 설정
plt.rc("font", family = "NanumGothic", size = 13)
plt.rcParams["axes.unicode_minus"] = False

# --- body
st.subheader("비트코인의 자산가치 분석을 위한 데이터 시각화")

# --- tabs1
tabs1 = st.tabs(["일별 시세", "주간 시세", "상관성 그래프"])

with tabs1[0] :
    # graph 1
    st.subheader("일별 시세 변동 그래프")
    scaler = MinMaxScaler()
    all_dates = pd.date_range(start = start_date, end = end_date, freq = "D")

    # interpolate() : 결측치 보간
    y1_series = daily_adj_close(bitcoin, start_date, end_date).reindex(all_dates).interpolate()
    y2_series = daily_adj_close(gold, start_date, end_date).reindex(all_dates).interpolate()
    y3_series = daily_adj_close(usd, start_date, end_date).reindex(all_dates).interpolate()
    y4_series = daily_adj_close(wti, start_date, end_date).reindex(all_dates).interpolate()

    # 스케일링할 데이터 모양 변경
    y1 = y1_series.values.reshape(-1, 1)
    y2 = y2_series.values.reshape(-1, 1)
    y3 = y3_series.values.reshape(-1, 1)
    y4 = y4_series.values.reshape(-1, 1)

    dates = y1_series.index # 날짜 추출

    g1 = plt.figure(figsize=(20, 10))
    plt.plot(dates, scaler.fit_transform(y1), label = "Bitcoin")
    plt.plot(dates, scaler.fit_transform(y2), label = "Gold")
    plt.plot(dates, scaler.fit_transform(y3), label = "EUR/USD")
    plt.plot(dates, scaler.fit_transform(y4), label = "WTI")
    plt.legend(loc="best", prop={"size": 20})
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation = 300)
    plt.grid(True)
    st.pyplot(g1)

with tabs1[1] :
    # graph 2
    st.subheader("주간 시세 변동 그래프")
    y1_wk = ticker_data_wk(bitcoin, start_date, end_date)["Adj Close"]
    y2_wk = ticker_data_wk(gold, start_date, end_date)["Adj Close"]
    y3_wk = ticker_data_wk(usd, start_date, end_date)["Adj Close"]
    y4_wk = ticker_data_wk(wti, start_date, end_date)["Adj Close"]

    # 스케일링할 데이터 모양 변경
    y1 = y1_wk.values.reshape(-1, 1)
    y2 = y2_wk.values.reshape(-1, 1)
    y3 = y3_wk.values.reshape(-1, 1)
    y4 = y4_wk.values.reshape(-1, 1)

    dates = y1_wk.index  # 날짜 추출

    g2 = plt.figure(figsize=(20, 10))
    plt.plot(y1_wk.index, scaler.fit_transform(y1), label = "Bitcoin")
    plt.plot(y2_wk.index, scaler.fit_transform(y2), label = "Gold")
    plt.plot(y3_wk.index, scaler.fit_transform(y3), label = "EUR/USD")
    plt.plot(y4_wk.index, scaler.fit_transform(y4), label = "WTI")
    plt.legend(loc="best", prop={"size": 20})
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation = 300)
    plt.grid(True)
    st.pyplot(g2)

with tabs1[2] :
    # graph 3
    # 상관성 그래프 시각화
    st.subheader("비트코인/티커 상관성 그래프")
    corr_y1 = corr_bit(bitcoin, start_date, end_date).reindex(all_dates).interpolate()
    corr_y2 = corr_bit(gold, start_date, end_date).reindex(all_dates).interpolate()
    corr_y3 = corr_bit(usd, start_date, end_date).reindex(all_dates).interpolate()
    corr_y4 = corr_bit(wti, start_date, end_date).reindex(all_dates).interpolate()

    y1 = y1_series.values.reshape(-1, 1)
    y2 = y2_series.values.reshape(-1, 1)
    y3 = y3_series.values.reshape(-1, 1)
    y4 = y4_series.values.reshape(-1, 1)

    dates = corr_y1.index
    g3 = plt.figure(figsize=(20, 10))
    plt.plot(dates, scaler.fit_transform(y1), label = "Bitcoin")
    plt.plot(dates, scaler.fit_transform(y2), label = "Gold")
    plt.plot(dates, scaler.fit_transform(y3), label = "EUR/USD")
    plt.plot(dates, scaler.fit_transform(y4), label = "WTI")
    plt.legend(loc = "best", prop = {"size" : 20})
    plt.xlabel("Date")
    plt.ylabel("Rate")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation = 300)
    plt.grid(True)
    st.pyplot(g3)
st.write("위의 그래프를 보면, 일별 시세 그래프와 주간 시세 그래프, 상관성 산출을 위해 계산식을 적용한 상관성 그래프 모두 같은 모양으로 나오는 것을 확인 할 수 있으므로, 비교 분석은 가독성이 편한 주간 시세 그래프로 진행하도록 한다.")

# --- tabs2
st.subheader("비트코인 비교군 간의 상관성 그래프")
tabs2 = st.tabs(["비트코인 / 금", "비트코인 / 달러", "비트코인 / 원유"])

with tabs2[0] :
    y1_wk = ticker_data_wk(bitcoin, start_date, end_date)["Adj Close"]
    y2_wk = ticker_data_wk(gold, start_date, end_date)["Adj Close"]

    # 스케일링할 데이터 모양 변경
    y1 = y1_wk.values.reshape(-1, 1)
    y2 = y2_wk.values.reshape(-1, 1)

    dates = y1_wk.index  # 날짜 추출

    g2 = plt.figure(figsize=(20, 10))
    plt.plot(y1_wk.index, scaler.fit_transform(y1), label="Bitcoin")
    plt.plot(y2_wk.index, scaler.fit_transform(y2), label="Gold")
    plt.legend(loc="best", prop={"size": 20})
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=300)
    plt.grid(True)
    st.pyplot(g2)

with tabs2[1] :
    y1_wk = ticker_data_wk(bitcoin, start_date, end_date)["Adj Close"]
    y3_wk = ticker_data_wk(usd, start_date, end_date)["Adj Close"]

    # 스케일링할 데이터 모양 변경
    y1 = y1_wk.values.reshape(-1, 1)
    y3 = y3_wk.values.reshape(-1, 1)

    dates = y1_wk.index  # 날짜 추출

    g2 = plt.figure(figsize=(20, 10))
    plt.plot(y1_wk.index, scaler.fit_transform(y1), label="Bitcoin")
    plt.plot(y3_wk.index, scaler.fit_transform(y3), label="EUR/USD")
    plt.legend(loc="best", prop={"size": 20})
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=300)
    plt.grid(True)
    st.pyplot(g2)

    with tabs2[2] :
        y1_wk = ticker_data_wk(bitcoin, start_date, end_date)["Adj Close"]
        y4_wk = ticker_data_wk(wti, start_date, end_date)["Adj Close"]

        # 스케일링할 데이터 모양 변경
        y1 = y1_wk.values.reshape(-1, 1)
        y4 = y4_wk.values.reshape(-1, 1)

        dates = y1_wk.index  # 날짜 추출

        g2 = plt.figure(figsize=(20, 10))
        plt.plot(y1_wk.index, scaler.fit_transform(y1), label="Bitcoin")
        plt.plot(y4_wk.index, scaler.fit_transform(y4), label="WTI")
        plt.legend(loc="best", prop={"size": 20})
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=300)
        plt.grid(True)
        st.pyplot(g2)

st.subheader("통계적 수치를 위한 단순 선형 회귀")
st.write("""
**✅ 안전/안정적 자산 비교군 추가**
- iShares 20+ Year Treasury Bond ETF : 미국 국채 ETF, 미국의 장기 국채에 투자하여 안정적인 수익을 추구, 종목코드 "TLT"
- iShares iBoxx $ Investment Grade Corporate Bond ETF : 미국 투자등급 회사채에 투자하는 ETF, 종목코드 "LQD"
- Duke Energy : 전력 및 천연가스 서비스를 제공하는 유틸리티 부문 공공 주식, 종목코드 "DUK"
- Gold ETF : 나스닥 100 지수를 추종하는 ETF, 금 가격에 연동되어 금에 투자하는 효과를 제공하는 ETF, 종목코드 "GLD"

**✅ 위험 자산 비교군 추가**
- Brent oil :  종목코드 "BZ=F"
- Tesla :  종목코드 "TSLA"
- Dow Jones Industrial Average : 다우존스 산업 평균 지수, 주로 대형 기업들로 구성되어 있어, 시장 전체보다는 주로 대형주 및 산업을 대표, 종목코드 "^DJI"
- Nasdaq-100 Index : 나스닥 100 지수를 추종하는 ETF, 종목코드 "QQQ"
- Emerging Markets ETF : 중국, 인도, 브라질 등 신흥 시장의 주식에 투자하는 ETF, 종목코드 "EEM"
""")
# 선형 회귀 계수 산출
# 안전 자산, 안정적 자산
gold_lin = lin_data(gold, start_date, end_date)
tlt_lin = lin_data("TLT", start_date, end_date)
lqd_lin = lin_data("LQD", start_date, end_date)
duk_lin = lin_data("DUK", start_date, end_date)
gold_etf_lin = lin_data("GLD", start_date, end_date)
usd_lin = lin_data(usd, start_date, end_date)
# 위험 자산
wti_lin = lin_data(wti, start_date, end_date)
brent_lin = lin_data("BZ=F", start_date, end_date)
tsla_lin = lin_data("TSLA", start_date, end_date)
dj_lin = lin_data("^DJI", start_date, end_date)
nasdq_lin = lin_data("QQQ", start_date, end_date)
em_etf_lin = lin_data("EEM", start_date, end_date)

df_lin = pd.concat([gold_lin, tlt_lin, lqd_lin, duk_lin, gold_etf_lin, usd_lin, wti_lin, brent_lin, tsla_lin, dj_lin, nasdq_lin, em_etf_lin], axis = 1)
df_lin.columns = ["Gold", "US ETF", "US CP ETF", "Duke", "Gold ETF", "EUR/USD", "WTI", "Brent", "Tesla", "Dow Jones", "Nasdaq-100 Index", "Emerging Markets ETF"]
st.dataframe(df_lin.style.format("{:.10g}"))
st.write("""
- 기울기 (Slope)
- 절편 (Intercept)
- 상관계수 (R-value)
    - 1 : 완벽한 양의 상관관계. 두 변수는 완벽하게 비례해서 증가
    - 0.7 ~ 0.9 : 강한 양의 상관관계
    - 0.5 ~ 0.7 : 중간 정도의 양의 상관관계
    - 0.3 ~ 0.5 : 약한 양의 상관관계
    - 0 : 상관관계 없음
    - -0.3 ~ -0.5 : 약한 음의 상관관계
    - -0.5 ~ -0.7 : 중간 정도의 음의 상관관계
    - -0.7 ~ -0.9 : 강한 음의 상관관계
    - -1 : 완벽한 음의 상관관계. 두 변수는 완벽하게 반비례해서 변화
- 유의확률 (P-value) : 일반적으로 P-value가 0.05보다 작으면 결과가 통계적으로 유의미하다고 판단
- 표준오차 (Standard Error)
- 절편의 표준오차 (Intercept Standard Error)
""")