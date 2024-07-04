import streamlit
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import datetime
import time
from wordcloud import WordCloud

# 기간
start_date = "2014-09-01"
end_date = "2019-12-31"
news_date_list = []

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
streamlit.dataframe(bit_ticker)