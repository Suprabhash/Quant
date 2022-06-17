import itertools

import investpy
import yfinance as yf
from datetime import date,timedelta
import pandas as pd
import math
import numpy as np
import eikon as ek
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pywt
import pywt.data
from sklearn import metrics
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc)
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier)
from sklearn.tree import DecisionTreeClassifier
from decimal import Decimal
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pickle
import os
import ssl
import datetime


API_KEY = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0MWY0OGUyMC1kOTlkLTRjZTItYjc4Ny00MmMyOTI1YTVmODIifQ=="

RANDOM_STATE = 835

def get_data_investpy( symbol, country, from_date, to_date ):
  find = investpy.search.search_quotes(text=symbol, products =["stocks", "etfs", "indices", "currencies"] )
  for f in find:
    #print( f )

    if f.symbol.lower() == symbol.lower() and f.country.lower() == country.lower():
      break
  if f.symbol.lower() != symbol.lower():
    return None
  ret = f.retrieve_historical_data(from_date=from_date, to_date=to_date )
  if ret is None:
    try:
      ret = investpy.get_stock_historical_data(stock=symbol,
                                      country=country,
                                      from_date=from_date,
                                      to_date=to_date)
    except:
      ret = None
  if ret is None:
    try:
      ret = investpy.get_etf_historical_data(etf=symbol,
                                      country=country,
                                      from_date=from_date,
                                      to_date=to_date)
    except:
      ret = None

  if ret is None:
    try:
      ret = investpy.get_index_historical_data(index=symbol,
                                      country=country,
                                      from_date=from_date,
                                      to_date=to_date)
    except:
      ret = None

  if ret is None:
    try:
      ret = investpy.currency_crosses.get_currency_cross_historical_data(currency_cross=symbol,
                                      from_date=from_date,
                                      to_date=to_date)
    except:
      ret = None
  ret.drop(["Change Pct"], axis=1, inplace=True)
  return ret

def add_pivots(temp_og,col_name):

    temp_og[f"{col_name}_TypeCurrentPivot"] = np.where((temp_og[f"{col_name}"] > temp_og[f"{col_name}"].shift(1)) & (temp_og[f"{col_name}"] > temp_og[f"{col_name}"].shift(-1)), 1, np.nan)
    temp_og[f"{col_name}_TypeCurrentPivot"] = np.where((temp_og[f"{col_name}"] < temp_og[f"{col_name}"].shift(1)) & (temp_og[f"{col_name}"] < temp_og[f"{col_name}"].shift(-1)), -1, temp_og[f"{col_name}_TypeCurrentPivot"])
    temp_og[f"{col_name}_PivotValue"] = np.where((temp_og[f"{col_name}"] > temp_og[f"{col_name}"].shift(1)) & (temp_og[f"{col_name}"] > temp_og[f"{col_name}"].shift(-1)), temp_og[f"{col_name}"], np.nan)
    temp_og[f"{col_name}_PivotValue"] = np.where((temp_og[f"{col_name}"] < temp_og[f"{col_name}"].shift(1)) & (temp_og[f"{col_name}"] < temp_og[f"{col_name}"].shift(-1)), temp_og[f"{col_name}"], temp_og[f"{col_name}_PivotValue"])
    temp_og[f"{col_name}_TypePreviousPivot"] = temp_og[f"{col_name}_TypeCurrentPivot"].fillna(method='ffill').shift(1).fillna(0)
    temp_og[f"{col_name}_PreviousPivotValue"] = temp_og[f"{col_name}_PivotValue"].fillna(method='ffill').shift(1).fillna(0)
    temp_og[f"{col_name}_TypePreviousPivot"] = np.where(np.isnan(temp_og[f"{col_name}_TypeCurrentPivot"]), np.nan, temp_og[f"{col_name}_TypePreviousPivot"])
    temp_og[f"{col_name}_PreviousPivotValue"] = np.where(np.isnan(temp_og[f"{col_name}_PivotValue"]), np.nan, temp_og[f"{col_name}_PreviousPivotValue"])
    temp_og[f"{col_name}_PreviousHighPivotValueCB"] = pd.DataFrame(np.where(((temp_og[f"{col_name}_TypePreviousPivot"]==-1)),temp_og[f"{col_name}_PivotValue"],np.nan)).fillna(method='ffill').shift(1)
    temp_og[f"{col_name}_PreviousLowPivotValueCB"] = pd.DataFrame(np.where(((temp_og[f"{col_name}_TypePreviousPivot"]==1)),temp_og[f"{col_name}_PivotValue"],np.nan)).fillna(method='ffill').shift(1)
    temp_og[f"{col_name}_DaysSincePreviousHighPivot"] = np.nan
    temp_og[f"{col_name}_DaysSincePreviousLowPivot"] = np.nan
    temp_og[f"{col_name}_IsHighPivot"] = np.where(temp_og[f"{col_name}_TypeCurrentPivot"]==1,1,0)
    temp_og[f"{col_name}_IsLowPivot"] = np.where(temp_og[f"{col_name}_TypeCurrentPivot"]==-1,1,0)
    for i in range(len(temp_og)):
        if not(np.isnan(temp_og.iloc[i][f"{col_name}_TypeCurrentPivot"])):
            try:
                if len(temp_og[f"Date"][temp_og[f"{col_name}_PivotValue"]==temp_og.iloc[i][f"{col_name}_PreviousPivotValue"]]) == 1:
                    temp_og.loc[i, f"{col_name}_DaysSincePreviousHighPivot"] = (temp_og.iloc[i][f"Date"] - temp_og[f"Date"][temp_og[f"{col_name}_PivotValue"] == temp_og.iloc[i][f"{col_name}_PreviousHighPivotValueCB"]].iloc[0]).days ##
                    temp_og.loc[i, f"{col_name}_DaysSincePreviousLowPivot"] = (temp_og.iloc[i][f"Date"] - temp_og[f"Date"][temp_og[f"{col_name}_PivotValue"] == temp_og.iloc[i][f"{col_name}_PreviousLowPivotValueCB"]].iloc[0]).days  ##
            except:
                continue
        else:
            try:
                temp_og.loc[i, f"{col_name}_DaysSincePreviousHighPivot"] = temp_og.loc[i-1, f"{col_name}_DaysSincePreviousHighPivot"] + 1
                temp_og.loc[i, f"{col_name}_DaysSincePreviousLowPivot"] = temp_og.loc[i-1, f"{col_name}_DaysSincePreviousLowPivot"] + 1
            except:
                continue

    #temp_og[f"{col_name}_TypeCurrentPivot"] = temp_og[f"{col_name}_TypeCurrentPivot"].shift(1)
    #temp_og[f"{col_name}_PivotValue"] = temp_og[f"{col_name}_PivotValue"].shift(1)
    #temp_og[f"{col_name}_TypePreviousPivot"] = temp_og[f"{col_name}_TypePreviousPivot"].shift(1)
    #temp_og[f"{col_name}_PreviousPivotValue"] = temp_og[f"{col_name}_PreviousPivotValue"].shift(1)
    temp_og[f"{col_name}_PreviousHighPivotValueCB"] = temp_og[f"{col_name}_PreviousHighPivotValueCB"].shift(1)
    temp_og[f"{col_name}_PreviousLowPivotValueCB"] = temp_og[f"{col_name}_PreviousLowPivotValueCB"].shift(1)
    temp_og[f"{col_name}_DaysSincePreviousHighPivot"] = temp_og[f"{col_name}_DaysSincePreviousHighPivot"].shift(1)
    temp_og[f"{col_name}_DaysSincePreviousLowPivot"] = temp_og[f"{col_name}_DaysSincePreviousLowPivot"].shift(1)
    temp_og[f"{col_name}_IsHighPivot"] = temp_og[f"{col_name}_IsHighPivot"].shift(1)
    temp_og[f"{col_name}_IsLowPivot"] = temp_og[f"{col_name}_IsLowPivot"].shift(1)
    # window = 10
    # th_high = -1
    # th_low = 1
    # num_stdev = 1
    #
    # temp_og[f"{col_name}RMean"] = temp_og[f"{col_name}"].rolling(window=window).mean()
    # temp_og[f"{col_name}RStDev"] = temp_og[f"{col_name}"].rolling(window=window).std()

    # temp_og[f"{col_name}PreviousPivotVal"] = np.where(abs(temp_og[f"{col_name}If{col_name}HighPivot"]) > 0,temp_og[f"{col_name}If{col_name}HighPivot"],temp_og[f"{col_name}If{col_name}LowPivot"])
    # temp_og[f"{col_name}PreviousPivotVal"] = temp_og[f"{col_name}PreviousPivotVal"].fillna(method='ffill').shift(1).fillna(0)

    # temp_og[f"{col_name}If{col_name}HighPivot"] = np.where((abs(temp_og[f"{col_name}If{col_name}HighPivot"])/abs(temp_og[f"{col_name}PreviousPivotVal"])-1>th_high)&(temp_og[f"{col_name}If{col_name}HighPivot"]>(temp_og[f"{col_name}RMean"]+num_stdev*temp_og[f"{col_name}RStDev"])), temp_og[f"{col_name}If{col_name}HighPivot"], np.nan)
    # temp_og[f"{col_name}If{col_name}LowPivot"] = np.where((abs(temp_og[f"{col_name}If{col_name}LowPivot"]) / abs(temp_og[f"{col_name}PreviousPivotVal"]) - 1 < th_low)&(temp_og[f"{col_name}If{col_name}LowPivot"]<(temp_og[f"{col_name}RMean"]-num_stdev*temp_og[f"{col_name}RStDev"])), temp_og[f"{col_name}If{col_name}LowPivot"], np.nan)
    # temp_og[f"Is{col_name}HighPivot"] = np.where((abs(temp_og[f"{col_name}If{col_name}HighPivot"]) / abs(temp_og[f"{col_name}PreviousPivotVal"]) - 1 > th_high)&(temp_og[f"{col_name}If{col_name}HighPivot"]>(temp_og[f"{col_name}RMean"]+num_stdev*temp_og[f"{col_name}RStDev"])),1, np.nan)
    # temp_og[f"Is{col_name}LowPivot"] = np.where((abs(temp_og[f"{col_name}If{col_name}LowPivot"]) / abs(temp_og[f"{col_name}PreviousPivotVal"]) - 1 < th_low)&(temp_og[f"{col_name}If{col_name}LowPivot"]<(temp_og[f"{col_name}RMean"]-num_stdev*temp_og[f"{col_name}RStDev"])),1, np.nan)

    return temp_og

def add_maxmin(temp_og, col_name,periods):
    for period in periods:
        temp_og[f"Max_{col_name}_{period}"] = temp_og[col_name].rolling(window = period).max()
        temp_og[f"Min_{col_name}_{period}"] = temp_og[col_name].rolling(window=period).min()

    return temp_og

def add_derivatives(temp_og, col_name, periods):
    for period in periods:
        temp_og[f"ROC_{col_name}_{period}"] = temp_og[col_name].pct_change(periods=period)
    if (col_name == "FMACB") | (col_name == "SMACB"):
        temp_og[f"Convexity_{col_name}"] = temp_og[col_name].pct_change().pct_change()
    return temp_og

def add_forward_return(temp_og, period):
    temp_og["FReturn"] = temp_og["Close"].shift(-period)/temp_og["Close"]-1
    temp_og["BinaryOutcome"]  = temp_og["FReturn"].apply(np.sign)
    return temp_og

def get_data_BTC():
    filepath = "https://www.cryptodatadownload.com/cdd/Bitstamp_BTCUSD_d.csv"
    ssl._create_default_https_context = ssl._create_unverified_context
    temp_og = pd.read_csv(filepath, parse_dates=True, skiprows=1)
    temp_og.drop(columns=["date", "symbol", "Volume BTC"], inplace=True)
    temp_og["unix"] = temp_og["unix"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000) if len(
        str(int(x))) == 13 else datetime.datetime.fromtimestamp(x))
    temp_og.rename(columns={"unix": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close",
                            "Volume USD": "Volume"}, inplace=True)
    temp_og["Date"] = pd.to_datetime(temp_og["Date"])
    temp_og.sort_values("Date", ascending=True, inplace=True)
    temp_og.reset_index(drop=True, inplace=True)

    if os.path.isdir('BTC_D.pkl'):
        with open(f'BTC_D.pkl', 'rb') as file:
            temp_og_imp = pickle.load(file)
        temp_og = pd.concat([temp_og_imp, temp_og], axis=0)
        temp_og.drop_duplicates(keep="first", inplace=True)
        temp_og.reset_index(drop=True, inplace=True)
        with open(f'BTC_D.pkl', 'wb') as file:
            pickle.dump(pd.DataFrame(temp_og), file)
    else:
        with open(f'BTC_D.pkl', 'wb') as file:
            pickle.dump(pd.DataFrame(temp_og), file)

    filepath = "https://www.cryptodatadownload.com/cdd/Bitstamp_BTCUSD_1h.csv"
    ssl._create_default_https_context = ssl._create_unverified_context
    temp_og1 = pd.read_csv(filepath, parse_dates=True, skiprows=1)
    temp_og1.drop(columns=["date", "symbol", "Volume BTC"], inplace=True)
    temp_og1["unix"] = temp_og1["unix"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000) if len(
        str(int(x))) == 13 else datetime.datetime.fromtimestamp(x))
    temp_og1.rename(columns={"unix": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close",
                             "Volume USD": "Volume"}, inplace=True)
    temp_og1["Date"] = pd.to_datetime(temp_og1["Date"])
    temp_og1.Date = temp_og1.Date.dt.tz_localize('Asia/Kolkata')
    temp_og1.sort_values("Date", ascending=True, inplace=True)
    temp_og1.reset_index(drop=True, inplace=True)

    if os.path.isdir('BTC_H.pkl'):
        with open(f'BTC_H.pkl', 'rb') as file:
            temp_og1_imp = pickle.load(file)
        temp_og1 = pd.concat([temp_og1_imp, temp_og1], axis=0)
        temp_og1.drop_duplicates(keep="first", inplace=True)
        temp_og1.reset_index(drop=True, inplace=True)
        with open(f'BTC_H.pkl', 'wb') as file:
            pickle.dump(pd.DataFrame(temp_og1), file)
    else:
        with open(f'BTC_H.pkl', 'wb') as file:
            pickle.dump(pd.DataFrame(temp_og1), file)

    return temp_og, temp_og1

def get_data(ticker, api, add_features=True):

    if api == "yfinance":
        temp_og = yf.download(ticker, start = '2007-01-01', end= str(date.today()+timedelta(1)))
        temp_og.reset_index(inplace=True)
        temp_og.drop(['Adj Close'], axis=1, inplace=True)
        if ticker=="GOLDBEES.NS":
            temp_og = temp_og.loc[temp_og["Close"]>1]

    if api =="investpy":
        temp_og = get_data_investpy(symbol=ticker, country='india', from_date="01/01/2007",to_date=(date.today()+timedelta(1)).strftime("%d/%m/%Y"))
        temp_og.reset_index(inplace=True)

    if api == "reuters":
        temp_og = ek.get_timeseries(ticker, start_date='2007-01-01', end_date=str(date.today() + timedelta(1)))
        temp_og.reset_index(inplace=True)
        temp_og.rename(columns={"HIGH": "High", "CLOSE": "Close", "LOW": "Low", "OPEN": "Open", "VOLUME": "Volume"},
                       inplace=True)
        temp_og.drop(['COUNT'], axis=1, inplace=True)

    if add_features:
        temp_og = constance_brown(temp_og)
        temp_og = add_pivots(temp_og,"CB")
        temp_og = add_pivots(temp_og, "FMACB")
        temp_og = add_pivots(temp_og, "SMACB")
        temp_og = add_maxmin(temp_og, "CB", [10,20,30,60])
        temp_og = add_maxmin(temp_og, "FMACB", [10,20,30,60])
        temp_og = add_maxmin(temp_og, "SMACB", [10,20,30,60])
        temp_og = add_derivatives(temp_og, "CB", [10, 20, 30, 60])
        temp_og = add_derivatives(temp_og, "FMACB", [10, 20, 30, 60])
        temp_og = add_derivatives(temp_og, "SMACB", [10, 20, 30, 60])
        temp_og = add_forward_return(temp_og, 1)

    return temp_og

def get_data_BTC():
    filepath = "https://www.cryptodatadownload.com/cdd/Bitstamp_BTCUSD_d.csv"
    ssl._create_default_https_context = ssl._create_unverified_context
    temp_og = pd.read_csv(filepath, parse_dates=True, skiprows=1)
    temp_og.drop(columns=["date", "symbol", "Volume BTC"], inplace=True)
    temp_og["unix"] = temp_og["unix"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000) if len(
        str(int(x))) == 13 else datetime.datetime.fromtimestamp(x))
    temp_og.rename(columns={"unix": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close",
                            "Volume USD": "Volume"}, inplace=True)
    temp_og["Date"] = pd.to_datetime(temp_og["Date"])
    temp_og.sort_values("Date", ascending=True, inplace=True)
    temp_og.reset_index(drop=True, inplace=True)

    if os.path.isdir('BTC_D.pkl'):
        with open(f'BTC_D.pkl', 'rb') as file:
            temp_og_imp = pickle.load(file)
        temp_og = pd.concat([temp_og_imp, temp_og], axis=0)
        temp_og.drop_duplicates(keep="first", inplace=True)
        temp_og.reset_index(drop=True, inplace=True)
        with open(f'BTC_D.pkl', 'wb') as file:
            pickle.dump(pd.DataFrame(temp_og), file)
    else:
        with open(f'BTC_D.pkl', 'wb') as file:
            pickle.dump(pd.DataFrame(temp_og), file)

    # filepath = "https://www.cryptodatadownload.com/cdd/Bitstamp_BTCUSD_1h.csv"
    # ssl._create_default_https_context = ssl._create_unverified_context
    # temp_og1 = pd.read_csv(filepath, parse_dates=True, skiprows=1)
    # temp_og1.drop(columns=["date", "symbol", "Volume BTC"], inplace=True)
    # temp_og1["unix"] = temp_og1["unix"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000) if len(
    #     str(int(x))) == 13 else datetime.datetime.fromtimestamp(x))
    # temp_og1.rename(columns={"unix": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close",
    #                          "Volume USD": "Volume"}, inplace=True)
    # temp_og1["Date"] = pd.to_datetime(temp_og1["Date"])
    # temp_og1.Date = temp_og1.Date.dt.tz_localize('Asia/Kolkata')
    # temp_og1.sort_values("Date", ascending=True, inplace=True)
    # temp_og1.reset_index(drop=True, inplace=True)
    #
    # if os.path.isdir('BTC_H.pkl'):
    #     with open(f'BTC_H.pkl', 'rb') as file:
    #         temp_og1_imp = pickle.load(file)
    #     temp_og1 = pd.concat([temp_og1_imp, temp_og1], axis=0)
    #     temp_og1.drop_duplicates(keep="first", inplace=True)
    #     temp_og1.reset_index(drop=True, inplace=True)
    #     with open(f'BTC_H.pkl', 'wb') as file:
    #         pickle.dump(pd.DataFrame(temp_og1), file)
    # else:
    #     with open(f'BTC_H.pkl', 'wb') as file:
    #         pickle.dump(pd.DataFrame(temp_og1), file)

    temp_og = constance_brown(temp_og)
    temp_og = add_pivots(temp_og, "CB")
    temp_og = add_pivots(temp_og, "FMACB")
    temp_og = add_pivots(temp_og, "SMACB")
    temp_og = add_maxmin(temp_og, "CB", [10, 20, 30, 60])
    temp_og = add_maxmin(temp_og, "FMACB", [10, 20, 30, 60])
    temp_og = add_maxmin(temp_og, "SMACB", [10, 20, 30, 60])
    temp_og = add_derivatives(temp_og, "CB", [10, 20, 30, 60])
    temp_og = add_derivatives(temp_og, "FMACB", [10, 20, 30, 60])
    temp_og = add_derivatives(temp_og, "SMACB", [10, 20, 30, 60])
    temp_og = add_forward_return(temp_og, 1)

    # temp_og1 = constance_brown(temp_og1)
    # temp_og1 = add_pivots(temp_og1, "CB")
    # temp_og1 = add_pivots(temp_og1, "FMACB")
    # temp_og1 = add_pivots(temp_og1, "SMACB")
    # temp_og1 = add_maxmin(temp_og1, "CB", [10, 20, 30, 60])
    # temp_og1 = add_maxmin(temp_og1, "FMACB", [10, 20, 30, 60])
    # temp_og1 = add_maxmin(temp_og1, "SMACB", [10, 20, 30, 60])
    # temp_og1 = add_derivatives(temp_og1, "CB", [10, 20, 30, 60])
    # temp_og1 = add_derivatives(temp_og1, "FMACB", [10, 20, 30, 60])
    # temp_og1 = add_derivatives(temp_og1, "SMACB", [10, 20, 30, 60])
    # temp_og1 = add_forward_return(temp_og1, 1)

    return temp_og#, temp_og1

def RSI(data_df,period):
    series = data_df['Close']

    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period - 1]] = np.mean(u[:period])  # first value is sum of avg gains
    u = u.drop(u.index[:(period - 1)])
    d[d.index[period - 1]] = np.mean(d[:period])  # first value is sum of avg losses
    d = d.drop(d.index[:(period - 1)])
    rs = pd.DataFrame.ewm(u, com=period - 1, adjust=False).mean() / \
         pd.DataFrame.ewm(d, com=period - 1, adjust=False).mean()
    rsi = (100 - 100 / (1 + rs))
    data_df = pd.concat([data_df, rsi], axis=1)
    rsi_df = (data_df.iloc[:, -1:])
    rsi_df.columns = ['RSI']
    return rsi_df.fillna(0).iloc[:,0]

def constance_brown(data_df):
  r=RSI(data_df,14)
  rsi_mom_length=9
  ma_length=3
  rsi_ma_length=3
  fastLength=13
  slowLength=33
  rsidelta=[0]*len(data_df)

  for i in range(len(data_df)):
    if i<rsi_mom_length:
        rsidelta[i] = np.nan
    else:
        rsidelta[i]=r[i]-r[i-rsi_mom_length]

  rsisigma=RSI(data_df,rsi_ma_length).rolling(window=ma_length).mean()
  rsidelta = [0 if math.isnan(x) else x for x in rsidelta]
  s=[0]*len(data_df)
  for i in range(len(rsidelta)):
    s[i]=rsidelta[i]+rsisigma[i]
  #s = [0 if math.isnan(x) else x for x in s]

  data_df["CB"] = s
  data_df["FMACB"] = pd.DataFrame(s).rolling(window=fastLength).mean()
  data_df["SMACB"] = pd.DataFrame(s).rolling(window=slowLength).mean()
  return data_df

def plot_price_volume(temp_og, start_date, end_date):
    data = temp_og[(temp_og["Date"] >= start_date) & (temp_og["Date"] < end_date)]
    fig = make_subplots(rows=2, cols=1, row_heights=[1.5, 0.5],
                        specs=[[{"secondary_y": True}], [{"secondary_y": True}]])

    fig.update_layout(xaxis2={'anchor': 'y', 'overlaying': 'x', 'side': 'top'},
                      yaxis_domain=[0, 1])

    fig.add_trace(go.Histogram(name="Volume Profile", x=data['Volume'], y=data['Close'], nbinsy=20, orientation='h'),
                  row=1, col=1)
    fig.add_trace(go.Candlestick(name="BJFS", x=data['Date'],
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close']), row=1, col=1)


    fig.data[1].update(xaxis='x2')
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(template="plotly_dark", width=1000, height=500)
    fig.show()

def plot_price_volume_cb(temp_og, start_date, end_date):
    data = temp_og[(temp_og["Date"] >= start_date) & (temp_og["Date"] < end_date)]
    fig = make_subplots(rows=2, cols=1, row_heights=[1.5, 0.5],
                        specs=[[{"secondary_y": True}], [{"secondary_y": True}]])

    fig.update_layout(xaxis2={'anchor': 'y', 'overlaying': 'x', 'side': 'top'},
                      yaxis_domain=[0, 1])

    fig.add_trace(go.Histogram(name="Volume Profile", x=data['Volume'], y=data['Close'], nbinsy=20, orientation='h'),
                  row=1, col=1)
    fig.add_trace(go.Candlestick(name=".NSEI", x=data['Date'],
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close']), row=1, col=1)

    fig.add_trace(go.Scatter(x=data["Date"], y=data["CB"], name="Constance-Brown Index"), row=2, col=1,
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=data["Date"], y=data["FMACB"], name="FMA Constance-Brown Index"), row=2, col=1,
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=data["Date"], y=data["SMACB"], name="SMA Constance-Brown Index"), row=2, col=1,
                  secondary_y=False)

    fig.data[1].update(xaxis='x2')
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(template="plotly_dark", width=1000, height=500)
    fig.show()

def plot_price_cb(data):
    fig = make_subplots(rows=2, cols=1, row_heights=[1, 1], shared_xaxes=True,shared_yaxes=True,
                        specs=[[{"secondary_y": True}], [{"secondary_y": True}]])

    fig.update_layout(xaxis2={'anchor': 'y', 'overlaying': 'x', 'side': 'top'},
                      yaxis_domain=[0, 1])

    fig.add_trace(go.Candlestick(name=".NSEI", x=data['Date'],
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close']), row=1, col=1)

    fig.add_trace(go.Scatter(x=data["Date"], y=data["CB"], name="Constance-Brown Index", mode='lines',marker=dict(color='darkorchid')), row=2,
                  col=1,
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=data["Date"], y=data["FMACB"], name="FMA Constance-Brown Index", mode='lines',marker=dict(color='orange')), row=2,
                  col=1,
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=data["Date"], y=data["SMACB"], name="SMA Constance-Brown Index", mode='lines',marker=dict(color='aquamarine')), row=2,
                  col=1,
                  secondary_y=False)

    fig.add_trace(go.Scatter(x=data["Date"][data[f"CB_TypeCurrentPivot"]==1], y=data[f"CB_PivotValue"][data[f"CB_TypeCurrentPivot"]==1], name="High Pivot Constance-Brown Index", mode='markers',marker=dict(
            color='lime',
            size=6,
            line=dict(
                color='darkorchid',
                width=1.5
            )
        )), row=2,
                  col=1,
                  secondary_y=False)
    fig.add_trace(
        go.Scatter(x=data["Date"][data[f"CB_TypeCurrentPivot"]==-1], y=data[f"CB_PivotValue"][data[f"CB_TypeCurrentPivot"]==-1], name="Low Pivot Constance-Brown Index", mode='markers',marker=dict(
            color='red',
            size=6,
            line=dict(
                color='darkorchid',
                width=1.5
            )
        )), row=2,
        col=1,
        secondary_y=False)
    fig.add_trace(
        go.Scatter(x=data["Date"][data[f"FMACB_TypeCurrentPivot"]==1], y=data[f"FMACB_PivotValue"][data[f"FMACB_TypeCurrentPivot"]==1], name="High Pivot FMA Constance-Brown Index", mode='markers',marker=dict(
            color='lime',
            size=6,
            line=dict(
                color='orange',
                width=1.5
            )
        )), row=2,
        col=1,
        secondary_y=False)
    fig.add_trace(
        go.Scatter(x=data["Date"][data[f"FMACB_TypeCurrentPivot"]==-1], y=data[f"FMACB_PivotValue"][data[f"FMACB_TypeCurrentPivot"]==-1], name="Low Pivot FMA Constance-Brown Index", mode='markers',marker=dict(
            color='red',
            size=6,
            line=dict(
                color='orange',
                width=1.5
            )
        )), row=2,
        col=1,
        secondary_y=False)
    fig.add_trace(
        go.Scatter(x=data["Date"][data[f"SMACB_TypeCurrentPivot"]==1], y=data[f"SMACB_PivotValue"][data[f"SMACB_TypeCurrentPivot"]==1], name="High Pivot SMA Constance-Brown Index", mode='markers',marker=dict(
            color='lime',
            size=6,
            line=dict(
                color='aquamarine',
                width=1.5
            )
        )), row=2,
        col=1,
        secondary_y=False)
    fig.add_trace(
        go.Scatter(x=data["Date"][data[f"SMACB_TypeCurrentPivot"]==-1], y=data[f"SMACB_PivotValue"][data[f"SMACB_TypeCurrentPivot"]==-1], name="Low Pivot SMA Constance-Brown Index", mode='markers',marker=dict(
            color='red',
            size=6,
            line=dict(
                color='aquamarine',
                width=1.5
            )
        )), row=2,
        col=1,
        secondary_y=False)


    fig.data[1].update(xaxis='x2')
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(template="plotly_dark", width=1500, height=1000)
    fig.show()

def plot_multiclass_roc(y_test_, y_pred, classes, ax):
    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test__dummies = pd.get_dummies(y_test_, drop_first=False).values
    for i, label in zip(range(len(classes)), classes):
        fpr[i], tpr[i], _ = roc_curve(y_test__dummies[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    for i, label in zip(range(len(classes)), classes):
        ax.plot(
            fpr[i],
            tpr[i],
            label="ROC curve (area = %0.2f) for label %i" % (roc_auc[i], label),
        )
    ax.legend(loc="best")
    ax.set_title("ROC-AUC")
    # sns.despine()
    return

def plot_multiclass_precision_recall_curve(y_test_, y_pred, classes, ax):
    # structures
    fpr = dict()
    tpr = dict()
    # pr_auc = dict()

    # precision recall curve
    precision = dict()
    recall = dict()

    # calculate dummies once
    y_test__dummies = pd.get_dummies(y_test_, drop_first=False).values
    for i, label in zip(range(len(classes)), classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_test__dummies[:, i], y_pred[:, i]
        )
        ax.plot(recall[i], precision[i], lw=2, label=f"class {label}")

    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.legend(loc="best")
    ax.set_title("precision vs. recall curve")
    return

def performance_evaluation_report_multiclass(model,X_test_, y_test_,show_plot=False,labels=None,show_pr_curve=False,custom_threshold=None,average=None):
    """
    Function for creating a performance report of a classification model.

    Parameters
    ----------
    model : scikit-learn estimator
        A fitted estimator for classification problems.
    X_test_ : pd.DataFrame
        DataFrame with features matching y_test_
    y_test_ : array/pd.Series
        Target of a classification problem.
    show_plot : bool
        Flag whether to show the plot
    labels : list
        List with the class names.
    show_pr_curve : bool
        Flag whether to also show the PR-curve. For this to take effect,
        show_plot must be True.

    Return
    ------
    stats : pd.Series
        A series with the most important evaluation metrics
    """

    if custom_threshold is None:  # default is 50%
        y_pred = model.predict(X_test_)
    else:
        # TODO UPDATE FOR THE MULTICLASS CASE
        threshold = 0.5
        y_pred = (model.predict_proba(X_test_)[:, 1] > threshold).astype(int)
        y_pred = np.where(y_pred == 0, -1, 1)

    y_pred_prob = model.predict_proba(X_test_)  # [:, 1]

    conf_mat = metrics.confusion_matrix(y_test_, y_pred)
    # REF:
    # https://stackoverflow.com/questions/50666091/true-positive-rate-and-false-positive-rate-tpr-fpr-for-multi-class-data-in-py
    fp = conf_mat.sum(axis=0) - np.diag(conf_mat)
    fn = conf_mat.sum(axis=1) - np.diag(conf_mat)
    tp = np.diag(conf_mat)
    tn = conf_mat.sum() - (fp + fn + tp)
    fp = fp.astype(float)
    fn = fn.astype(float)
    tp = tp.astype(float)
    tn = tn.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    tpr = tp / (tp + fn)
    # Specificity or true negative rate
    tnr = tn / (tn + fp)
    # Precision or positive predictive value
    ppv = tp / (tp + fp)
    # Negative predictive value
    npv = tn / (tn + fn)
    # Fall out or false positive rate
    fpr = fp / (fp + tn)
    # False negative rate
    fnr = fn / (tp + fn)
    # False discovery rate
    fdr = fp / (tp + fp)
    # Overall accuracy
    acc = (tp + tn) / (tp + fp + fn + tn)

    precision = (metrics.precision_score(y_test_, y_pred, average=average),)
    recall = (metrics.recall_score(y_test_, y_pred, average=average),)

    if show_plot:

        if labels is None:
            labels = ["Negative", "Positive"]

        N_SUBPLOTS = 3 if show_pr_curve else 2
        N_SUBPLOT_ROWS = 1 if show_pr_curve else 1
        PLOT_WIDTH = 17 if show_pr_curve else 12
        PLOT_HEIGHT = 10 if show_pr_curve else 6

        fig = plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT), tight_layout=True)
        gs = gridspec.GridSpec(N_SUBPLOT_ROWS, N_SUBPLOTS)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        # ax3 = fig.add_subplot(gs[1, 1])

        fig.suptitle("Performance Evaluation", fontsize=16, y=1.05)

        total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
        normed_conf_mat = conf_mat.astype("float") / total_samples

        text_array = np.empty_like(conf_mat, dtype="object")
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                norm_val = normed_conf_mat[i, j]
                int_val = conf_mat[i, j]
                text_array[i, j] = f"({norm_val:.1%})\n{int_val}"

        g = sns.heatmap(
            conf_mat,
            annot=text_array,
            fmt="s",
            linewidths=0.5,
            cmap="Blues",
            square=True,
            cbar=False,
            ax=ax0,
            annot_kws={"ha": "center", "va": "center"},
        )

        ax0.set(
            xlabel="Predicted label", ylabel="Actual label", title="Confusion Matrix"
        )
        ax0.xaxis.set_ticklabels(labels)
        ax0.yaxis.set_ticklabels(labels)

        _ = plot_multiclass_roc(y_test_, y_pred_prob, labels, ax1)
        ax1.plot(
            fp / (fp + tn), tp / (tp + fn), "ro", markersize=8, label="Decision Point"
        )

        if show_pr_curve:
            _ = plot_multiclass_precision_recall_curve(
                y_test_, y_pred_prob, labels, ax2
            )

    stats = {
        "accuracy": np.round(acc, 4),
        "precision": np.round(ppv, 4),
        "recall": np.round(tpr, 4),
        "mcc": round(metrics.matthews_corrcoef(y_test_, y_pred), 4),
        "specificity": np.round(tnr, 4),
        "f1_score": np.round(metrics.f1_score(y_test_, y_pred, average=average), 4),
        "cohens_kappa": round(metrics.cohen_kappa_score(y_test_, y_pred), 4),
        # "roc_auc": round(roc_auc, 4),
        # "pr_auc": round(pr_auc, 4),
    }
    return stats

def run_rf_model(X_train_, y_train_, X_test_, y_test_, classes,params=None):
    if not params:
        rf_clf_ = RandomForestClassifier(
            criterion="entropy",
            max_depth=18,
            class_weight="balanced_subsample",
            n_estimators=1500,
            random_state=RANDOM_STATE,
            oob_score=True,
            n_jobs=-1,
        )
    else:
        rf_clf_ = RandomForestClassifier(
            criterion="entropy",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            oob_score=True,
            n_jobs=-1,
            **params
        )

    rf_clf_.fit(X_train_, y_train_)

    rf_clf_perf = performance_evaluation_report_multiclass(
        rf_clf_,
        X_test_,
        y_test_,
        show_plot=True,
        show_pr_curve=True,
        average=None,
        labels=classes,
    )
    print(rf_clf_perf)
    return rf_clf_, rf_clf_perf

def run_bag_model(X_train_, y_train_, X_test_, y_test_,classes):
    clf_base = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,
        class_weight="balanced",
        min_weight_fraction_leaf=0,
        random_state=RANDOM_STATE,
    )
    bag_clf = BaggingClassifier(
        base_estimator=clf_base,
        n_estimators=2000,
        max_features=0.05,
        max_samples=0.05,
        oob_score=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    bag_clf.fit(X_train_, y_train_)
    bag_clf_perf = performance_evaluation_report_multiclass(
        bag_clf,
        X_test_,
        y_test_,
        show_plot=True,
        show_pr_curve=True,
        average=None,
        labels=classes,
    )
    print(bag_clf_perf)
    return bag_clf, bag_clf_perf

def run_gbm_model(X_train_, y_train_, X_test_, y_test_, classes,best_params=None):
    if best_params is None:
        gbm = lgb.LGBMClassifier(
            boosting_type="dart",
            learning_rate=0.15,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            n_estimators=2000,
            silent=True,
            n_jobs=-1,
            random_state=78,
            # predict_disable_shape_check=True,
        )
    else:
        gbm = lgb.LGBMClassifier(silent=True, n_jobs=-1, random_state=78, **best_params)

    gbm.fit(X_train_, y_train_)
    gbm_perf = performance_evaluation_report_multiclass(
        gbm,
        X_test_,
        y_test_,
        show_plot=True,
        show_pr_curve=True,
        average=None,
        labels=classes,
    )
    print(gbm_perf)
    return gbm, gbm_perf

def format_results(clf_perf):
    import copy

    out_perf = copy.copy(clf_perf)
    out_perf["accuracy"] = quantize(clf_perf["accuracy"].mean(), 4)
    out_perf["f1_score"] = quantize(clf_perf["f1_score"].mean(), 4)
    out_perf["precision"] = quantize(clf_perf["precision"].mean(), 4)
    out_perf["recall"] = quantize(clf_perf["recall"].mean(), 4)
    out_perf["specificity"] = quantize(clf_perf["specificity"].mean(), 4)
    return out_perf

def prepare_clustering_dataframe(df_train, df_train_unscaled, how):
    df_cluster = df_train.copy()

    if how == "UpsideMaxAll":
        for i in range(1,31):
            df_train_unscaled[f"FReturn{i}"] = df_train_unscaled["Close"].shift(-i)/df_train_unscaled["Close"]-1
        df_train_unscaled[f"FReturn"] = df_train_unscaled[[f"FReturn{i}" for i in range(1,31)]].max( axis=1)
        df_train_unscaled = df_train_unscaled.drop(columns=[f"FReturn{i}" for i in range(1,30)])
        df_train_unscaled.dropna(inplace=True)
        df_cluster = pd.concat([df_cluster, df_train_unscaled[["FReturn","FReturn30"]]], axis=1)

    if how == "UpsideMax":
        for i in range(1,31):
            df_train_unscaled[f"FReturn{i}"] = df_train_unscaled["Close"].shift(-i)/df_train_unscaled["Close"]-1
        df_train_unscaled[f"FReturn"] = df_train_unscaled[[f"FReturn{i}" for i in range(1,31)]].max( axis=1)
        df_train_unscaled = df_train_unscaled.drop(columns=[f"FReturn{i}" for i in range(1,30)])
        df_train_unscaled.dropna(inplace=True)
        df_train_unscaled = rolling_percentile_freturns(df_train_unscaled, 300)
        df_cluster = pd.concat([df_cluster, df_train_unscaled[["FReturn", "FReturn30", "Percentile"]]], axis=1)
        df_cluster = df_cluster[df_cluster.Percentile>75]
        df_cluster = df_cluster.drop(columns=["Percentile"])

    if how == "Upside":
        df_train_unscaled[f"FReturn"] = df_train_unscaled["Close"].shift(-30)/df_train_unscaled["Close"] - 1
        df_train_unscaled.dropna(inplace=True)
        df_train_unscaled = rolling_percentile_freturns(df_train_unscaled, 300)
        df_cluster = pd.concat([df_cluster, df_train_unscaled[["FReturn", "Percentile"]]], axis=1)
        df_cluster = df_cluster[df_cluster.Percentile > 75]
        df_cluster = df_cluster.drop(columns=["Percentile"])

    if how == "DownsideMinAll":
        for i in range(1,31):
            df_train_unscaled[f"FReturn{i}"] = df_train_unscaled["Close"].shift(-i)/df_train_unscaled["Close"]-1
        df_train_unscaled[f"FReturn"] = df_train_unscaled[[f"FReturn{i}" for i in range(1,31)]].min( axis=1)
        df_train_unscaled = df_train_unscaled.drop(columns=[f"FReturn{i}" for i in range(1,30)])
        df_train_unscaled.dropna(inplace=True)
        df_cluster = pd.concat([df_cluster, df_train_unscaled[["FReturn","FReturn30"]]], axis=1)

    if how == "DownsideMin":
        for i in range(1,31):
            df_train_unscaled[f"FReturn{i}"] = df_train_unscaled["Close"].shift(-i)/df_train_unscaled["Close"]-1
        df_train_unscaled[f"FReturn"] = df_train_unscaled[[f"FReturn{i}" for i in range(1,31)]].min( axis=1)
        df_train_unscaled = df_train_unscaled.drop(columns=[f"FReturn{i}" for i in range(1,30)])
        df_train_unscaled.dropna(inplace=True)
        df_train_unscaled = rolling_percentile_freturns(df_train_unscaled, 300)
        df_cluster = pd.concat([df_cluster, df_train_unscaled[["FReturn", "FReturn30", "Percentile"]]], axis=1)
        df_cluster = df_cluster[df_cluster.Percentile < 25]
        df_cluster = df_cluster.drop(columns=["Percentile"])

    if how == "Downside":
        df_train_unscaled[f"FReturn"] = c
        df_train_unscaled.dropna(inplace=True)
        df_train_unscaled = rolling_percentile_freturns(df_train_unscaled, 300)
        df_cluster = pd.concat([df_cluster, df_train_unscaled[["FReturn", "Percentile"]]], axis=1)
        df_cluster = df_cluster[df_cluster.Percentile < 25]
        df_cluster = df_cluster.drop(columns=["Percentile"])

    df_cluster = df_cluster.drop(columns=["BinaryOutcome", "Open", "High", "Low", "Close", 'Max_CB_20', 'Min_CB_20',
           'Max_CB_30', 'Min_CB_30', 'Max_CB_60', 'Min_CB_60', 'Max_FMACB_20', 'Min_FMACB_20', 'Max_FMACB_30',
           'Min_FMACB_30', 'Max_FMACB_60', 'Min_FMACB_60', 'Max_SMACB_20', 'Min_SMACB_20', 'Max_SMACB_30',
           'Min_SMACB_30', 'Max_SMACB_60', 'Min_SMACB_60',
           'ROC_CB_20', 'ROC_CB_30', 'ROC_CB_60', 'ROC_FMACB_20',
           'ROC_FMACB_30', 'ROC_FMACB_60', 'Convexity_FMACB',
           'ROC_SMACB_20', 'ROC_SMACB_30', 'ROC_SMACB_60'])

    return df_cluster

def elbow_test(df_cluster, selected_features):
    df_cluster_input = df_cluster[selected_features]
    X = StandardScaler().fit_transform(df_cluster_input)
    sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        k_means = KMeans(n_clusters=k)
        model = k_means.fit(X)
        sum_of_squared_distances.append(k_means.inertia_)
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('sum_of_squared_distances')
    plt.title('elbow method for optimal k')
    plt.show()

def cluster_grid_search(df_cluster):
    results = []
    all_features = list(df_cluster.columns)
    all_features.remove('FReturn')
    all_features.remove('FReturn30')
    for num_features in [3,4,5]:
        selected_features = list(itertools.combinations(all_features, num_features))
        for i in range(len(selected_features)):
            for num_clusters in [5,6,7]:
                df_cluster_input = df_cluster[list(selected_features[i])]
                X = df_cluster_input.values
                k_means = KMeans(n_clusters=num_clusters, random_state = RANDOM_STATE)
                model = k_means.fit(X)
                labels = k_means.labels_
                df_cluster_input = pd.concat([df_cluster_input.reset_index(), df_cluster["FReturn"].reset_index()["FReturn"],pd.DataFrame(labels).rename({0: 'labels'}, axis=1)], axis=1).set_index("Date")
                stdevs = []
                for label in range(num_clusters):
                    if len(df_cluster_input[df_cluster_input["labels"] == label]) > 0:
                        stdevs.append(df_cluster_input[df_cluster_input["labels"] == label]["FReturn"].std(axis=0))
                    else:
                        stdevs.append(0)
                results.append({"Num Features": num_features, "num_clusters": num_clusters,"selected_features": list(selected_features[i]), "stdevs": stdevs})
    res = pd.DataFrame(results)
    return res

def rolling_percentile_freturns(df, lookforward_freturn, lookback_percentile):
    df["Percentile"] = np.nan
    for i in range(len(df)):
        try:
            df.loc[df.index[i], "Percentile"] = stats.percentileofscore(df.iloc[i - lookforward_freturn - lookback_percentile + 1:i - lookforward_freturn]["FReturn"], df.iloc[i]["FReturn"])
        except:
            continue
    return df

def quantize(number, digits=2):
    dstr = "." + (digits - 1) * "0" + "1"
    q = Decimal(number).quantize(Decimal(dstr))
    q = float(q)
    return q

def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = np.fft.fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def plot_fft_plus_power(time, signal, figname=None):
    dt = (time[1] - time[0]).days
    N = len(signal)
    fs = 1/dt
    fig, ax = plt.subplots(2, 1, figsize=(15, 3), sharex=True)
    variance = np.std(signal)**2
    f_values, fft_values = get_fft_values(signal, dt, N, fs)
    fft_power = variance * abs(fft_values) ** 2  # FFT power spectrum
    ax[0].plot(f_values, fft_values, 'r-', label='Fourier Transform')
    ax[1].plot(f_values, fft_power, 'k--',
               linewidth=1, label='FFT Power Spectrum')
    ax[1].set_xlabel('Frequency [Hz / year]', fontsize=18)
    ax[1].set_ylabel('Amplitude', fontsize=12)
    ax[0].set_ylabel('Amplitude', fontsize=12)
    ax[0].legend()
    ax[1].legend()
    plt.plot()

def plot_wavelet(time, signal):
    waveletname = 'cmor1.5-1.0'
    scales = np.arange(1,8)
    dt = (time[1] - time[0]).days
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies

    scale0 = 8
    numlevels = 10

    levels = [scale0]
    for ll in range(1, numlevels):
        scale0 *= 2
        levels.append(scale0)

    cmap=plt.cm.Spectral
    title='Wavelet Transform (Power Spectrum) of signal'
    ylabel='Period (years)'
    xlabel='Time'

    contourlevels = np.log2(levels)
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power),
                     contourlevels, extend='both', cmap=cmap)
    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    yticks = 2**np.arange(np.ceil(np.log2(period.min())),
                          np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.plot()

class backtester:
    metrics = {'Sharpe': 'Sharpe Ratio',
               'CAGR': 'Compound Average Growth Rate',
               'MDD': 'Maximum Drawdown',
               'NHR': 'Normalized Hit Ratio',
               'OTS': 'Optimal trade size'}

    def __init__(self, data, signals,  start=None, end=None):

        """

        """
        self.signals = signals
        self.data = data  # the dataframe
        self.data['yr'] = self.data['Date'].dt.year
        self.n_days = (self.data.Date.iloc[-1] - self.data.Date.iloc[0])
        self.start = start  # the beginning date of the sample period
        self.end = end  # the ending date of the sample period

    def generate_signals(self, charts=True):
        """

        """
        bval = +1
        sval = 0
        self.data["signal"] = self.signals

        # Closing positions at end of time period
        self.data["signal"][-1:] = 0

        buy_plot_mask = ((self.data.signal.shift(-1) == bval) & (self.data.signal == sval))
        sell_plot_mask = ((self.data.signal.shift(-1) == sval) & (self.data.signal == bval))

        #initialize with a buy position
        buy_plot_mask[0] = True

        mask = (self.data.signal != self.data.signal.shift(1)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

        #self.data.to_csv("Test.csv")

        ## display chart
        if charts:
            plt.plot(self.data['Date'], self.data['Close'], color='black', label='Price')
            plt.plot(self.data.loc[buy_plot_mask, 'Date'], self.data.loc[buy_plot_mask, 'Close'], r'^', ms=15,
                     label="Entry Signal", color='green', markeredgecolor='k', markeredgewidth=1)
            plt.plot(self.data.loc[sell_plot_mask, 'Date'], self.data.loc[sell_plot_mask, 'Close'], r'^', ms=15,
                     label="Exit Signal", color='red', markeredgecolor='k', markeredgewidth=1)
            plt.title('Strategy Backtest')
            plt.legend(loc=0)
            d_color = {}
            d_color[1] = '#90ee90'  ## light green
            d_color[-1] = "#ffcccb"  ## light red
            d_color[0] = '#ffffff'

            j = 0
            for i in range(1, self.data.shape[0]):
                if np.isnan(self.data.signal[i - 1]):
                    j = i
                elif (self.data.signal[i - 1] == self.data.signal[i]) and (i < (self.data.shape[0] - 1)):
                    continue
                else:
                    plt.axvspan(self.data['Date'][j], self.data['Date'][i],
                                alpha=0.5, color=d_color[self.data.signal[i - 1]], label="interval")
                    j = i
            plt.show()


        return self.data[["Date", "signal"]]

    def signal_performance(self, allocation, interest_rate):
        """
        Another instance method
        """
        self.allocation = allocation
        self.int = interest_rate

        # creating returns and portfolio value series
        self.data['Return'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['S_Return'] = self.data['signal'].shift(1) * self.data['Return']
        self.data['S_Return'] = self.data['S_Return'].fillna(0)
        self.data['Return'] = self.data['Return'].fillna(0)
        self.data["S_Return"] = self.data["S_Return"] + (self.int / 25200) * (1 - self.data['signal'].shift(1))
        self.data['Market_Return'] = self.data['Return'].expanding().sum()
        self.data['Strategy_Return'] = self.data['S_Return'].expanding().sum()
        self.data['Portfolio Value'] = ((self.data['Strategy_Return'] + 1) * self.allocation)
        self.data['Wins'] = np.where(self.data['S_Return'] > 0, 1, 0)
        self.data['Losses'] = np.where(self.data['S_Return'] < 0, 1, 0)


    @staticmethod
    def kelly(p, b):
        return np.round(p - (1 - p) / b, 4)

    def plot_performance(self, allocation=1, interest_rate = 6):
        # intializing a variable for initial allocation
        # to be used to create equity curve
        self.signal_performance(allocation, interest_rate)

        # yearly performance
        self.yearly_performance()

        # Plotting the Performance of the strategy
        plt.plot(self.data['Date'], self.data['Market_Return'], color='black', label='Market Returns')
        plt.plot(self.data['Date'], self.data['Strategy_Return'], color='blue', label='Strategy Returns')
        # plt.plot(self.data['Date'],self.data['fisher_rate'],color='red', label= 'Fisher Rate')
        plt.title('Strategy Backtest')
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()

        plt.plot(self.data['Date'], self.data['Portfolio Value'], color='blue')
        plt.title('Portfolio Value')
        plt.show()
        # return(self.data)

    def yearly_performance(self):
        """
        Instance method
        Adds an instance attribute: yearly_df
        """
        _yearly_df = self.data.groupby(['yr', 'signal']).S_Return.sum().unstack()
        _yearly_df.rename(columns={-1.0: 'Sell', 1.0: 'Buy'}, inplace=True)
        _yearly_df['Return'] = _yearly_df.sum(1)

        # yearly_df
        self.yearly_df = _yearly_df.style.bar(color=["#ffcccb", '#90ee90'], align='zero').format({
            'Sell': '{:,.2%}'.format, 'Buy': '{:,.2%}'.format, 'Return': '{:,.2%}'.format})

def optimize_rf_model(X_train, y_train, X_validate, y_validate):

    neptune.init('suprabhash/ConstanceBrownClassifier', api_token=API_KEY)
    neptune.create_experiment(
        "RandomForestForward1", upload_source_files=["*.py"]
    )
    neptune_callback = optuna_utils.NeptuneCallback(log_study=True, log_charts=True)

    def objective(trial):
        params = {
            "criterion": "entropy",
            "n_estimators": int(
                trial.suggest_discrete_uniform("n_estimators", 50, 1500, 50)
            ),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_split": trial.suggest_uniform("min_samples_split", 0.1, 1.0),
            "max_features": trial.suggest_uniform("max_features", 0.1, 1.0),
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
            "oob_score": True,
        }

        rf_clf = RandomForestClassifier(**params)
        print("fitting...")
        rf_clf.fit(X_train, y_train)
        y_pred = rf_clf.predict(X_validate)
        score = metrics.matthews_corrcoef(y_validate, y_pred)
        print(f"internal score: {score:.4f}")
        return score


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25, callbacks=[neptune_callback])
    optuna_utils.log_study(study)


