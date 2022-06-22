import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import math
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import pickle
import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
import neptune.new as neptune
from datetime import datetime
import scipy.stats as stats
import neptune.new.integrations.optuna as optuna_utils
import optuna
import importlib.util
import os
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
API_KEY="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZDhlMWM3Zi1iMDExLTRmZDQtOTY4OS03MmY3ZjNiMzE5NzIifQ=="
import yfinance as yf

def get_data(ticker, frequency):
    """
    :param ticker: Ticker as on Reuters. Investpy and yfinance tickers can be passed using the lookup dict in tickers.py
    :param frequency: Frequency of the data required. Currently supports daily and hourly. Pass "D" or "H"
    :return:  Returns the OHLCV dataframe indexed by datetime
    """
    if frequency=='H':
        interval = "1H"
    if frequency == 'D':
        interval = "1D"

    ticker_dataframe = yf.download(ticker, interval=interval, start="2009-01-01", end=str(datetime.now())[:10]).reset_index()
    if 'Date' in ticker_dataframe.columns:
        ticker_dataframe.rename(columns={'Date': 'Datetime'}, inplace=True)
    ticker_dataframe["Datetime"] = pd.to_datetime(ticker_dataframe["Datetime"])
    return ticker_dataframe

def plot_performance(df, prices, features, actions_history, equity_curve, save=False):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 7))
    # fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.plot(prices, label='Close')
    ax1_copy = ax1.twinx()
    ax1_copy.plot(actions_history, label='Actions')
    ax2.plot(actions_history, label='Actions')
    ax2_copy = ax2.twinx()
    for feature in [feature["feature"] for feature in features]:
        ax2_copy.plot(df[feature].values[:len(actions_history)], label=feature, color='green', ls='dotted')
    ax2_copy.axhline(0.0, ls='--', color='grey')
    ax3.plot(equity_curve, label='Net worth')
    ax3.plot([price*10000 / prices[0] for price in prices], label='Benchmark')
    ax1.legend()
    # ax2.legend()
    ax3.legend()
    if type(save)==str:
        plt.savefig(save)
    else:
        plt.show()
    return fig

def add_fisher(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'Fisher{lookback}' not in temp.columns:
        temp[f'Fisher{lookback}'] = fisher(temp, lookback)
    return temp

def fisher(ohlc, period):
    def __round(val):
        if (val > .99):
            return .999
        elif val < -.99:
            return -.999
        return val

    from numpy import log, seterr
    seterr(divide="ignore")
    med = (ohlc["High"] + ohlc["Low"]) / 2
    ndaylow = med.rolling(window=period).min()
    ndayhigh = med.rolling(window=period).max()
    med = [0 if math.isnan(x) else x for x in med]
    ndaylow = [0 if math.isnan(x) else x for x in ndaylow]
    ndayhigh = [0 if math.isnan(x) else x for x in ndayhigh]
    raw = [0] * len(med)
    for i in range(0, len(med)):
        try:
            raw[i] = 2 * ((med[i] - ndaylow[i]) / (ndayhigh[i] - ndaylow[i]) - 0.5)
        except:
            ZeroDivisionError
    value = [0] * len(med)
    value[0] = __round(raw[0] * 0.33)
    for i in range(1, len(med)):
        try:
            value[i] = __round(0.33 * raw[i] + 0.67 * value[i - 1])
        except:
            ZeroDivisionError
    _smooth = [0 if math.isnan(x) else x for x in value]
    fish1 = [0] * len(_smooth)
    for i in range(1, len(_smooth)):
        fish1[i] = ((0.5 * (np.log((1 + _smooth[i]) / (1 - _smooth[i]))))) + (0.5 * fish1[i - 1])

    return fish1


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

def add_constance_brown(input):
    temp = input[0].copy()
    if f'ConstanceBrown' not in temp.columns:
        r = RSI(temp, 14)
        rsi_mom_length = 9
        ma_length = 3
        rsi_ma_length = 3
        rsidelta = [0] * len(temp)
        for i in range(len(temp)):
            if i < rsi_mom_length:
                rsidelta[i] = np.nan
            else:
                rsidelta[i] = r[i] - r[i - rsi_mom_length]
        rsisigma = RSI(temp, rsi_ma_length).rolling(window=ma_length).mean()
        rsidelta = [0 if math.isnan(x) else x for x in rsidelta]
        s = [0] * len(temp)
        for i in range(len(rsidelta)):
            s[i] = rsidelta[i] + rsisigma[i]
        # s = [0 if math.isnan(x) else x for x in s]

        temp["ConstanceBrown"] = s
    return temp
def time_consolidator(df, period):
    aggregate = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Up": "sum",
        "Down": "sum",
        "Volume": "sum",
    }
    return df.resample(f"{period}Min").agg(aggregate).dropna()

def get_stock_data(symbol):
    if symbol == 'sinx':
        df = get_data("^NSEI", 'D')
        df.drop(columns=["Volume"], inplace=True)
        df["Close"] = df["Open"] = df["High"] = df["Low"] = np.sin(df.index / 10 ) +2
    elif symbol == 'SPY':
        df = pd.read_csv('SPY.txt')
        df["Datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"], infer_datetime_format=True
        )
        df["Volume"] = df["Up"] + df["Down"]
        df = time_consolidator(df.set_index("Datetime"), 60).reset_index()
    else:
        df = get_data(symbol, 'D')
    df.set_index("Datetime", inplace=True)
    df.dropna(inplace=True)
    return df

def f_discretize(values, num_states=10):
    states_value = dict()
    step_size = 1./num_states
    for i in range(num_states):
        if i == num_states - 1:
            states_value[i] = values.max()
        else:
            states_value[i] = values.quantile((i+1)*step_size)
    return states_value

def value_to_state(value, states_value):
    if np.isnan(value):
        return np.nan
    else:
        for state, v in states_value.items():
            if value <= v:
                return state
        return state

def add_features(df, features, state_lookback, train_percent):
    lookbacks = []
    for feature, lookback in [(feature["feature"], feature["lookback"]) for feature in features]:
        lookbacks.append(lookback)

        if feature.startswith("Fisher"):
            df[feature] = add_fisher([df, lookback])[[f"Fisher{lookback}"]]

        if feature.startswith("Close_as_a_feature"):
            df[feature] = df["Close"]

        if feature.startswith("diff_of_close"):
            df[feature] = df["Close"].diff()

        if feature.startswith("Momentum"):
            def aqr_momentum(array):
                returns = np.diff(np.log(array))
                x = np.arange(len(returns))
                slope, _, rvalue, _, _ = stats.linregress(x, returns)
                return ((1 + slope) ** 252) * (rvalue ** 2)
            df[feature] = df["Close"].rolling(lookback).apply(aqr_momentum)

        if feature.startswith("IBS"):
            df[feature] = (df.Close - df.Low) / (df.High - df.Low)

        if feature.startswith("CB"):
            df[feature] = add_constance_brown([df])[["ConstanceBrown"]]

        if feature.startswith("Volume"):
            with open(f'NIFc1_non_absolute_percentage_of_poc_vah_val_polv_and_ohlc_avg_across_various_lookbacks_for_all_days.pkl','rb') as file:
                vol_feats = pickle.load(file)/100
            if feature.startswith("VolumePOC"):
                df = pd.concat([df, vol_feats[f"{lookback}_day_lookback_poc"]], axis=1)
                df = df.rename(columns={f"{lookback}_day_lookback_poc": feature})
            if feature.startswith("VolumeVAL"):
                df = pd.concat([df, vol_feats[f"{lookback}_day_lookback_val"]], axis=1)
                df = df.rename(columns={f"{lookback}_day_lookback_val": feature})
            if feature.startswith("VolumeVAH"):
                df = pd.concat([df, vol_feats[f"{lookback}_day_lookback_vah"]], axis=1)
                df = df.rename(columns={f"{lookback}_day_lookback_vah": feature})
            if feature.startswith("VolumePOLV"):
                df = pd.concat([df, vol_feats[f"{lookback}_day_lookback_polv"]], axis=1)
                df = df.rename(columns={f"{lookback}_day_lookback_polv": feature})

        if feature.startswith("AvgDeviation"):
            with open(f'percentage_deviation_avg.pkl','rb') as file:
                avg_deviation = pickle.load(file)
            df = pd.concat([df,avg_deviation["AvgDeviation"]],axis=1)

        for i in range(1, state_lookback):
            df[f"{feature}_shift{i}"] = df[feature].shift(1)

    df = df.iloc[max(lookbacks)+state_lookback:]
    df.dropna(inplace=True)
    for feature in [feature['feature'] for feature in features] + [f"{col}_shift{i}" for col in [feature['feature'] for feature in features] for i in range(1, state_lookback)]:
        states = f_discretize(df[feature].iloc[:int(df.shape[0] * train_percent)], 100)
        df[feature] = df[feature].apply(lambda x: value_to_state(x, states))/100
    return df