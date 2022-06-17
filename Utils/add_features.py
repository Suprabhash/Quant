#File contains utility functions

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import datetime
from hurst import *

###
# Sample function
###
from Utils.utils import rolling_percentile_freturns


def feature_creator(df):
    #Add Features here
    return df

###
#   PART-1 : FISHER TRANSFORM
###

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
    fish2 = fish1[1:len(fish1)]
    # plt.figure(figsize=(18, 8))
    # plt.plot(ohlc.index, fish1, linewidth=1, label="Fisher_val")
    # plt.legend(loc="upper left")
    # plt.show()
    return fish1


def add_RSI(temp, col_name, lookback):
    delta = temp[col_name].diff().dropna()
    ups = delta*0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[lookback-1]] = np.mean(ups[:lookback]) # The first element should be a simple mean average
    ups = ups.drop(ups.index[:(lookback-1)])
    downs[downs.index[lookback-1]] = np.mean(downs[:lookback]) # The first element should be a simple mean average
    downs = downs.drop(downs.index[:(lookback-1)])
    rs = ups.ewm(com=lookback-1, min_periods=0, adjust=False, ignore_na=False).mean()/ \
         downs.ewm(com=lookback-1, min_periods=0, adjust=False, ignore_na=False).mean()
    rsi = round(100 - 100/(1+rs), 2)
    temp[f'RSI{lookback}'] = np.nan
    temp.loc[lookback:,f'RSI{lookback}'] = rsi
    temp[f'RSI{lookback}'].fillna(0, inplace=True)
    return temp[f'RSI{lookback}']

###
#  ZSCORE
###

def add_zscore(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'z_score{lookback}_ohlc4' not in temp.columns:
        roll = temp["ohlc4"].rolling(lookback)
        temp[f'z_score{lookback}_ohlc4'] = (temp["ohlc4"] - roll.mean()) / roll.std()
    return temp

###
#  Inverse Fisher
###
def add_inverse_fisher(input):
    temp = input[0].copy()
    lookback = input[1]

    temp["ohlc4"] = (temp["Open"] + temp["Close"] + temp["High"] + temp["Low"]) / 4
    if f'z_score{lookback}_ohlc4' not in temp.columns:
        roll = temp["ohlc4"].rolling(lookback)
        temp[f'z_score{lookback}_ohlc4'] = (temp["ohlc4"] - roll.mean()) / roll.std()

    if f'tanh_z_score{lookback}_ohlc4' not in temp.columns:
        temp[[f"tanh_z_score{lookback}_ohlc4"]] = np.tanh(temp[[f"z_score{lookback}_ohlc4"]])

    temp[f"InverseFisher{lookback}"] = temp[f"tanh_z_score{lookback}_ohlc4"]
    temp = temp.drop(columns=["ohlc4", f'z_score{lookback}_ohlc4', f'tanh_z_score{lookback}_ohlc4'])
    return temp

###
# Add Constance Brown
###
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

###
#  TANH
###

def add_tanh_zscores(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'tanh_z_score{lookback}_ohlc4' not in temp.columns:
        temp[[f"tanh_z_score{lookback}_ohlc4"]] = np.tanh(temp[[f"z_score{lookback}_ohlc4"]])
    return temp

###
# Add Mday Accumulation of Nday Highs/Lows
###

def add_accumulation_MB(input):
    temp = input[0].copy()
    M = input[1]
    N = input[2]
    tickers = input[3]
    if (f"NumHighs{N}" not in temp.columns):
        for ticker in tickers:
            temp[f"{ticker}_is{N}DayHigh"] = np.where(temp[ticker] == temp[ticker].rolling(window=N).max(), 1, 0)

        temp[f"NumHighs{N}"] = 0

        for i in range(len(temp)):
            for ticker in tickers:
                temp.loc[i, f"NumHighs{N}"] = temp[f"NumHighs{N}"].iloc[i] + temp[f"{ticker}_is{N}DayHigh"].iloc[i] * \
                                              temp[f"{ticker}_isConstituent"].iloc[i]

        temp[f"{M}DayHighAccumulation_{N}dayHighs"] = temp[f"NumHighs{N}"].rolling(window=M).sum()

    if (f"NumLows{N}" not in temp.columns):
        for ticker in tickers:
            temp[f"{ticker}_is{N}DayLow"] = np.where(temp[ticker] == temp[ticker].rolling(window=N).min(), 1, 0)

        temp[f"NumLows{N}"] = 0
        for i in range(len(temp)):
            for ticker in tickers:
                temp.loc[i, f"NumLows{N}"] = temp[f"NumLows{N}"].iloc[i] + temp[f"{ticker}_is{N}DayLow"].iloc[i] * \
                                             temp[f"{ticker}_isConstituent"].iloc[i]

        temp[f"{M}DayLowAccumulation_{N}dayLows"] = temp[f"NumLows{N}"].rolling(window=M).sum()

    temp[f"{M}DayAccumulation_{N}dayMB"] = temp[f"{M}DayHighAccumulation_{N}dayHighs"] - temp[f"{M}DayLowAccumulation_{N}dayLows"]

    return temp

###
#  Function transforms
###

def add_F(input):
    temp = input[0].copy()
    zlookback = input[1]
    f = input[2]
    f_lookback = input[3]

    if f.__name__ == "x":
        pass
    else:
        if f"{f.__name__}_{f_lookback}_tanh_z_score{zlookback}_ohlc4" not in temp.columns:
            temp = f(temp, f"tanh_z_score{zlookback}_ohlc4", f_lookback)

    return temp

def add_F_MarketBreadthMB(input):
    temp = input[0].copy()
    M = input[1]
    N = input[2]
    f = input[3]
    f_lookback = input[4]

    if f.__name__ == "x":
        pass
    else:
        if f"{f.__name__}{f_lookback}_{M}DayAccumulation_{N}dayMB" not in temp.columns:
            temp = f(temp, f"{M}DayAccumulation_{N}dayMB", f_lookback)

    return temp

def Fisher(df, col_name,lookback):
    def fisher(ohlc, col_name, period):
        def __round(val):
            if (val > .99):
                return .999
            elif val < -.99:
                return -.999
            return val

        from numpy import log, seterr
        seterr(divide="ignore")
        med = (ohlc[col_name])
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
        fish2 = fish1[1:len(fish1)]
        # plt.figure(figsize=(18, 8))
        # plt.plot(ohlc.index, fish1, linewidth=1, label="Fisher_val")
        # plt.legend(loc="upper left")
        # plt.show()
        return fish1
    df[f'Fisher{lookback}_{col_name}'] = fisher(df, col_name, lookback)
    return df

def Stochastic(df, col_name,lookback):
    df[f'Stochastic{lookback}_{col_name}'] = add_RSI(df, col_name, lookback)/10
    return df

def x(df, col_name,lookback):
    return df

def max_over_lookback(df, col_name,lookback):
    df[f"max_over_lookback{lookback}_{col_name}"] = df[col_name].rolling(window=lookback).max()
    return df

def min_over_lookback(df, col_name,lookback):
    df[f"min_over_lookback{lookback}_{col_name}"] = df[col_name].rolling(window=lookback).min()
    return df

def sma(df, col_name,lookback):
    df[f"sma{lookback}_{col_name}"] = df[col_name].rolling(window=lookback).mean()
    return df

def shift(df, col_name,lookback):
    df[f"shift{lookback}_{col_name}"] = df[col_name].shift(lookback)
    return df

###
#  Add Volume Features
###

def return_volume_features_minute_hourly(df_hour, df_min):
    temp = []
    for i in tqdm(range(len(df_hour))):
        res = {}
        res["Datetime"] = df_hour.iloc[i]["Datetime"]
        for n in [2, 5, 10, 21, 42, 63, 126, 252, 504]:
            try:
                if i >= n - 1:
                    volumes, high_prices, low_prices = return_dataframe_minute(df_min, df_hour, i, n)
                    res[f"CalcHow_{n}"] = "Minute"
                    price_levels = calc_distribution(high_prices, low_prices, volumes)
                    res[f"PriceLevels_{n}"] = price_levels
                else:
                    res[f"PriceLevels_{n}"] = {}
                    res[f"CalcHow_{n}"] = "DataNotAvailable"
            except:
                res[f"PriceLevels_{n}"] = {}
                res[f"CalcHow_{n}"] = "DataNotAvailable"
        temp.append(res)
    return temp


def return_dataframe_minute(temp_og1, temp_og, i, n):
    volumes = list(temp_og1[(temp_og1["Datetime"] > pd.to_datetime(
            datetime.datetime(temp_og["Datetime"].iloc[i - n].year, temp_og["Datetime"].iloc[i - n].month,
                              temp_og["Datetime"].iloc[i - n].day, temp_og["Datetime"].iloc[i - n].hour, temp_og["Datetime"].iloc[i - n].minute))) &
                                (temp_og1["Datetime"] <= pd.to_datetime(
                                    datetime.datetime(temp_og["Datetime"].iloc[i].year, temp_og["Datetime"].iloc[i].month,
                                                      temp_og["Datetime"].iloc[i].day, temp_og["Datetime"].iloc[i].hour, temp_og["Datetime"].iloc[i].minute)))]["Volume"])
    high_prices = list(temp_og1[(temp_og1["Datetime"] > pd.to_datetime(
            datetime.datetime(temp_og["Datetime"].iloc[i - n].year, temp_og["Datetime"].iloc[i - n].month,
                              temp_og["Datetime"].iloc[i - n].day, temp_og["Datetime"].iloc[i - n].hour, temp_og["Datetime"].iloc[i - n].minute))) &
                                (temp_og1["Datetime"] <= pd.to_datetime(
                                    datetime.datetime(temp_og["Datetime"].iloc[i].year, temp_og["Datetime"].iloc[i].month,
                                                      temp_og["Datetime"].iloc[i].day, temp_og["Datetime"].iloc[i].hour, temp_og["Datetime"].iloc[i].minute)))]["High"])
    low_prices = list(temp_og1[(temp_og1["Datetime"] > pd.to_datetime(
            datetime.datetime(temp_og["Datetime"].iloc[i - n].year, temp_og["Datetime"].iloc[i - n].month,
                              temp_og["Datetime"].iloc[i - n].day, temp_og["Datetime"].iloc[i - n].hour, temp_og["Datetime"].iloc[i - n].minute))) &
                                (temp_og1["Datetime"] <= pd.to_datetime(
                                    datetime.datetime(temp_og["Datetime"].iloc[i].year, temp_og["Datetime"].iloc[i].month,
                                                      temp_og["Datetime"].iloc[i].day, temp_og["Datetime"].iloc[i].hour, temp_og["Datetime"].iloc[i].minute)))]["Low"])

    return volumes, high_prices, low_prices


def calc_distribution(highs,lows,volumes, plot_hist=False):
    x = []
    y = []
    for i in range(len(volumes)):
        prices = np.round(np.linspace(lows[i], highs[i], num=10),2)
        for j in range(10):
            x.append(prices[j])
            y.append(volumes[i]/10)
    prices = np.linspace(min(x), max(x), num=25)
    p = [0]*(len(prices)-1)
    v = [0]*(len(prices)-1)
    for j in range(len(prices)-1):
        p[j] = (prices[j] + prices[j+1])/2
    for i in range(len(x)):
        for j in range(len(prices)-1):
            if (x[i]>prices[j]-0.001) & (x[i]<=prices[j+1]):
                v[j] = v[j] + (y[i])

    poc = p[v.index(max(v))]
    profile_high = max(highs)
    profile_low = min(lows)
    target_volume = 0.7*sum(v)
    vol = max(v)
    bars_in_value_area = [v.index(max(v))]
    while vol < target_volume:
        # print("*"*100)
        # print(f"Target vol: {target_volume}")
        # print(f"Vol before: {vol}")
        # print(f"bars_in_value_area before: {bars_in_value_area}")
        if max(bars_in_value_area) > 21:
            vol_above = 0
        else:
            vol_above = v[max(bars_in_value_area) + 1] + v[max(bars_in_value_area) + 2]
        if min(bars_in_value_area) < 2:
            vol_below = 0
        else:
            vol_below = v[min(bars_in_value_area) - 1] + v[min(bars_in_value_area) - 2]
        if vol_above > vol_below:
            if max(bars_in_value_area) < 22:
                vol = vol + vol_above
                bars_in_value_area.extend([max(bars_in_value_area) + 1, max(bars_in_value_area) + 2])
            else:
                vol = vol + vol_below
                bars_in_value_area.extend([min(bars_in_value_area) - 1, min(bars_in_value_area) - 2])
        else:
            if min(bars_in_value_area) > 1:
                vol = vol + vol_below
                bars_in_value_area.extend([min(bars_in_value_area) - 1, min(bars_in_value_area) - 2])
            else:
                vol = vol + vol_above
                bars_in_value_area.extend([max(bars_in_value_area) + 1, max(bars_in_value_area) + 2])
        bars_in_value_area.sort()
        # print(f"bars_in_value_area after: {bars_in_value_area}")
        # print(f"Vol after: {vol}")
        if bars_in_value_area[-1] > 30:
            raise NameError("Hi there")
    vah = p[max(bars_in_value_area)]
    val = p[min(bars_in_value_area)]
    if plot_hist:
        plt.bar(p,v)
        plt.plot(p, v)
        plt.axvline(x=poc, color="orange")
        plt.axvline(x=profile_high, color="darkgreen")
        plt.axvline(x=profile_low, color="maroon")
        plt.axvline(x=vah, color="lime")
        plt.axvline(x=val, color="red")
        plt.legend(["Point of Control", "Profile High", "Profile Low", "Value Area High", "Value Area Low"])
    return {"poc": poc,"profile_high": profile_high,"profile_low": profile_low,"vah": vah,"val": val}


###
#  Add volume Features
###

def prepare_volume_features(ohlcv, vol_feat):
    date_col = "Datetime"
    vol_temp = pd.DataFrame()
    vol_feat = pd.concat([vol_feat.set_index(date_col), ohlcv.set_index(date_col)], axis=1, join="inner").reset_index()
    vol_temp[date_col] = vol_feat[date_col]

    lookbacks = [2, 5]  #, 10, 21, 42, 63, 126, 252, 504]

    #adding vol_levels, pct deviation wrt price level, delta of price wrt level, velocity of price wrt level, acceleration of price wrt level
    print("adding vol_levels, pct deviation wrt price level, delta of price wrt level, velocity of price wrt level, acceleration of price wrt level")
    for n in tqdm(lookbacks):  #, 10, 21, 42, 63, 126, 252, 504
        for i in range(len(vol_temp)):
            try:
                vol_temp.loc[i, f"poc_{n}"] = vol_feat.iloc[i][f"PriceLevels_{n}"]['poc']
                vol_temp.loc[i, f"vah_{n}"] = vol_feat.iloc[i][f"PriceLevels_{n}"]['vah']
                vol_temp.loc[i, f"val_{n}"] = vol_feat.iloc[i][f"PriceLevels_{n}"]['val']
                vol_temp.loc[i, f"dev_from_poc_{n}"] = vol_feat.iloc[i]["Close"] / vol_feat.iloc[i][f"PriceLevels_{n}"]['poc'] - 1
                vol_temp.loc[i, f"dev_from_vah_{n}"] = vol_feat.iloc[i]["Close"] / vol_feat.iloc[i][f"PriceLevels_{n}"]['vah'] - 1
                vol_temp.loc[i, f"dev_from_val_{n}"] = vol_feat.iloc[i]["Close"] / vol_feat.iloc[i][f"PriceLevels_{n}"]['val'] - 1
                vol_temp.loc[i, f"delta_from_poc_{n}"] = vol_feat.iloc[i]["Close"] - vol_feat.iloc[i][f"PriceLevels_{n}"]['poc']
                vol_temp.loc[i, f"delta_from_vah_{n}"] = vol_feat.iloc[i]["Close"] - vol_feat.iloc[i][f"PriceLevels_{n}"]['vah']
                vol_temp.loc[i, f"delta_from_val_{n}"] = vol_feat.iloc[i]["Close"] - vol_feat.iloc[i][f"PriceLevels_{n}"]['val']
            except:
                try:
                    vol_temp.loc[i, f"poc_{n}"] = vol_feat.iloc[i-1][f"PriceLevels_{n}"]['poc']
                    vol_temp.loc[i, f"vah_{n}"] = vol_feat.iloc[i-1][f"PriceLevels_{n}"]['vah']
                    vol_temp.loc[i, f"val_{n}"] = vol_feat.iloc[i-1][f"PriceLevels_{n}"]['val']
                    vol_temp.loc[i, f"dev_from_poc_{n}"] = vol_feat.iloc[i]["Close"] / vol_feat.iloc[i-1][f"PriceLevels_{n}"]['poc'] - 1
                    vol_temp.loc[i, f"dev_from_vah_{n}"] = vol_feat.iloc[i]["Close"] / vol_feat.iloc[i-1][f"PriceLevels_{n}"]['vah'] - 1
                    vol_temp.loc[i, f"dev_from_val_{n}"] = vol_feat.iloc[i]["Close"] / vol_feat.iloc[i-1][f"PriceLevels_{n}"]['val'] - 1
                    vol_temp.loc[i, f"delta_from_poc_{n}"] = vol_feat.iloc[i]["Close"] - vol_feat.iloc[i-1][f"PriceLevels_{n}"]['poc']
                    vol_temp.loc[i, f"delta_from_vah_{n}"] = vol_feat.iloc[i]["Close"] - vol_feat.iloc[i-1][f"PriceLevels_{n}"]['vah']
                    vol_temp.loc[i, f"delta_from_val_{n}"] = vol_feat.iloc[i]["Close"] - vol_feat.iloc[i-1][f"PriceLevels_{n}"]['val']
                except:
                    continue
        vol_temp[f"velocity_poc_{n}"] = vol_temp[f"poc_{n}"].diff()
        vol_temp[f"velocity_val_{n}"] = vol_temp[f"val_{n}"].diff()
        vol_temp[f"velocity_vah_{n}"] = vol_temp[f"vah_{n}"].diff()
        vol_temp[f"acceleration_poc_{n}"] = vol_temp[f"velocity_poc_{n}"].diff()
        vol_temp[f"acceleration_val_{n}"] = vol_temp[f"velocity_val_{n}"].diff()
        vol_temp[f"acceleration_vah_{n}"] = vol_temp[f"velocity_vah_{n}"].diff()
        vol_temp[f"relvelocity_of_Price_from_poc_{n}"] = vol_temp[f"delta_from_poc_{n}"].diff()
        vol_temp[f"relvelocity_of_Price_from_val_{n}"] = vol_temp[f"delta_from_val_{n}"].diff()
        vol_temp[f"relvelocity_of_Price_from_vah_{n}"] = vol_temp[f"delta_from_vah_{n}"].diff()
        vol_temp[f"relacceleration_of_Price_from_poc_{n}"] = vol_temp[f"relvelocity_of_Price_from_poc_{n}"].diff()
        vol_temp[f"relacceleration_of_Price_from_vah_{n}"] = vol_temp[f"relvelocity_of_Price_from_vah_{n}"].diff()
        vol_temp[f"relacceleration_of_Price_from_val_{n}"] = vol_temp[f"relvelocity_of_Price_from_val_{n}"].diff()
    #add relative deltas, velocities and accelerations between levels
    print("add relative deltas, velocities and accelerations between levels")
    for n1 in tqdm(lookbacks):
        for l1 in ["vah", "val", "poc"]:
            for n2 in lookbacks:
                for l2 in ["vah", "val", "poc"]:
                    if (n1==n2)&(l1==l2):
                        continue
                    if (f"delta_{l2}{n2}_{l1}{n1}") in list(vol_temp.columns):
                        continue
                    vol_temp[f"delta_{l1}{n1}_{l2}{n2}"] = (vol_temp[f"{l1}_{n1}"] - vol_temp[f"{l2}_{n2}"])
                    vol_temp[f"velocity_{l1}{n1}_{l2}{n2}"] = vol_temp[f"delta_{l1}{n1}_{l2}{n2}"].diff()
                    vol_temp[f"acceleration_{l1}{n1}_{l2}{n2}"] = vol_temp[f"velocity_{l1}{n1}_{l2}{n2}"].diff()


    vol_temp = pd.concat([vol_temp.set_index(date_col), ohlcv.set_index(date_col)], axis=1, join="outer").reset_index()
    return vol_temp

def add_percentile_of_forward_returns(vol_temp, return_lookforwards, percentile_lookbacks, freturn):
    #Adding metrics
    for return_lookforward in tqdm(return_lookforwards):
        for percentile_lookback in percentile_lookbacks:
            if freturn == "max":
                for i in range(1, return_lookforward+1):
                    vol_temp[f"FReturn{i}"] = vol_temp["Close"].shift(-i) / vol_temp["Close"] - 1
                vol_temp[f"FReturn"] = vol_temp[[f"FReturn{i}" for i in range(1, return_lookforward+1)]].max(axis=1)
                vol_temp = vol_temp.drop(columns=[f"FReturn{i}" for i in range(1, return_lookforward+1)])
            if freturn == "min":
                for i in range(1, return_lookforward+1):
                    vol_temp[f"FReturn{i}"] = vol_temp["Close"].shift(-i) / vol_temp["Close"] - 1
                vol_temp[f"FReturn"] = vol_temp[[f"FReturn{i}" for i in range(1, return_lookforward+1)]].min(axis=1)
                vol_temp = vol_temp.drop(columns=[f"FReturn{i}" for i in range(1, return_lookforward+1)])
            if freturn == "simple":
                vol_temp[f"FReturn"] = vol_temp["Close"].shift(-return_lookforward) / vol_temp["Close"] - 1
            vol_temp[f"{return_lookforward}FReturn_percentile_over_{percentile_lookback}"] = rolling_percentile_freturns(vol_temp, return_lookforward, percentile_lookback)["Percentile"]/100
            vol_temp.drop(columns="FReturn", inplace=True)
            vol_temp.drop(columns="Percentile", inplace=True)

    vol_temp.dropna(inplace=True)
    vol_temp.reset_index(drop=True, inplace=True)
    return vol_temp


###
# Adding Hurst Value
###

def add_hurst(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'H{lookback}' not in temp.columns:
        temp[f'H{lookback}'] = temp["Close"].rolling(lookback).apply(lambda x: compute_Hc(x.to_numpy(), kind='price', simplified=False)[0])
        temp[f'H{lookback}'] = temp["Close"].rolling(lookback).apply(
            lambda x: compute_Hc(x.to_numpy(), kind='price', simplified=False)[0])
    return temp

def add_hurst_C(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'H{lookback}' not in temp.columns:
        temp[f'H{lookback}'] = temp["Close"].rolling(lookback).apply(lambda x: compute_Hc(x.to_numpy(), kind='price', simplified=False)[0])
        temp[f'C{lookback}'] = temp["Close"].rolling(lookback).apply(lambda x: compute_Hc(x.to_numpy(), kind='price', simplified=False)[1])
    return temp

def add_MA_hurst(input):
    temp = input[0].copy()
    hurst_lookback = input[1]
    MA_lookback = input[2]
    if f"MA{MA_lookback}_H{hurst_lookback}" not in temp.columns:
        temp[f"MA{MA_lookback}_H{hurst_lookback}"] = temp[f"H{hurst_lookback}"].rolling(MA_lookback).mean()
    return temp

def add_ROC_MA_hurst(input):
    temp = input[0].copy()
    hurst_lookback = input[1]
    MA_lookback = input[2]
    ROC_lookback = input[3]

    if f"ROC{ROC_lookback}_MA{MA_lookback}_H{hurst_lookback}" not in temp.columns:
        temp[f"ROC{ROC_lookback}_MA{MA_lookback}_H{hurst_lookback}"] = temp[f"MA{MA_lookback}_H{hurst_lookback}"].pct_change(ROC_lookback)

    return temp