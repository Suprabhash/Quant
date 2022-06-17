"""
Contains utility functions used throughout the framework
"""
import multiprocessing

import pandas as pd
from datetime import date

from matplotlib import pyplot as plt
from tqdm.asyncio import tqdm

from Backtester.backtester import backtester
import numpy as np

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import pickle
import ssl
import datetime
from scipy import stats

###
# Emailer
###
from Optimisers.MCMC.MCMC import MCMC
calc_nmi = MCMC.nmi


def SendMail(emailIDs, strategy, ticker, sortby, ImgFileNameList):

    cc = ""
    for emailID in emailIDs:
        cc = cc + emailID + ","

    msg = MIMEMultipart()
    msg['Subject'] = f'{strategy}:{ticker} Top 3 strategies sorted by {sortby}'
    msg['From'] = 'suprabhash@quantiniti.com'
    msg['Cc'] = cc[:-1]   #
    msg['To'] = 'suprabhash@quantiniti.com'

    text = MIMEText(f'{ticker} Top 3 strategies sorted by {sortby}')
    msg.attach(text)
    for ImgFileName in ImgFileNameList:
        with open(ImgFileName, 'rb') as f:
            img_data = f.read()
        image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
        msg.attach(image)

    s = smtplib.SMTP('smtpout.secureserver.net', 465)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login('suprabhash@quantiniti.com', 'esahYah8')
    s.sendmail('suprabhash@quantiniti.com', emailIDs, msg.as_string())  #
    s.quit()

###
#  Utility functions for processing dates
###

def valid_dates(dates_all):           #Dates until parameter date: change name
    dates = []
    i = 0
    while True:
        dates.append(dates_all[i])
        if dates_all[i] > pd.to_datetime(date.today()):
            break
        i = i + 1
    return dates

def interpret_time_unit(str):
    frequency = {
        "TradingDays": "B",
        "Days": "D",
        "Weeks": "W",
        "Months": "M",
        "Hours": "H",
        "Minutes": "min"
    }

    num = int(str.split('_')[0])
    units = frequency[str.split('_')[1]]
    return num, units

###
# Strategy Filters
###

def callable_functions_helper(params):
    param_func_dict = {}
    for num in range(len(params)):
        if callable(params[num]):
            param_func_dict[params[num].__name__] = params[num]
            params[num] = params[num].__name__
    return params, param_func_dict

def correlation_filter(strategies_df, strategy, strategy_name,number_selected_strategies, start, end):
    returns = pd.DataFrame()
    for i in range(len(strategies_df)):
        with open(f'Caches/{strategy.ticker}/{strategy.frequency}/{strategy_name}/SelectedStrategies/Backtests/{tuple(callable_functions_helper(list(strategies_df.iloc[i]["params"]))[0])}.pkl','rb') as file:
            sreturn = pickle.load(file)
        sreturn=sreturn["equity_curve"]
        sreturn = sreturn.loc[(sreturn["Datetime"]>start) & (sreturn["Datetime"]<=end)].reset_index(drop=True)
        sreturn = sreturn.dropna()
        if i == 0:
            returns = sreturn['S_Return'].to_frame().rename(columns={'S_Return': f'Strategy{i+1}'}).set_index(sreturn["Datetime"])
        else:
            returns = pd.merge(returns, (sreturn['S_Return'].to_frame().rename(columns={'S_Return': f'Strategy{i+1}'}).set_index(sreturn["Datetime"])),left_index=True, right_index=True)


    corr_mat = returns.corr()
    strategies = [column for column in returns]
    selected_strategies = ["Strategy1"]
    strategies.remove("Strategy1")
    last_selected_strategy = "Strategy1"

    while len(selected_strategies) < number_selected_strategies:
        corrs = corr_mat.loc[strategies][last_selected_strategy]
        corrs = corrs.loc[corrs > 0.9]
        strategies = [st for st in strategies if st not in corrs.index.to_list()]
        if len(strategies) == 0:
            break
        strat = strategies[0]
        selected_strategies.append(strat)
        strategies.remove(strat)
        last_selected_strategy = strat

    selected_strategies = strategies_df.iloc[[int(strategy[8:])-1 for strategy in selected_strategies]].reset_index(drop=True)
    return selected_strategies

###
#  Code for creating float ranges with no floating point errors
###

import numpy as np
import decimal
def frange(start, end, jump):
    naive_list = np.arange(start, end, jump).tolist()
    decimals = []
    for el in naive_list:
        decimals.append(decimal.Decimal(str(el)).as_tuple().exponent)
    final_list = [round(el,-1*max(decimals)) for el in naive_list]
    return final_list

###
# Resampling data
###
def resample_data(df, minutes):
    return df.set_index("Datetime").groupby(pd.Grouper(freq=f'{minutes}Min')).agg({"Open": "first",
                                                 "Close": "last",
                                                 "Low": "min",
                                                 "High": "max",
                                                "Volume": "sum"}).reset_index()

###
#  Code for downloading ETH minute data from cryptodatadownload
###
def get_data_ETH_minute(path):
    filepaths = ["https://www.cryptodatadownload.com/cdd/Bitstamp_ETHUSD_2017_minute.csv",
                 "https://www.cryptodatadownload.com/cdd/Bitstamp_ETHUSD_2018_minute.csv",
                 "https://www.cryptodatadownload.com/cdd/Bitstamp_ETHUSD_2019_minute.csv",
                 "https://www.cryptodatadownload.com/cdd/Bitstamp_ETHUSD_2020_minute.csv",
                 "https://www.cryptodatadownload.com/cdd/Bitstamp_ETHUSD_2021_minute.csv"]
    data = pd.DataFrame()
    for filepath in filepaths:
        ssl._create_default_https_context = ssl._create_unverified_context
        temp_og = pd.read_csv(filepath, parse_dates=True, skiprows=1)
        temp_og.drop(columns=["date", "symbol", "Volume USD"], inplace=True)
        temp_og["unix"] = temp_og["unix"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000) if len(
            str(int(x))) == 13 else datetime.datetime.fromtimestamp(x))
        temp_og.rename(columns={"unix": "Datetime", "open": "Open", "high": "High", "low": "Low", "close": "Close",
                                "Volume ETH": "Volume"}, inplace=True)
        temp_og["Datetime"] = pd.to_datetime(temp_og["Datetime"])
        temp_og.sort_values("Datetime", ascending=True, inplace=True)
        temp_og = temp_og[temp_og["Datetime"] > "2017-08-18"]
        temp_og.reset_index(drop=True, inplace=True)
        data = pd.concat([data, temp_og])

    if os.path.isdir(path):
        with open(path, 'rb') as file:
            temp_og_imp = pickle.load(file)
        temp_og = pd.concat([temp_og_imp, data], axis=0)
        temp_og.drop_duplicates(keep="first", inplace=True)
        temp_og.reset_index(drop=True, inplace=True)
        with open(path, 'wb') as file:
            pickle.dump(pd.DataFrame(temp_og), file)
    else:
        with open(path, 'wb') as file:
            pickle.dump(pd.DataFrame(data), file)

    return data

###
# Calculating percentils of forward returns
###
def rolling_percentile_freturns(df, lookforward_freturn, lookback_percentile):
    df["Percentile"] = np.nan
    for i in range(len(df)):
        try:
            df.loc[df.index[i], "Percentile"] = stats.percentileofscore(df.iloc[i - lookforward_freturn - lookback_percentile + 1:i - lookforward_freturn]["FReturn"], df.iloc[i]["FReturn"])
        except:
            continue
    return df


###
#   Plot Convergence of MCMC
###
def plot_convergence_MCMC(rs, overlap=True):
    if isinstance(rs[0][1], list):
        for k in range(len(rs)):
            iters = []
            metric = []
            for i in range(len(rs[k])):
                iters.append(rs[k][i][2])
                metric.append(rs[k][i][1])
            plt.plot(iters, metric)
            if not(overlap):
                plt.show()
        plt.show()
    else:
        iters = []
        metric = []
        plt.clf()
        for i in range(len(rs)):
            iters.append(rs[i][2])
            metric.append(rs[i][1])
        plt.plot(iters, metric)
        plt.show()

###
# Percentile Scaler
###
def rolling_percentile(inp):
    df = inp[0]
    lookback_percentile = inp[1]
    columns = inp[2]
    for column in tqdm(columns):
        df[f"{column}_percentile_over_{lookback_percentile}"] = np.nan
        for i in range(len(df)):
            try:
                df.loc[df.index[i], f"{column}_percentile_over_{lookback_percentile}"] = stats.percentileofscore(
                    df.iloc[i - lookback_percentile + 1:i][column], df.iloc[i][column])/100
            except:
                continue
        df.loc[:lookback_percentile, f"{column}_percentile_over_{lookback_percentile}"] = np.nan
    return df[[f"{column}_percentile_over_{lookback_percentile}" for column in columns]]

def rolling_percentile_parallelized(df_inp, lookback_percentiles, columns):
    df = df_inp.copy()
    inputs = []
    for lookback_percentile in lookback_percentiles:
        inputs.append([df, lookback_percentile, columns])
    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results = pool.map(rolling_percentile, inputs)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    for result in results:
        df = pd.concat([df, result], axis=1)
    return df


###
#  Calculate NMI between an input and output series
###
def calc_NMI(inp, out):  #inp and out are pandas series
    inp = inp.to_numpy()
    out = out.to_numpy()
    nmi = calc_nmi(inp, out, None)
    return nmi


###
# Using NMI to select features
###

def NMI_feature_selection(df, input_cols, output_cols, threshold, plot=False):
    NMIS = []
    for col1 in tqdm(input_cols):
        for col2 in output_cols:
            NMIS.append({'Feature': col1, 'Target': col2, 'NMI': calc_NMI(df[col1], df[col2])})
    sorted_features = pd.DataFrame(NMIS)
    sorted_features = sorted_features.sort_values(by="NMI", axis=0, ascending=False).reset_index(drop=True)
    if plot:
        sorted_features["NMI"].hist()
    return list(sorted_features[sorted_features["NMI"] > threshold]["Feature"]), zip(list(sorted_features[sorted_features["NMI"] > threshold]["Feature"]),list(sorted_features[sorted_features["NMI"] > threshold]["Target"])), sorted_features


###
# Pivots on Indicator Space
###
def add_pivot_features(data, col, n, shift=True):
    data[f"{col}_TypeCurrentPivot"] = np.where(data[f"{col}_IsHighPivot"]==1, 1, np.nan)
    data[f"{col}_TypeCurrentPivot"] = np.where(data[f"{col}_IsLowPivot"]==1, -1, data[f"{col}_TypeCurrentPivot"])
    data[f"{col}_PivotValue"] = np.where(data[f"{col}_IsHighPivot"]==1, data[f"{col}"], np.nan)
    data[f"{col}_PivotValue"] = np.where(data[f"{col}_IsLowPivot"]==1, data[f"{col}"], data[f"{col}_PivotValue"])
    data[f"{col}_TypePreviousPivot"] = data[f"{col}_TypeCurrentPivot"].fillna(method='ffill').shift(1).fillna(0)
    data[f"{col}_PreviousPivotValue"] = data[f"{col}_PivotValue"].fillna(method='ffill').shift(1).fillna(0)
    data[f"{col}_TypePreviousPivot"] = np.where(np.isnan(data[f"{col}_TypeCurrentPivot"]), np.nan, data[f"{col}_TypePreviousPivot"])
    data[f"{col}_PreviousPivotValue"] = np.where(np.isnan(data[f"{col}_PivotValue"]), np.nan, data[f"{col}_PreviousPivotValue"])
    data[f"{col}_PreviousHighPivotValue"] = pd.DataFrame(np.where(((data[f"{col}_TypePreviousPivot"]==-1)),data[f"{col}_PivotValue"],np.nan)).fillna(method='ffill').shift(1)
    data[f"{col}_PreviousLowPivotValue"] = pd.DataFrame(np.where(((data[f"{col}_TypePreviousPivot"]==1)),data[f"{col}_PivotValue"],np.nan)).fillna(method='ffill').shift(1)
    data[f"{col}_DaysSincePreviousHighPivot"] = np.nan
    data[f"{col}_DaysSincePreviousLowPivot"] = np.nan
    for i in range(1,len(data)):
        if data.iloc[i][f"{col}_TypeCurrentPivot"]==1:
            data.loc[i, f"{col}_DaysSincePreviousHighPivot"] = 0
            data.loc[i, f"{col}_DaysSincePreviousLowPivot"] = data.loc[i-1, f"{col}_DaysSincePreviousLowPivot"] + (data.loc[i, "Datetime"]- data.loc[i-1, "Datetime"]).days
        elif data.iloc[i][f"{col}_TypeCurrentPivot"]==-1:
            data.loc[i, f"{col}_DaysSincePreviousHighPivot"] = data.loc[i-1, f"{col}_DaysSincePreviousHighPivot"] + (data.loc[i, "Datetime"]- data.loc[i-1, "Datetime"]).days
            data.loc[i, f"{col}_DaysSincePreviousLowPivot"] = 0
        else:
            try:
                data.loc[i, f"{col}_DaysSincePreviousHighPivot"] = data.loc[i-1, f"{col}_DaysSincePreviousHighPivot"] + (data.loc[i, "Datetime"]- data.loc[i-1, "Datetime"]).days
                data.loc[i, f"{col}_DaysSincePreviousLowPivot"] = data.loc[i-1, f"{col}_DaysSincePreviousLowPivot"] + (data.loc[i, "Datetime"]- data.loc[i-1, "Datetime"]).days
            except:
                continue
    if shift:
        data[f"{col}_TypeCurrentPivot"] = data[f"{col}_TypeCurrentPivot"].shift(n)
        data[f"{col}_PivotValue"] = data[f"{col}_PivotValue"].shift(n)
        data[f"{col}_TypePreviousPivot"] = data[f"{col}_TypePreviousPivot"].shift(n)
        data[f"{col}_PreviousPivotValue"] = data[f"{col}_PreviousPivotValue"].shift(n)
        data[f"{col}_PreviousHighPivotValue"] = data[f"{col}_PreviousHighPivotValue"].shift(n)
        data[f"{col}_PreviousLowPivotValue"] = data[f"{col}_PreviousLowPivotValue"].shift(n)
        data[f"{col}_DaysSincePreviousHighPivot"] = data[f"{col}_DaysSincePreviousHighPivot"].shift(n)
        data[f"{col}_DaysSincePreviousLowPivot"] = data[f"{col}_DaysSincePreviousLowPivot"].shift(n)
        data[f"{col}_IsHighPivot"] = data[f"{col}_IsHighPivot"].shift(n)
        data[f"{col}_IsLowPivot"] = data[f"{col}_IsLowPivot"].shift(n)

    return data

#Identifying Pivots on Indicator Space based on comparison with Last n and Next n values

def compare_for_high_pivot(inp):
    x = inp.copy()
    n = int((len(x)-1)/2)
    x = np.array(x)
    if x.std()==0:
        return np.nan
    conditionals = []
    for i in range(0,n):
        conditionals.append(x[i]<=x[i+1])
    for i in range(n,len(x)-1):
        conditionals.append(x[i]>=x[i+1])
    if (len(set(conditionals))==1)&(list(set(conditionals))[0]):
        return 1
    else:
        return np.nan

def compare_for_low_pivot(inp):
    x = inp.copy()
    n = int((len(x)-1)/2)
    x = np.array(x)
    if x.std()==0:
        return np.nan
    conditionals = []
    for i in range(0,n):
        conditionals.append(x[i]>=x[i+1])
    for i in range(n,len(x)-1):
        conditionals.append(x[i]<=x[i+1])
    if (len(set(conditionals))==1)&(list(set(conditionals))[0]):
        return 1
    else:
        return np.nan

def add_pivot_Comparison_with_values(data, col, n, shift=True, plot=True):
    data[f"{col}_IsHighPivot"] = data[col].rolling(2*n+1, center=True).apply(compare_for_high_pivot)
    data[f"{col}_IsLowPivot"] = data[col].rolling(2*n+1, center=True).apply(compare_for_low_pivot)
    data = add_pivot_features(data, col, n, shift)
    if plot:
        plt.plot(data['Datetime'], data[col], color='black', label=col)
        plt.plot(data[data[f"{col}_IsHighPivot"]==1]['Datetime'], data[data[f"{col}_IsHighPivot"]==1][f"{col}_PivotValue"], color='black', marker='o', ms=5, linestyle = 'None',mec='r')
        plt.plot(data[data[f"{col}_IsLowPivot"]==1]['Datetime'], data[data[f"{col}_IsLowPivot"]==1][f"{col}_PivotValue"], color='black', marker='o', ms=5, linestyle = 'None',mec='g')
        plt.legend()
    return data