"""
Contains code for calculating all metrics
Each function expects the backtest dataframe of the form: data[['Datetime', 'Close', 'signal', 'Return', 'S_Return', 'trade_num', 'Market_Return', 'Strategy_Return', 'Portfolio_Value']]
"""

import numpy as np
import math
import pandas as pd

def DailyHitRatio(data):
    data['Wins'] = np.where(data['S_Return'] > 0, 1, 0)
    data['Losses'] = np.where(data['S_Return'] < 0, 1, 0)
    TotalWins = data['Wins'].sum()
    TotalLosses = data['Losses'].sum()
    TotalTrades = TotalWins + TotalLosses
    if TotalTrades==0:
        DailyHitRatio = 0
    else:
        DailyHitRatio = round(TotalWins/TotalTrades, 2)
    return DailyHitRatio

def SharpeRatio(data):
    int = 6
    if round(data["S_Return"].std(), 2) == 0:
        return 0
    else:
        return (data["S_Return"].mean() - int/25200)/round(data["S_Return"].std(), 2) * (252 ** .5)

def SharpeRatio_hourly(data):
    int = 6
    if round(data["S_Return"].std(), 2) == 0:
        return 0
    else:
        return (data["S_Return"].mean() -int/(25200*24))/ round(data["S_Return"].std(), 2) * ((252*24) ** .5)

def SortinoRatio(data):
    StDev_Annualized_Downside_Return = round(data.loc[data["S_Return"] < 0, "S_Return"].std(), 2) * (252 ** .5)
    if math.isnan(StDev_Annualized_Downside_Return):
        StDev_Annualized_Downside_Return = 0.0
    if StDev_Annualized_Downside_Return != 0.0:
        SortinoRatio = (data["S_Return"].mean() - 0.06 / 252) * 252 / StDev_Annualized_Downside_Return
    else:
        SortinoRatio = 0
    return SortinoRatio

def SortinoRatio_hourly(data):
    StDev_Annualized_Downside_Return = data.loc[data["S_Return"] < 0, "S_Return"].std() * ((252*24) ** .5)
    if math.isnan(StDev_Annualized_Downside_Return):
        StDev_Annualized_Downside_Return = 0.0
    if StDev_Annualized_Downside_Return != 0.0:
        SortinoRatio = (data["S_Return"].mean() - 0.06 / (252*24)) * (252*24) / StDev_Annualized_Downside_Return
    else:
        SortinoRatio = 0
    return SortinoRatio

def CAGR(data):
    if len(data['Strategy_Return']) != 0:
        n_days = (data.Datetime.iloc[-1] - data.Datetime.iloc[0])
        CAGR = (1 + data['Strategy_Return']).iloc[-1] ** (365.25 / n_days.days) - 1
    else:
        CAGR = 0
    return CAGR

def MaxDrawdown(data):
    return  (1.0 - data['Portfolio_Value'] / data['Portfolio_Value'].cummax()).max()

def HitRatio(data):
    ecdf = data[data["signal"] == 1]
    trade_wise_results = []
    if len(ecdf) > 0:
        for i in range(max(ecdf['trade_num'])):
            trade_num = i + 1
            entry = ecdf[ecdf["trade_num"] == trade_num].iloc[0]["Close"]
            exit = ecdf[ecdf["trade_num"] == trade_num].iloc[-1]["Close"]
            trade_wise_results.append({'Trade Number': trade_num, 'Entry': entry, 'Exit': exit})
    trade_wise_results = pd.DataFrame(trade_wise_results)
    if len(trade_wise_results) > 0:
        trade_wise_results["Win/Loss"] = np.where(trade_wise_results["Exit"] > trade_wise_results["Entry"], "Win",
                                                  "Loss")
        trade_wise_results["Return on Trade"] = trade_wise_results["Exit"] / trade_wise_results["Entry"] - 1
        TotalWins = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Win"])
        TotalLosses = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"])
        TotalTrades = TotalWins + TotalLosses
        if TotalTrades == 0:
            HitRatio = 0
        else:
            HitRatio = round(TotalWins / TotalTrades, 4)
    else:
        HitRatio = 0
    return HitRatio 
                
def WinByLossRet(data):
    ecdf = data[data["signal"] == 1]
    trade_wise_results = []
    if len(ecdf) > 0:
        for i in range(max(ecdf['trade_num'])):
            trade_num = i + 1
            entry = ecdf[ecdf["trade_num"] == trade_num].iloc[0]["Close"]
            exit = ecdf[ecdf["trade_num"] == trade_num].iloc[-1]["Close"]
            trade_wise_results.append({'Trade Number': trade_num, 'Entry': entry, 'Exit': exit})
    trade_wise_results = pd.DataFrame(trade_wise_results)
    if len(trade_wise_results) > 0:
        trade_wise_results["Win/Loss"] = np.where(trade_wise_results["Exit"] > trade_wise_results["Entry"], "Win",
                                                  "Loss")
        trade_wise_results["Return on Trade"] = trade_wise_results["Exit"] / trade_wise_results["Entry"] - 1
        AvgWinRet = np.round(
            trade_wise_results[trade_wise_results["Win/Loss"] == "Win"]["Return on Trade"].mean(), 4)
        if math.isnan(AvgWinRet):
            AvgWinRet = 0.0
        AvgLossRet = np.round(trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"]["Return on Trade"].mean(), 4)
        if math.isnan(AvgLossRet):
            AvgLossRet = 0.0
        if AvgLossRet != 0:
            WinByLossRet = np.round(abs(AvgWinRet / AvgLossRet), 2)
        else:
            WinByLossRet = 100000.0
        if math.isnan(WinByLossRet):
            WinByLossRet = 0.0
        if math.isinf(WinByLossRet):
            WinByLossRet = 100000.0
    else:
        WinByLossRet = 0.0
    return WinByLossRet


def rolling_sharpe(data):
    r_window = 252
    ecdf = data[["Datetime", "S_Return"]]
    ecdf['RSharpeRatio_Series'] = (ecdf["S_Return"].rolling(window=r_window, min_periods=1).mean() - 0.06 / 252) / round(ecdf[
        "S_Return"].rolling(window=r_window, min_periods=1).std(), 2) * (252 ** .5)
    ecdf['RSharpeRatio_Series'] = ecdf['RSharpeRatio_Series'].replace(to_replace=math.inf, value=0)
    ecdf['RSharpeRatio_Series'] = ecdf['RSharpeRatio_Series'].replace(to_replace=math.nan, value=0)
    RSharpeRatio = ecdf['RSharpeRatio_Series'].mean()
    return np.float64(RSharpeRatio)

def rolling_sharpe_hourly(data):
    r_window = 252
    ecdf = data[["Datetime", "S_Return"]]
    ecdf['RSharpeRatio_Series'] = (ecdf["S_Return"].rolling(window=r_window, min_periods=1).mean() - 0.06 / (252*24)) / round(ecdf[
        "S_Return"].rolling(window=r_window, min_periods=1).std(), 2) * ((252*24) ** .5)
    ecdf['RSharpeRatio_Series'] = ecdf['RSharpeRatio_Series'].replace(to_replace=math.inf, value=0)
    ecdf['RSharpeRatio_Series'] = ecdf['RSharpeRatio_Series'].replace(to_replace=math.nan, value=0)
    RSharpeRatio = ecdf['RSharpeRatio_Series'].mean()
    return np.float64(RSharpeRatio)

def rolling_sortino(data):
    r_window = 252
    ecdf = data[["Datetime", "S_Return"]]
    ecdf["S_Return_stdev"] = ecdf["S_Return"].copy()
    ecdf["S_Return_stdev"] = np.where(ecdf["S_Return_stdev"] >= 0, ecdf["S_Return_stdev"], np.nan)
    ecdf['RStDev Annualized Downside Return_Series'] = ecdf["S_Return_stdev"].rolling(window=r_window,
                                                                                      min_periods=1).std() * (
                                                               252 ** .5)
    ecdf['RStDev Annualized Downside Return_Series'] = ecdf['RStDev Annualized Downside Return_Series'].replace(
        to_replace=math.inf, value=0)
    ecdf['RStDev Annualized Downside Return_Series'] = ecdf['RStDev Annualized Downside Return_Series'].replace(
        to_replace=math.nan, value=0)
    ecdf['RSortinoRatio_Series'] = np.where(ecdf['RStDev Annualized Downside Return_Series'] != 0.0,
                                            (ecdf["S_Return"].rolling(window=r_window,
                                                                      min_periods=1).mean() - 0.06 / 252) * 252 /
                                            ecdf['RStDev Annualized Downside Return_Series'], 0)
    RSortinoRatio = ecdf['RSortinoRatio_Series'].mean()
    return np.float64(RSortinoRatio)

def rolling_sortino_hourly(data):
    r_window = 252
    ecdf = data[["Datetime", "S_Return"]]
    ecdf["S_Return_stdev"] = ecdf["S_Return"].copy()
    ecdf["S_Return_stdev"] = np.where(ecdf["S_Return_stdev"] >= 0, ecdf["S_Return_stdev"], np.nan)
    ecdf['RStDev Annualized Downside Return_Series'] = ecdf["S_Return_stdev"].rolling(window=r_window,
                                                                                      min_periods=1).std() * (
                                                               (252*24) ** .5)
    ecdf['RStDev Annualized Downside Return_Series'] = ecdf['RStDev Annualized Downside Return_Series'].replace(
        to_replace=math.inf, value=0)
    ecdf['RStDev Annualized Downside Return_Series'] = ecdf['RStDev Annualized Downside Return_Series'].replace(
        to_replace=math.nan, value=0)
    ecdf['RSortinoRatio_Series'] = np.where(ecdf['RStDev Annualized Downside Return_Series'] != 0.0,
                                            (ecdf["S_Return"].rolling(window=r_window,
                                                                      min_periods=1).mean() - 0.06 / (252*24)) * (252*24) /
                                            ecdf['RStDev Annualized Downside Return_Series'], 0)
    RSortinoRatio = ecdf['RSortinoRatio_Series'].mean()
    return np.float64(RSortinoRatio)

def rolling_cagr(data):
    ecdf = data[["Datetime", "S_Return"]]
    ecdf["Datetime"] = pd.to_datetime(ecdf["Datetime"])
    ecdf = ecdf['S_Return'].to_frame().set_index(ecdf["Datetime"])
    ecdf['Portfolio'] = 1 + ecdf['S_Return'].expanding().sum()
    ecdf['Portfolio'] = ecdf['Portfolio']
    ecdf['Portfolio_1yr'] = ecdf['Portfolio'].to_frame().shift(365, freq='D')
    ecdf['Portfolio_1yr'] = ecdf['Portfolio_1yr'].fillna(method="ffill")
    ecdf['RCAGR_Strategy_Series'] = ecdf['Portfolio'] / ecdf['Portfolio_1yr'] - 1
    RCAGR_Strategy = ecdf['RCAGR_Strategy_Series'].mean()
    return np.float64(RCAGR_Strategy)


def maxdrawup_by_maxdrawdown(data):
    r_window = 252
    ecdf = data[["Datetime", "S_Return"]]
    ecdf['Strategy_Return'] = ecdf['S_Return'].expanding().sum()
    ecdf['Portfolio Value'] = ((ecdf['Strategy_Return'] + 1) * 1)
    ecdf['Portfolio Value'][0] = 1
    ecdf['Drawdown_Series'] = 1 - ecdf['Portfolio Value'] / np.maximum.accumulate(ecdf['Portfolio Value'])
    ecdf['Max_Drawdown_Series'] = ecdf['Drawdown_Series'].rolling(window=r_window, min_periods=1).max()
    ecdf['Drawup_Series'] = ecdf['Portfolio Value'] / np.minimum.accumulate(ecdf['Portfolio Value']) - 1
    ecdf['Max_Drawup_Series'] = ecdf['Drawup_Series'].rolling(window=r_window, min_periods=1).max()
    ecdf['Drawup/Drawdown_Series'] = ecdf['Max_Drawup_Series'] / ecdf['Max_Drawdown_Series']
    ecdf['Drawup/Drawdown_Series'] = ecdf['Drawup/Drawdown_Series'].replace(math.inf, 100)
    RDrawupDrawdown = ecdf['Drawup/Drawdown_Series'].mean()
    # ecdf['Drawdown_Series'] = 1 - ecdf['Portfolio Value'] / np.maximum.accumulate(ecdf['Portfolio Value'])
    # ecdf['Drawup_Series'] = ecdf['Portfolio Value'] / np.minimum.accumulate(ecdf['Portfolio Value']) - 1
    # RDrawupDrawdown = ecdf['Drawup_Series'].max() / ecdf['Drawdown_Series'].max()
    return np.float64(RDrawupDrawdown)


def outperformance(data):
    r_window = 365
    data["Datetime"] = pd.to_datetime(data["Datetime"])
    ecdf1 = data['S_Return'].to_frame().set_index(data["Datetime"])
    ecdf2 = data['Return'].to_frame().set_index(data["Datetime"])
    ecdf1['Portfolio'] = 1 + ecdf1['S_Return'].expanding().sum()
    ecdf1['Portfolio_1yr'] = ecdf1['Portfolio'].to_frame().shift(r_window, freq='D')
    ecdf2['Portfolio'] = 1 + ecdf2['Return'].expanding().sum()
    ecdf2['Portfolio_1yr'] = ecdf2['Portfolio'].to_frame().shift(r_window, freq='D')
    ecdf1['Portfolio_1yr'] = ecdf1['Portfolio_1yr'].fillna(method="ffill")
    ecdf1['RCAGR_Strategy_Series'] = ecdf1['Portfolio'] / ecdf1['Portfolio_1yr'] - 1
    ecdf2['Portfolio_1yr'] = ecdf2['Portfolio_1yr'].fillna(method="ffill")
    ecdf2['RCAGR_Market_Series'] = ecdf2['Portfolio'] / ecdf2['Portfolio_1yr'] - 1
    RCAGR_Strategy = ecdf1['RCAGR_Strategy_Series'].mean()
    RCAGR_Market = ecdf2['RCAGR_Market_Series'].mean()
    ROutperformance = RCAGR_Strategy - RCAGR_Market
    return np.float64(ROutperformance)

def total_return(data):
    initial_portfolio = list(data['Portfolio_Value'])[0]
    final_portfolio = list(data['Portfolio_Value'])[-1]
    total_ret = (final_portfolio - initial_portfolio) / initial_portfolio * 100
    return total_ret

def rolling_yearly_return_median(data):
    r_window = 252
    data['Portfolio_Value_1yr'] = data['Portfolio_Value'].shift(r_window)
    data['Portfolio_1yr_Ret'] = (data['Portfolio_Value'] - data['Portfolio_Value_1yr']) / data['Portfolio_Value_1yr']
    median_ret = data['Portfolio_1yr_Ret'].median()
    return median_ret

    