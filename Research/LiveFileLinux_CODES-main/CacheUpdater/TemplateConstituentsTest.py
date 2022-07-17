import multiprocessing
import os
import pickle
import warnings

warnings.filterwarnings('ignore')
import zipfile
import yfinance as yf
import math
import numpy as np
import investpy
from datetime import date, datetime
from datetime import timedelta
import pandas as pd
from scipy.stats import percentileofscore
import scipy


def add_fisher(temp):
    for f_look in range(50, 400, 20):
        temp[f'Fisher{f_look}'] = fisher(temp, f_look)
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


def get_data_investpy(symbol, country, from_date, to_date):
    find = investpy.search.search_quotes(text=symbol, products=["stocks", "etfs", "indices", "currencies"])
    for f in find:
        # print( f )
        if f.symbol.lower() == symbol.lower() and f.country.lower() == country.lower():
            break
    if f.symbol.lower() != symbol.lower():
        return None
    ret = f.retrieve_historical_data(from_date=from_date, to_date=to_date)
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


def get_data(ticker, api, country):
    if api == "yfinance":

        temp_og = yf.download(ticker, start='2007-01-01', end=str(date.today() + timedelta(1)))
        if len(temp_og) == 0:
            temp_og = yf.download(ticker, start='2007-01-01', end=str(date.today()))
        temp_og.reset_index(inplace=True)
        temp_og.drop(['Adj Close'], axis=1, inplace=True)
        if ticker == "GOLDBEES.NS":
            temp_og = temp_og.loc[temp_og["Close"] > 1]
        temp_og = add_fisher(temp_og)

    if api == "investpy":
        temp_og = get_data_investpy(symbol=ticker, country=country, from_date="01/01/2007",
                                    to_date=(date.today() + timedelta(1)).strftime("%d/%m/%Y"))
        temp_og.reset_index(inplace=True)
        temp_og = add_fisher(temp_og)

    # if api == "reuters":
    #     temp_og = ek.get_timeseries(ticker, start_date='2007-01-01', end_date=str(date.today() + timedelta(1)))
    #     temp_og.reset_index(inplace=True)
    #     temp_og.rename(columns={"HIGH": "High", "CLOSE": "Close", "LOW": "Low", "OPEN": "Open", "VOLUME": "Volume"},
    #                    inplace=True)
    #     temp_og.drop(['COUNT'], axis=1, inplace=True)

    return temp_og


def get_data_alpha(ticker):

    if ticker == "GOLDBEES.NS":
        ticker_inv = "GBES"
    if ticker == "^NSEI":
        ticker_inv = "NSEI"

    temp_og = get_data(ticker, "yfinance", "india")

    if ticker == "GOLDBEES.NS":
        temp_og = temp_og[temp_og["Close"] < 100]

    today_data = yf.download(ticker, start=str(date.today() - timedelta(days=1)), interval="1m")

    if len(today_data)==0:
        today_time_close = temp_og.iloc[-1]["Date"]
    else:
        today_data.reset_index(inplace=True)
        today_data.drop(['Adj Close'], axis=1, inplace=True)
        if ticker == "GOLDBEES.NS":
            today_data = today_data.loc[today_data["Close"] > 1]

        today_time_close = today_data.iloc[-1]["Datetime"]

        temp_og = pd.concat([temp_og, pd.DataFrame([{"Date": pd.Timestamp(year=today_data.iloc[-1]["Datetime"].year,
                                                                          month=today_data.iloc[-1]["Datetime"].month,
                                                                          day=today_data.iloc[-1]["Datetime"].day),\
                                                     "Open": today_data.iloc[-1]["Open"],
                                                     "Close": today_data.iloc[-1]["Close"],
                                                     "High": today_data.iloc[-1]["High"],
                                                     "Low": today_data.iloc[-1]["Low"],
                                                     "Volume": today_data.iloc[-1]["Volume"]}])],axis=0).reset_index().drop(['index'], axis=1)
        temp_og.drop_duplicates(subset="Date",
                                keep='first', inplace=True)
    temp_og = add_fisher(temp_og)
    return temp_og, today_time_close


class FISHER_bounds_strategy_opt:

    def __init__(self, data, zone_low, zone_high, start=None, end=None):
        self.zl = zone_low
        self.zh = zone_high
        self.data = data  # the dataframe
        # self.data['yr'] = self.data['Date'].dt.year
        # self.n_days = (self.data.Date.iloc[-1] - self.data.Date.iloc[0])
        self.start = start  # the beginning date of the sample period
        self.end = end  # the ending date of the sample period

    def generate_signals(self, charts=True):
        self.data = self.data.loc[(self.data.fisher != 0)]
        self.data["fisher_lag"] = self.data.fisher.shift(1)
        self.data["lb"] = self.zl
        self.data["ub"] = self.zh
        self.data.dropna()
        self.data.reset_index(inplace=True)
        self.data = self.data.drop(['index'], axis=1)

        ## creating signal
        ##Upper and lower bounds
        buy_mask = (self.data["fisher"] > self.data["lb"]) & (self.data["fisher_lag"] < self.data["lb"])
        sell_mask = ((self.data["fisher"] < self.data["ub"]) & (self.data["fisher_lag"] > self.data["ub"])) | (
                self.data["fisher"] < np.minimum(self.data["lb"], self.data["ub"]))

        bval = +1
        sval = 0  # -1 if short selling is allowed, otherwise 0

        self.data['signal_bounds'] = np.nan
        self.data.loc[buy_mask, 'signal_bounds'] = bval
        self.data.loc[sell_mask, 'signal_bounds'] = sval
        # initialize with long
        self.data["signal_bounds"][0] = 1

        self.data.signal_bounds = self.data.signal_bounds.fillna(method="ffill")

        self.data["signal"] = self.data.signal_bounds

        # Closing positions at end of time period
        self.data["signal"][-1:] = 0

        mask = ((self.data.signal == 1) & (self.data.signal.shift(1) == 0)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

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
        self.data["S_Return"] = self.data["S_Return"] + (self.int / 25200) * (1 - self.data['signal'].shift(1))
        self.data['Market_Return'] = self.data['Return'].expanding().sum()
        self.data['Strategy_Return'] = self.data['S_Return'].expanding().sum()
        self.data['Portfolio Value'] = ((self.data['Strategy_Return'] + 1) * self.allocation)
        self.data['Wins'] = np.where(self.data['S_Return'] > 0, 1, 0)
        self.data['Losses'] = np.where(self.data['S_Return'] < 0, 1, 0)

        return self.data[['Date', 'Close', 'signal', 'S_Return', 'trade_num', 'Return']]


def get_strategies_brute_force(inp):
    def get_equity_curve_embeddings(*args):
        f_look = args[0]
        f_look = 1 * round(f_look / 1)
        lb = round(10 * args[1]) / 10
        ub = round(10 * args[2]) / 10

        temp["fisher"] = temp[f'Fisher{f_look}']

        test_strategy = FISHER_bounds_strategy_opt(temp, lb, ub)
        _ = test_strategy.generate_signals()
        ec = test_strategy.signal_performance(10000, 6)
        return ec

    def AvgWinLoss(x, y, bins):
        ecdf = x[["S_Return", "Close", "signal", "trade_num"]]
        ecdf = ecdf[ecdf["signal"] == 1]
        trade_wise_results = []
        for i in range(max(ecdf['trade_num'])):
            trade_num = i + 1
            entry = ecdf[ecdf["trade_num"] == trade_num].iloc[0]["Close"]
            exit = ecdf[ecdf["trade_num"] == trade_num].iloc[-1]["Close"]
            trade_wise_results.append({'Trade Number': trade_num, 'Entry': entry, 'Exit': exit})
        trade_wise_results = pd.DataFrame(trade_wise_results)
        d_tp = {}
        if len(trade_wise_results) > 0:
            trade_wise_results["Win/Loss"] = np.where(trade_wise_results["Exit"] > trade_wise_results["Entry"], "Win",
                                                      "Loss")
            trade_wise_results["Return on Trade"] = trade_wise_results["Exit"] / trade_wise_results["Entry"] - 1
            d_tp["TotalWins"] = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Win"])
            d_tp["TotalLosses"] = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"])
            d_tp['TotalTrades'] = d_tp["TotalWins"] + d_tp["TotalLosses"]
            if d_tp['TotalTrades'] == 0:
                d_tp['HitRatio'] = 0
            else:
                d_tp['HitRatio'] = round(d_tp['TotalWins'] / d_tp['TotalTrades'], 4)
            d_tp['AvgWinRet'] = np.round(
                trade_wise_results[trade_wise_results["Win/Loss"] == "Win"]["Return on Trade"].mean(), 4)
            if math.isnan(d_tp['AvgWinRet']):
                d_tp['AvgWinRet'] = 0.0
            d_tp['AvgLossRet'] = np.round(
                trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"]["Return on Trade"].mean(), 4)
            if math.isnan(d_tp['AvgLossRet']):
                d_tp['AvgLossRet'] = 0.0
            if d_tp['AvgLossRet'] != 0:
                d_tp['WinByLossRet'] = np.round(abs(d_tp['AvgWinRet'] / d_tp['AvgLossRet']), 2)
            else:
                d_tp['WinByLossRet'] = 0
            if math.isnan(d_tp['WinByLossRet']):
                d_tp['WinByLossRet'] = 0.0
            if math.isinf(d_tp['WinByLossRet']):
                d_tp['WinByLossRet'] = 0.0
        else:
            d_tp["TotalWins"] = 0
            d_tp["TotalLosses"] = 0
            d_tp['TotalTrades'] = 0
            d_tp['HitRatio'] = 0
            d_tp['AvgWinRet'] = 0
            d_tp['AvgLossRet'] = 0
            d_tp['WinByLossRet'] = 0

        return np.float64(d_tp['WinByLossRet'])

    date_i = inp[0]
    dates = inp[1]
    temp_og = inp[2]
    train_months = inp[3]

    temp = temp_og.loc[(temp_og["Date"] > str(dates[date_i])) & (temp_og["Date"] < str(
        dates[date_i + int(train_months / 3)]))].reset_index().drop(['index'], axis=1)
    res = pd.DataFrame(columns=["Lookback", "Low Bound", "High Bound", "AvgWinLoss"])

    for f_look in range(50, 70, 20):
        max_metric = 0
        for lb in np.round(np.arange(-1, 1, 2), decimals=1):
            for ub in np.round(np.arange(-1, 1, 2), decimals=1):
                metric = AvgWinLoss(get_equity_curve_embeddings(f_look, lb, ub), 0, 0)
                if metric > max_metric:
                    max_metric = metric
                    res_iter = pd.DataFrame(
                        [{"Lookback": f_look, "Low Bound": lb, "High Bound": ub, "AvgWinLoss": metric}])
                    res = pd.concat([res, res_iter], axis=0)

    res.drop_duplicates(keep='first', inplace=True, ignore_index=True)
    res.sort_values("AvgWinLoss", axis=0, ascending=False, inplace=True)
    res["Optimization_Years"] = train_months / 12
    res = res.reset_index().drop(['index'], axis=1)
    return (date_i, res)


def select_all_strategies(train_monthsf, datesf, temp_ogf, ticker, save=True):
    inputs = []
    for date_i in range(len(datesf) - (int(train_monthsf / 3) + 1)):
        inputs.append([date_i, datesf, temp_ogf, train_monthsf])

    if len(inputs)==0:
        res_test_update = [pd.DataFrame(columns=["Lookback", "Low Bound", "High Bound", "AvgWinLoss", \
                                             "Optimization_Years"])]

    else:

        results = [get_strategies_brute_force(inputs[-1])]

        res_test_update = [pd.DataFrame(columns=["Lookback", "Low Bound", "High Bound", "AvgWinLoss", \
                                                 "Optimization_Years"])]

        res_test_update[0] = pd.concat([res_test_update[0], results[0][1].reset_index().drop(['index'], axis=1)], axis=0)

    if save == True:

        if not os.path.exists(f'{ticker}/SelectedStrategies'):
            os.makedirs(f'{ticker}/SelectedStrategies')

        res_test = [res_test_update[-1]]
        with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_{int(train_monthsf / 12)}_All_Strategies.pkl',
                  'wb') as file:
            pickle.dump(res_test, file)
    return res_test


def select_strategies_from_corr_filter(res_testf2, res_testf4, res_testf8, datesf, temp_ogf, num_opt_periodsf,
                                       num_strategiesf, ticker, save=True):
    train_monthsf = 24  # minimum optimization lookback
    res_total = [None]

    if num_opt_periodsf == 1:
        res_total[0] = pd.concat([res_testf2[0]], axis=0)
    if num_opt_periodsf == 2:
        res_total[0] = pd.concat([res_testf2[0], res_testf4[0]], axis=0)
    if num_opt_periodsf == 3:
        res_total[0] = pd.concat([res_testf2[0], res_testf4[0], res_testf8[0]], axis=0)
    res_total[0] = res_total[0].reset_index().drop(['index'], axis=1)

    ss_test_update = [None]
    res_test_update = [None]
    inputs = []
    for date_i in range(len(datesf) - (int(train_monthsf / 3) + 1)):
        inputs.append([date_i, datesf, temp_ogf, res_total, num_strategiesf, train_monthsf])

    results_filtered = [corr_sortino_filter(inputs[-1])]

    ss_test_update[0] = results_filtered[0][1]
    res_test_update[0] = results_filtered[0][2]

    if save == True:
        ss_test = [ss_test_update[-1]]
        res_test = [res_test_update[-1]]

        with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{num_opt_periodsf}_Selected_Strategies_ss.pkl',
                  'wb') as file:
            pickle.dump(ss_test, file)
        with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{num_opt_periodsf}_Selected_Strategies_res.pkl',
                  'wb') as file:
            pickle.dump(res_test, file)

    return ss_test, res_test


def corr_sortino_filter(inp):
    date_i = inp[0]
    dates = inp[1]
    temp_og = inp[2]
    res_total = inp[3]
    num_strategies = inp[4]
    train_monthsf = inp[5]

    temp = temp_og.loc[(temp_og["Date"] > str(dates[date_i])) & (
                temp_og["Date"] < str(dates[date_i + (int(train_monthsf / 3) + 1)]))].reset_index().drop(['index'],
                                                                                                         axis=1)
    res = res_total[0]
    x, y = corr_filter(temp, res, dates, date_i, num_strategies, train_monthsf)
    return date_i, x, y


def corr_filter(temp, res, dates, date_i, num_strategies, train_monthsf):
    res.sort_values("AvgWinLoss", axis=0, ascending=False, inplace=True)
    res.reset_index().drop(['index'], axis=1)
    returns, _ = top_n_strat_params_rolling(temp, res, to_train=True, num_of_strat=len(res),
                                            split_date=str(dates[date_i + int(train_monthsf / 3)]))
    if returns.empty:
        return [], pd.DataFrame(columns=["Lookback", "Low Bound", "High Bound", "AvgWinLoss", "Optimization_Years"])
    corr_mat = returns.corr()
    first_selected_strategy = 'Strategy1'
    selected_strategies = strategy_selection(returns, corr_mat, num_strategies, first_selected_strategy)
    params = selected_params(selected_strategies, res)
    res = params.drop(["Name"], axis=1)
    return (selected_strategies, res)


def selected_params(selected_strategies, res):
    selected_params = []
    for strategy in selected_strategies:
        selected_params.append(
            {"Name": strategy, "Lookback": res.iloc[int(strategy[8:]) - 1]["Lookback"],
             "Low Bound": res.iloc[int(strategy[8:]) - 1]["Low Bound"],
             "High Bound": res.iloc[int(strategy[8:]) - 1]["High Bound"],
             # "Sortino": res.iloc[int(strategy[8:])-1]["Sortino"],
             "AvgWinLoss": res.iloc[int(strategy[8:]) - 1]["AvgWinLoss"],
             "Optimization_Years": res.iloc[int(strategy[8:]) - 1]["Optimization_Years"]})
    selected_params = pd.DataFrame(selected_params)
    return selected_params


def strategy_selection(returns, corr_mat, num_strat, first_selected_strategy):
    strategies = [column for column in returns]
    selected_strategies = [first_selected_strategy]
    strategies.remove(first_selected_strategy)
    last_selected_strategy = first_selected_strategy

    while len(selected_strategies) < num_strat:
        corrs = corr_mat.loc[strategies][last_selected_strategy]
        corrs = corrs.loc[corrs > 0.9]
        strategies = [st for st in strategies if st not in corrs.index.to_list()]

        if len(strategies) == 0:
            break

        strat = strategies[0]

        selected_strategies.append(strat)
        strategies.remove(strat)
        last_selected_strategy = strat

    return selected_strategies


def top_n_strat_params_rolling(temp, res, to_train, num_of_strat, split_date):
    if len(res) > 0:
        for i in range(num_of_strat):
            f_look = res.iloc[i, 0]
            bf = res.iloc[i, 1]
            sf = res.iloc[i, 2]
            temp["fisher"] = temp[f'Fisher{int(f_look)}']
            if to_train:
                train = temp.loc[(temp["Date"] <= split_date)].reset_index().drop(['index'], axis=1)
            else:
                train = temp.loc[temp["Date"] > split_date].reset_index().drop(['index'], axis=1)
            test_strategy = FISHER_bounds_strategy_opt(train, zone_low=bf, zone_high=sf)
            dummy_signal = test_strategy.generate_signals()
            dummy = test_strategy.signal_performance(10000, 6)
            if i == 0:
                strat_sig_returns = dummy['S_Return'].to_frame().rename(
                    columns={'S_Return': f'Strategy{i + 1}'}).set_index(dummy["Date"])
                strat_sig = dummy_signal['signal'].to_frame().rename(columns={'signal': f'Strategy{i + 1}'}).set_index(
                    dummy_signal["Date"])
                # fisher_test = temp["fisher"].to_frame().rename(columns={"fisher": f'Fisher{asset}{i + 1}'}).set_index(temp["Date"])
            else:
                strat_sig_returns = pd.merge(strat_sig_returns, (
                    dummy['S_Return'].to_frame().rename(columns={'S_Return': f'Strategy{i + 1}'}).set_index(
                        dummy["Date"])), left_index=True, right_index=True)
                strat_sig = pd.concat([strat_sig, (
                    dummy_signal['signal'].to_frame().rename(columns={'signal': f'Strategy{i + 1}'}).set_index(
                        dummy_signal["Date"]))], axis=1)
                # fisher_test = pd.concat([fisher_test, (temp["fisher"].to_frame().rename(columns={'fisher': f'Fisher{asset}{i + 1}'}).set_index(temp["Date"]))], axis = 1)
            # strat_sig_returns = pd.merge(strat_sig_returns,dummy['S_Return'].to_frame().rename(columns = {'S_Return':f'Strategy{i + 1}'}).set_index(dummy["Date"]), left_index=True, right_index=True)
        # return dummy
        return strat_sig_returns, strat_sig  # , fisher_test
    else:
        return pd.DataFrame(), pd.DataFrame()


def get_data_constituents(ticker, api, country):
    if api == "yfinance":

        temp_og = yf.download(ticker, start='2007-01-01', end=str(date.today() + timedelta(1)))
        if len(temp_og) == 0:
            temp_og = yf.download(ticker, start='2007-01-01', end=str(date.today()))
        temp_og.reset_index(inplace=True)
        temp_og.drop(['Adj Close'], axis=1, inplace=True)
        if ticker == "GOLDBEES.NS":
            temp_og = temp_og.loc[temp_og["Close"] > 1]
        temp_og = add_fisher(temp_og)

    if api == "investpy":
        temp_og = get_data_investpy(symbol=ticker, country=country, from_date="01/07/2007",
                                    to_date=(date.today() + timedelta(1)).strftime("%d/%m/%Y"))
        temp_og.reset_index(inplace=True)
        temp_og = add_fisher(temp_og)

    # if api == "reuters":
    #     temp_og = ek.get_timeseries(ticker, start_date='2007-01-01', end_date=str(date.today() + timedelta(1)))
    #     temp_og.reset_index(inplace=True)
    #     temp_og.rename(columns={"HIGH": "High", "CLOSE": "Close", "LOW": "Low", "OPEN": "Open", "VOLUME": "Volume"},
    #                    inplace=True)
    #     temp_og.drop(['COUNT'], axis=1, inplace=True)

    return temp_og


def get_data_alpha_investpy_yf(ticker, ticker_yfinance):
    if ticker == "GOLDBEES.NS":
        ticker_inv = "GBES"
    if ticker == "^NSEI":
        ticker_inv = "NSEI"

    temp_og = get_data_constituents(ticker, "investpy", "india")

    if ticker == "GOLDBEES.NS":
        temp_og = temp_og[temp_og["Close"] < 100]

    today_data = yf.download(ticker_yfinance, start=str(date.today() - timedelta(days=1)), interval="1m")

    if len(today_data)==0:
        today_time_close = temp_og.iloc[-1]["Date"]
    else:
        today_data.reset_index(inplace=True)
        today_data.drop(['Adj Close'], axis=1, inplace=True)
        if ticker == "GOLDBEES.NS":
            today_data = today_data.loc[today_data["Close"] > 1]

        today_time_close = today_data.iloc[-1]["Datetime"]

        temp_og = pd.concat([temp_og, pd.DataFrame([{"Date": pd.Timestamp(year=today_data.iloc[-1]["Datetime"].year,
                                                                          month=today_data.iloc[-1]["Datetime"].month,
                                                                          day=today_data.iloc[-1]["Datetime"].day), \
                                                     "Open": today_data.iloc[-1]["Open"],
                                                     "Close": today_data.iloc[-1]["Close"],
                                                     "High": today_data.iloc[-1]["High"],
                                                     "Low": today_data.iloc[-1]["Low"],
                                                     "Volume": today_data.iloc[-1]["Volume"]}])],
                            axis=0).reset_index().drop(['index'], axis=1)
        temp_og.drop_duplicates(subset="Date",
                                keep='first', inplace=True)
    temp_og = add_fisher(temp_og)
    return temp_og, today_time_close


def create_final_signal_weights(signal, params, weights, nos):
    params = params[:nos]
    for i in range(len(params)):
        if i == 0:
            signals = signal[params.iloc[i]["Name"]].to_frame().rename(
                columns={'params.iloc[i]["Name"]': f'Signal{i + 1}'})
        else:
            signals = pd.merge(signals, signal[params.iloc[i]["Name"]].to_frame().rename(
                columns={'params.iloc[i]["Name"]': f'Signal{i + 1}'}), left_index=True, right_index=True)
            # signalsg = pd.concat([signalsg, signalg[paramsg.iloc[i]["Name"]].to_frame().rename(columns={'paramsg.iloc[i]["Name"]': f'Signal{i + 1}'})],axis=1)

    sf = pd.DataFrame(np.dot(np.where(np.isnan(signals), 0, signals), weights))
    # return sf.set_index(signals.index).rename(columns={0: 'signal'})

    return pd.DataFrame(np.where(sf > 0.5, 1, 0)).set_index(signals.index).rename(columns={0: 'signal'})


class FISHER_MCMC:

    def __init__(self, data, signals, start=None, end=None):
        self.signals = signals
        self.data = data  # the dataframe
        self.start = start  # the beginning date of the sample period
        self.end = end  # the ending date of the sample period

    def generate_signals(self, charts=True):
        self.data["signal"] = self.signals

        # Closing positions at end of time period
        self.data["signal"][-1:] = 0

        mask = ((self.data.signal == 1) & (self.data.signal.shift(1) == 0)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

    def signal_performance(self, allocation, interest_rate):
        self.allocation = allocation
        self.int = interest_rate
        self.data = self.data.reset_index().rename(columns={'index': 'Date'})
        # self.data['yr'] = self.data['Date'].dt.year
        # self.n_days = (self.data.Date.iloc[-1] - self.data.Date.iloc[0])

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

        return self.data[['Date', 'Close', 'signal', 'S_Return', 'trade_num', 'Return']]


def optimize_weights_live(input):
    def get_equity_curve(*args):
        weights = []
        for weight in args:
            weights.append(weight)
        # weights.append((1-sum(weights)))
        weights = pd.DataFrame(weights)
        weights = weights / weights.sum()
        signal_final = create_final_signal_weights(signal, params, weights, num_strategies)
        inp = pd.merge(train.set_index(train["Date"]), signal_final, left_index=True, right_index=True)
        test_strategy = FISHER_MCMC(inp.drop(['signal', 'Date'], axis=1), inp['signal'])
        _ = test_strategy.generate_signals()
        ec = test_strategy.signal_performance(10000, 6)

        return ec

    def sortino(x, y, bins):
        ecdf = x[["S_Return"]]
        stdev_down = ecdf.loc[ecdf["S_Return"] < 0, "S_Return"].std() * (252 ** .5)
        if math.isnan(stdev_down):
            stdev_down = 0.0
        if stdev_down != 0.0:
            sortino = (ecdf["S_Return"].mean() - 0.06 / 252) * 252 / stdev_down
        else:
            sortino = 0
        return np.float64(sortino)

    def sharpe(x, y, bins):
        ecdf = x[["S_Return"]]

        stdev = ecdf["S_Return"].std() * (252 ** .5)
        if math.isnan(stdev):
            stdev = 0.0
        if stdev != 0.0:
            sharpe = (ecdf["S_Return"].mean() - 0.06 / 252) * 252 / stdev
        else:
            sharpe = 0
        return np.float64(sharpe)

    def rolling_sharpe(x, y, bins):
        r_window = 252
        ecdf = x[["Date", "S_Return"]]
        ecdf['RSharpeRatio_Series'] = (ecdf["S_Return"].rolling(window=r_window, min_periods=1).mean() - 0.06 / 252) / \
                                      ecdf[
                                          "S_Return"].rolling(window=r_window, min_periods=1).std() * (252 ** .5)
        ecdf['RSharpeRatio_Series'] = ecdf['RSharpeRatio_Series'].replace(to_replace=math.inf, value=0)
        ecdf['RSharpeRatio_Series'] = ecdf['RSharpeRatio_Series'].replace(to_replace=math.nan, value=0)
        RSharpeRatio = ecdf['RSharpeRatio_Series'].mean()
        return np.float64(RSharpeRatio)

    def rolling_sortino(x, y, bins):
        r_window = 252
        ecdf = x[["Date", "S_Return"]]
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

    def rolling_cagr(x, y, bins):
        ecdf = x[["Date", "S_Return"]]
        ecdf = ecdf['S_Return'].to_frame().set_index(ecdf["Date"])
        ecdf['Portfolio'] = 1 + ecdf['S_Return'].expanding().sum()
        ecdf['Portfolio'] = ecdf['Portfolio']
        ecdf['Portfolio_1yr'] = ecdf['Portfolio'].to_frame().shift(365, freq='D')
        ecdf['Portfolio_1yr'] = ecdf['Portfolio_1yr'].fillna(method="ffill")
        ecdf['RCAGR_Strategy_Series'] = ecdf['Portfolio'] / ecdf['Portfolio_1yr'] - 1
        RCAGR_Strategy = ecdf['RCAGR_Strategy_Series'].mean()
        return np.float64(RCAGR_Strategy)

    def maxdrawup_by_maxdrawdown(x, y, bins):
        r_window = 252
        ecdf = x[["Date", "S_Return"]]
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

    def outperformance(x, y, bins):
        r_window = 252
        ecdf1 = x['S_Return'].to_frame().set_index(x["Date"])
        ecdf2 = x['Return'].to_frame().set_index(x["Date"])
        ecdf1['Portfolio'] = 1 + ecdf1['S_Return'].expanding().sum()
        ecdf1['Portfolio_1yr'] = ecdf1['Portfolio'].to_frame().shift(365, freq='D')
        ecdf2['Portfolio'] = 1 + ecdf2['Return'].expanding().sum()
        ecdf2['Portfolio_1yr'] = ecdf2['Portfolio'].to_frame().shift(365, freq='D')
        ecdf1['Portfolio_1yr'] = ecdf1['Portfolio_1yr'].fillna(method="ffill")
        ecdf1['RCAGR_Strategy_Series'] = ecdf1['Portfolio'] / ecdf1['Portfolio_1yr'] - 1
        ecdf2['Portfolio_1yr'] = ecdf2['Portfolio_1yr'].fillna(method="ffill")
        ecdf2['RCAGR_Market_Series'] = ecdf2['Portfolio'] / ecdf2['Portfolio_1yr'] - 1
        RCAGR_Strategy = ecdf1['RCAGR_Strategy_Series'].mean()
        RCAGR_Market = ecdf2['RCAGR_Market_Series'].mean()
        ROutperformance = RCAGR_Strategy - RCAGR_Market
        return np.float64(ROutperformance)

    def prior(params):
        return 1

    date_i = input[0]
    dates = input[1]
    temp_og = input[2]
    ss_test = input[3]
    res_test = input[4]
    num_strategies = input[5]
    metric = input[6]
    recalib_months = input[7]
    dates_all = input[8]

    if (date_i - int(24 / recalib_months)) < 0:
        temp = temp_og.loc[
            (temp_og["Date"] > str(dates_all[dates_all.index(dates[date_i]) - int(24 / 3)])) & (
                    temp_og["Date"] < str(dates[date_i + 1]))].reset_index().drop(
            ['index'], axis=1)
    else:
        temp = temp_og.loc[
            (temp_og["Date"] > str(dates[date_i - int(24 / recalib_months)])) & (
                    temp_og["Date"] < str(dates[date_i + 1]))].reset_index().drop(
            ['index'], axis=1)

    train = temp.loc[(temp["Date"] < str(dates[date_i]))].reset_index().drop(['index'], axis=1)
    test = temp.loc[(temp["Date"] >= str(dates[date_i]))].reset_index().drop(['index'], axis=1)

    if len(ss_test[0]) > 0:
        if len(ss_test[0]) > num_strategies:
            selected_strategies = ss_test[0][:num_strategies]
        else:
            selected_strategies = ss_test[0]

        if len(res_test[0]) > num_strategies:
            res = res_test[0][:num_strategies]
        else:
            res = res_test[0]

        # print("Optimizing weights")
        strategies = ["Strategy" + str(i) for i in range(1, len(res) + 1)]
        params = selected_params(strategies, res)

        _, signal = top_n_strat_params_rolling(temp, res, to_train=True, num_of_strat=len(selected_strategies),
                                               split_date=str(dates[date_i]))
        # print("Running MCMC")
        guess = (0.5 * np.ones([1, len(selected_strategies)])).tolist()[0]
        num_iterations = 10

        if len(guess) > 1:
            if metric == 'rolling_sharpe':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=num_iterations, prior=prior,
                          optimize_fn=rolling_sharpe, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'rolling_sortino':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=num_iterations, prior=prior,
                          optimize_fn=rolling_sortino, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'rolling_cagr':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=num_iterations, prior=prior,
                          optimize_fn=rolling_cagr, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'maxdrawup_by_maxdrawdown':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=num_iterations, prior=prior,
                          optimize_fn=maxdrawup_by_maxdrawdown, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'outperformance':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=num_iterations, prior=prior,
                          optimize_fn=outperformance, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            # Printing results:
            weights = []
            for weight in mc.analyse_results(rs, top_n=1)[0][0]:
                weights.append(weight)
            weights = pd.DataFrame(weights)
            weights = weights / weights.sum(axis=0)

        else:
            weights = pd.DataFrame([1])

    else:
        weights = pd.DataFrame()

    return date_i, weights


class MCMC():

    def __init__(self, alpha_fn, alpha_fn_params_0, target, num_iters, prior, burn=0.00, optimize_fn=None,
                 lower_limit=-10000,
                 upper_limit=10000):
        self.alpha_fn = alpha_fn
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.initial_params = alpha_fn_params_0
        self.target = target
        if optimize_fn is not None:
            self.optimize_fn = optimize_fn
        else:
            self.optimize_fn = MCMC.nmi
        self.num_iters = num_iters
        self.burn = burn
        self.prior = prior

    def transition_fn(self, cur, iter):

        # print("Inside transition_fn")

        std = self.std_guess(iter, self.num_iters)
        new_guesses = []
        for c, s in zip(cur, std):

            # print("Inside for loop")

            loop = True
            while loop:

                # print("Inside while loop")

                new_guess = np.random.normal(c, s, (1,))

                # print(f"New guess {new_guess}")
                # print(f"c: {c}")
                # print(f"s: {s}")

                if new_guess[0] <= self.upper_limit and new_guess[0] >= self.lower_limit:
                    new_guesses.append(new_guess[0])
                    loop = False
        return list(new_guesses)

    @staticmethod
    def __to_percentile(arr):
        pct_arr = []
        for idx in range(0, len(arr)):
            pct_arr.append(round(percentileofscore(np.array(arr), arr[idx])))
        return pct_arr

    @staticmethod
    def __shan_entropy(c):
        c_normalized = c / float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = -sum(c_normalized * np.log2(c_normalized))
        return H

    @staticmethod
    def nmi(X, Y, bins):

        c_XY = np.histogram2d(X, Y, bins)[0]
        c_X = np.histogram(X, bins)[0]
        c_Y = np.histogram(Y, bins)[0]

        H_X = MCMC.__shan_entropy(c_X)
        H_Y = MCMC.__shan_entropy(c_Y)
        H_XY = MCMC.__shan_entropy(c_XY)

        NMI = 2 * (H_X + H_Y - H_XY) / (H_X + H_Y)
        return NMI

    def do_step(self, iter, prev_params, prev_nmi):

        # print("Inside do_step")

        next_params = self.transition_fn(prev_params, iter)

        if self.prior(next_params) != 0:

            # y_pred = MCMC.__to_percentile( self.alpha_fn( *next_params ) )
            # print( y_pred )
            # y_true = MCMC.__to_percentile( self.target )
            # print( y_true )

            X = self.alpha_fn(*next_params)
            Y = self.target
            next_nmi = self.optimize_fn(X, Y, round(len(X) / 5))

            # print("Iter:", iter)
            # print( "Next MI:" + str( next_nmi ))

            if next_nmi > prev_nmi:
                # print( "Exploit:")
                # print( next_nmi )
                # print( next_params )
                # print( self.std_guess(iter, self.num_iters))
                # print( self.explore_factor(iter, self.num_iters))
                return [next_params, next_nmi]
            else:
                ratio = next_nmi / prev_nmi

                uniform = np.random.uniform(0, 1)
                if ratio > uniform * self.explore_factor(iter, self.num_iters):
                    # print("Explore:")
                    # print(next_nmi)
                    # print(next_params)
                    # print(self.std_guess(iter, self.num_iters))
                    # print(self.explore_factor(iter, self.num_iters))
                    return [next_params, next_nmi]
                else:
                    return [prev_params, prev_nmi]
        else:
            return [prev_params, prev_nmi]

    def optimize(self):

        prev_params = self.initial_params
        [prev_params, prev_nmi] = self.do_step(0, prev_params, -1)
        all_results = []

        for i in range(0, self.num_iters):
            # print( i )
            # if round( i / 100 ) == i/100:
            #     print( "Current: "  + str( i ) + " of " + str( self.num_iters ))
            [next_params, next_nmi] = self.do_step(i, prev_params, prev_nmi)
            all_results.append([next_params, next_nmi, i])
            prev_params = next_params
            prev_nmi = next_nmi

        return all_results

    def explore_factor(self, iter, num_iters):
        if iter < 0.1 * num_iters:
            return 0.5
        if iter < 0.3 * num_iters:
            return 0.8
        if iter < 0.5 * num_iters:
            return 1
        if iter < 0.75 * num_iters:
            return 1.5
        if iter < 0.8 * num_iters:
            return 2
        if iter < 0.9 * num_iters:
            return 3
        if iter < 1 * num_iters:
            return 4
        return 5
        # return 0.1

    def std_guess(self, iter, num_iters):
        stds = []
        guesses = self.initial_params
        for guess in guesses:
            num_digits = len(str(round(guess)))
            std = (10 ** (num_digits - 2))
            if iter < 0.5 * num_iters:
                std_factor = 2
            elif iter < 0.65 * num_iters:
                std_factor = 1
            elif iter < 0.85 * num_iters:
                std_factor = 0.75
            elif iter < 0.95 * num_iters:
                std_factor = 0.5
            elif iter < 0.99 * num_iters:
                std_factor = 0.1
            elif iter < num_iters:
                std_factor = 0.01
            # std_factor = 0.1
            stds.append(std * std_factor)
        return stds

    def analyse_results(self, all_results, top_n=5):
        params = [x[0] for x in all_results[round(self.burn * len(all_results)):]]
        nmis = [x[1] for x in all_results[round(self.burn * len(all_results)):]]
        iteration = [x[2] for x in all_results[round(self.burn * len(all_results)):]]
        best_nmis = sorted(nmis, reverse=True)
        best_nmis = best_nmis[:top_n]

        best_params = []
        best_nmi = []
        best_iteration = []

        for p, n, it in zip(params, nmis, iteration):
            if n >= best_nmis[-1]:
                best_params.append(p)
                best_nmi.append(n)
                best_iteration.append(it)
            if len(best_nmi) == top_n:
                break

        return best_params, best_nmi, best_iteration


class DataframeManipulator:

    def __init__(self, df):
        self.df = df

    def look_back(self, column_name, num_rows, new_column_name=None):
        if new_column_name is None:
            new_column_name = column_name + "_T-" + str(num_rows)
        self.df[new_column_name] = self.df[column_name].shift(num_rows)

    def look_forward(self, column_name, num_rows, new_column_name=None):
        if new_column_name is None:
            new_column_name = column_name + "_T+" + str(num_rows)
        self.df[new_column_name] = self.df[column_name].shift(-num_rows)

    def extend_explicit(self, values, new_column_name):
        self.df[new_column_name] = values

    def delete_cols(self, column_names):
        if column_names != []:
            self.df = self.df.drop(column_names, axis=1)

    def make_hl2(self, high, low):
        self.df["HL2"] = (self.df[high] + self.df[low]) / 2

    def extend_with_func(self, func, new_column_name, args=()):
        self.df[new_column_name] = self.df.apply(func, axis=1, args=args)

    def drop_na(self):
        self.df = self.df.dropna().copy()

    def add_lookback_func(self, column_name, lookback_fn, lookback_dur, new_column_name=None, adjust=False):
        df_temp = self.df[column_name]
        if new_column_name is None:
            new_column_name = column_name + "_" + lookback_fn + "_" + str(lookback_dur)
        if lookback_fn == "max":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.max()
        elif lookback_fn == "rma":
            r = df_temp.ewm(min_periods=lookback_dur, adjust=adjust, alpha=1 / lookback_dur)
            self.df[new_column_name] = r.mean()
        elif lookback_fn == "ema":
            r = df_temp.ewm(com=lookback_dur - 1, min_periods=lookback_dur, adjust=adjust)
            self.df[new_column_name] = r.mean()
        elif lookback_fn == "sma":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.mean()
        elif lookback_fn == "max":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.max()
        elif lookback_fn == "min":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.min()
        elif lookback_fn == "percentile":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.apply(lambda x: scipy.stats.percentileofscore(x, x[-1]))
        elif lookback_fn == "std":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.std()
        elif lookback_fn == "sum":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.sum()

    def reverse_column(self, column_name, new_column_name):
        df_temp = self.df[column_name]
        df_temp = df_temp.iloc[::-1].values
        if new_column_name is None:
            self.df[column_name] = df_temp
        else:
            self.df[new_column_name] = df_temp

    def find_filter(self, column_name, filter_mask):
        df_temp = self.df[filter_mask]
        return df_temp[column_name]


class Misc:
    BLANK = "<BLANK>"

    def __init__(self):
        pass

    @staticmethod
    def apply_if_not_present(df, cls, to_delete):
        try:
            idx = df.columns.get_loc(cls.describe())
        except:
            print("Could not find " + str(cls.describe()))
            df = cls.apply(df)
            to_delete.append(cls.describe())
        return [df, to_delete]

    @staticmethod
    def roc_pct(row, horizon, feature):
        change = row[feature] - row[feature + "_T-" + str(horizon)]
        change_pct = change / row[feature + "_T-" + str(horizon)]
        return change_pct

    @staticmethod
    def change(row, horizon, feature):
        chg = row[feature] - row[feature + "_T-" + str(horizon)]
        return chg

    @staticmethod
    def rsi(row, rma_adv, rma_dec, sum_n_adv, sum_n_dec):
        sum_n_adv_v = abs(row[sum_n_adv])
        sum_n_dec_v = abs(row[sum_n_dec])

        rma_adv_v = abs(row[rma_adv])
        rma_dec_v = abs(row[rma_dec])

        mean_adv_v = rma_adv_v
        mean_dec_v = rma_dec_v

        if mean_dec_v == 0:
            ratio = 0
        else:
            ratio = 100 / (1 + (mean_adv_v / mean_dec_v))

        r = 100 - ratio

        return r


class Indicator:

    def __init__(self, feature):
        self.feature = feature

    def apply(self, df):
        pass

    def describe(self):
        return ("SHELL_INDICATOR")


class ROC(Indicator):

    def __init__(self, period, feature):
        self.period = period
        Indicator.__init__(self, feature)

    def describe(self):
        return ("ROC_" + self.feature + "_" + str(self.period))

    def apply(self, df):
        dfm = DataframeManipulator(df)
        dfm.look_back(self.feature, self.period)
        dfm.extend_with_func(Misc.roc_pct, self.describe(), (self.period, self.feature,))
        return dfm.df


class Metrics():

    def __init__(self, feature):
        self.feature = feature
        pass

    def describe(self):
        return ("Empty")

    def apply(self, df):
        return None


class RollingSharpe(Metrics):
    def __init__(self, lookback, feature):
        self.lookback = lookback
        Metrics.__init__(self, feature)

    def describe(self):
        return ("RSharpe_" + str(self.lookback) + "_" + self.feature)

    @staticmethod
    def __make_sharpe(row, roc_col, std_col):
        roc = row[roc_col]
        std = row[std_col]
        return roc / std * math.sqrt(252)

    def apply(self, df):
        roc = ROC(self.lookback, self.feature)
        df = roc.apply(df)
        df[self.describe()] = df[roc.describe()]
        dfm = DataframeManipulator(df)
        dfm.add_lookback_func(self.feature, "std", self.lookback)
        dfm.extend_with_func(RollingSharpe.__make_sharpe, self.describe(),
                             (roc.describe(), self.feature + "_std_" + str(self.lookback),))
        dfm.delete_cols([roc.describe(), self.feature + "_std_" + str(self.lookback)])
        return dfm.df


class RollingFSharpe(Metrics):
    def __init__(self, lookfwd, feature):
        self.lookfwd = lookfwd
        Metrics.__init__(self, feature)

    def describe(self):
        return ("RFSharpe_" + str(self.lookfwd) + "_" + self.feature)

    def apply(self, df):
        delete_cols = []
        sharpe = RollingSharpe(self.lookfwd, self.feature)
        try:
            idx = df.columns.get_loc(sharpe.describe())
        except:
            df = sharpe.apply(df)
            delete_cols.append(sharpe.describe())
        dfm = DataframeManipulator(df)
        dfm.look_forward(sharpe.describe(), self.lookfwd - 1, self.describe())
        if delete_cols != []:
            dfm.delete_cols(delete_cols)
        return dfm.df


class RollingMax(Metrics):
    def __init__(self, lookback, feature):
        self.lookback = lookback
        Metrics.__init__(self, feature)

    def describe(self):
        return ("RMX_" + str(self.lookback) + "_" + self.feature)

    @staticmethod
    def __Max(row, du_col, feature):
        return (max(row[feature], row[du_col]))

    def apply(self, df):
        dfm = DataframeManipulator(df)
        dfm.add_lookback_func(self.feature, "max", self.lookback)
        dfm.extend_with_func(RollingMax.__Max, self.describe(),
                             (self.feature + "_max_" + str(self.lookback), self.feature))
        dfm.delete_cols([self.feature + "_max_" + str(self.lookback)])
        return dfm.df


class RollingFMax(Metrics):
    def __init__(self, lookfwd, feature):
        self.lookfwd = lookfwd
        Metrics.__init__(self, feature)

    def describe(self):
        return ("RFMX_" + str(self.lookfwd) + "_" + self.feature)

    def apply(self, df):
        to_delete = []
        rmx = RollingMax(self.lookfwd, self.feature)
        [df, to_delete] = Misc.apply_if_not_present(df, rmx, to_delete)
        dfm = DataframeManipulator(df)
        dfm.look_forward(rmx.describe(), self.lookfwd - 1, self.describe())
        dfm.delete_cols(to_delete)
        return dfm.df


class RollingMin(Metrics):
    def __init__(self, lookback, feature):
        self.lookback = lookback
        Metrics.__init__(self, feature)

    def describe(self):
        return ("RMN_" + str(self.lookback) + "_" + self.feature)

    @staticmethod
    def __Min(row, dd_col, feature):
        return (min(row[feature], row[dd_col]))

    def apply(self, df):
        dfm = DataframeManipulator(df)
        dfm.add_lookback_func(self.feature, "min", self.lookback)
        dfm.extend_with_func(RollingMin.__Min, self.describe(),
                             (self.feature + "_min_" + str(self.lookback), self.feature))
        dfm.delete_cols([self.feature + "_min_" + str(self.lookback)])
        return dfm.df


class RollingFMin(Metrics):
    def __init__(self, lookfwd, feature):
        self.lookfwd = lookfwd
        Metrics.__init__(self, feature)

    def describe(self):
        return ("RFMN_" + str(self.lookfwd) + "_" + self.feature)

    def apply(self, df):
        to_delete = []
        rmn = RollingMin(self.lookfwd, self.feature)
        [df, to_delete] = Misc.apply_if_not_present(df, rmn, to_delete)

        dfm = DataframeManipulator(df)
        dfm.look_forward(rmn.describe(), self.lookfwd - 1, self.describe())
        dfm.delete_cols(to_delete)
        return dfm.df


class RollingFDD(Metrics):
    def __init__(self, lookback, feature):
        self.lookback = lookback
        Metrics.__init__(self, feature)

    def describe(self):
        return ("RFDD_" + str(self.lookback) + "_" + self.feature)

    @staticmethod
    def __DD(row, dd_col, feature):
        return (row[dd_col] - row[feature]) / row[feature]

    def apply(self, df):
        to_delete = []
        fmn = RollingFMin(self.lookback, self.feature)
        [df, to_delete] = Misc.apply_if_not_present(df, fmn, to_delete)
        dfm = DataframeManipulator(df)
        dfm.extend_with_func(RollingFDD.__DD, self.describe(), (fmn.describe(), self.feature))

        dfm.delete_cols(to_delete)
        return dfm.df


class RollingFDU(Metrics):
    def __init__(self, lookback, feature):
        self.lookback = lookback
        Metrics.__init__(self, feature)

    def describe(self):
        return ("RFDU_" + str(self.lookback) + "_" + self.feature)

    @staticmethod
    def __DU(row, du_col, feature):
        return (row[du_col] - row[feature]) / row[feature]

    def apply(self, df):
        to_delete = []
        fmx = RollingFMax(self.lookback, self.feature)
        [df, to_delete] = Misc.apply_if_not_present(df, fmx, to_delete)

        dfm = DataframeManipulator(df)
        dfm.extend_with_func(RollingFDU.__DU, self.describe(), (fmx.describe(), self.feature))
        dfm.delete_cols(to_delete)
        return dfm.df


class RollingReturn(Metrics):
    def __init__(self, lookback, feature):
        self.lookback = lookback
        Metrics.__init__(self, feature)

    def describe(self):
        return ("RRet_" + str(self.lookback) + "_" + self.feature)

    def apply(self, df):
        roc = ROC(self.lookback, self.feature)
        df = roc.apply(df)
        df[self.describe()] = df[roc.describe()].copy()
        dfm = DataframeManipulator(df)
        dfm.delete_cols([roc.describe()])
        return dfm.df


class RollingFReturn(Metrics):
    def __init__(self, lookfwd, feature):
        self.lookfwd = lookfwd
        Metrics.__init__(self, feature)

    def describe(self):
        return ("RFRet_" + str(self.lookfwd) + "_" + self.feature)

    def apply(self, df):
        to_delete = []
        rr = RollingReturn(self.lookfwd, self.feature)
        [df, to_delete] = Misc.apply_if_not_present(df, rr, to_delete)

        dfm = DataframeManipulator(df)
        dfm.look_forward(rr.describe(), self.lookfwd - 1, self.describe())
        dfm.delete_cols(to_delete)
        df = dfm.df
        return df


class RollingFRR(Metrics):
    def __init__(self, lookfwd, feature):
        self.lookfwd = lookfwd
        Metrics.__init__(self, feature)

    @staticmethod
    def __do_rr(row, du_col, dd_col):
        if row[dd_col] == 0:
            return 100000
        else:
            return abs(row[du_col] / row[dd_col])

    def describe(self):
        return ("RFRR_" + str(self.lookfwd) + "_" + self.feature)

    def apply(self, df):
        to_delete = []
        du = RollingFDU(self.lookfwd, self.feature)
        dd = RollingFDD(self.lookfwd, self.feature)

        [df, to_delete] = Misc.apply_if_not_present(df, du, to_delete)
        [df, to_delete] = Misc.apply_if_not_present(df, dd, to_delete)

        dfm = DataframeManipulator(df)
        dfm.extend_with_func(RollingFRR.__do_rr, self.describe(), (du.describe(), dd.describe(),))

        dfm.delete_cols(to_delete)
        df = dfm.df

        return df


class MCMC_Indicator(MCMC):

    def __init__(self, indicator, initial_args, feature, target_col, df, num_iters, prior, fltr):
        self.target_col = target_col
        self.filter = fltr
        self.indicator = indicator
        self.feature = feature
        self.df = df
        MCMC.__init__(self, alpha_fn=self.create_alpha_fn(), alpha_fn_params_0=self.create_alpha_args(initial_args),
                      target=self.create_target(),
                      num_iters=num_iters, prior=prior)

    def transition_fn(self, cur, iter):
        std = self.std_guess(iter, self.num_iters)
        return [round(x) for x in np.random.normal(cur, std, (len(cur),))]

    def create_alpha_fn(self):
        indicator = self.indicator

        def alpha_fn(*args_to_optimize):
            feature = self.feature
            df = self.df
            ind_args = list(args_to_optimize)
            print(ind_args)
            ind_args.append(feature)
            print("Indicator initialization args")
            print(ind_args)
            id = indicator(*ind_args)
            print("Indicator application args")
            modified_df = id.apply(df)

            modified_df = modified_df.drop([self.target_col], axis=1)
            modified_df = pd.concat([modified_df, self.df[self.target_col]], axis=1, join="inner")
            modified_df = self.filter(modified_df, id.describe(), self.target_col)
            modified_df = modified_df.dropna()

            self.target = modified_df[self.target_col].values
            return modified_df[id.describe()].values

        return alpha_fn

    def create_target(self):
        target = self.df[self.target_col]
        print(target.tail(10))
        return target

    def create_alpha_args(self, args):
        all_args = args

        print("Alpha args")
        print(all_args)
        return all_args


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))


def valid_dates(dates_all):
    dates = []
    i = 0
    while True:
        dates.append(dates_all[i])
        if dates_all[i] > pd.to_datetime(date.today()):
            break
        i = i + 1
    return dates


if __name__ == '__main__':

    ticker = ticker_inp
    number_of_optimization_periods = number_of_optimization_periods_inp
    recalib_months = recalib_months_inp
    num_strategies = num_strategies_inp
    metric = metric_inp
    ticker_yf = ticker_yf_inp

    temp_og, today_time_close = get_data_alpha_investpy_yf(ticker, ticker_yf)

    dates_all_ss = pd.date_range(start=str(temp_og.iloc[1]['Date'] + timedelta(days=365))[:10], end="2024-06-15",
                                 freq=f'3M')

    dates_ss = valid_dates(dates_all_ss)

    if (pd.to_datetime(date.today()) in dates_all_ss):
        print(f"Recalibrating {ticker} at {datetime.now()}")

        if number_of_optimization_periods == 1:
            print(f"Number of optimization periods: {number_of_optimization_periods}")
            res_test2 = select_all_strategies(24, dates_ss, temp_og, ticker, save=True)
            res_test4 = 0
            res_test8 = 0
            print("Finished selecting all strategies")
        if number_of_optimization_periods == 2:
            print(f"Number of optimization periods: {number_of_optimization_periods}")
            res_test2 = select_all_strategies(24, dates_ss, temp_og, ticker, save=True)
            res_test4 = select_all_strategies(48, dates_ss, temp_og, ticker, save=True)
            res_test8 = 0
            print("Finished selecting all strategies")
        if number_of_optimization_periods == 3:
            print(f"Number of optimization periods: {number_of_optimization_periods}")
            res_test2 = select_all_strategies(24, dates_ss, temp_og, ticker, save=True)
            res_test4 = select_all_strategies(48, dates_ss, temp_og, ticker, save=True)
            res_test8 = select_all_strategies(96, dates_ss, temp_og, ticker, save=True)
            print("Finished selecting all strategies")

        ss_test_imp, res_test_imp = select_strategies_from_corr_filter(res_test2, res_test4, res_test8, dates_ss, temp_og,
                                                                       number_of_optimization_periods, 10, ticker,
                                                                       save=True)

        res_test = res_test_imp
        ss_test = ss_test_imp
        dates = []
        for date_i in range(len(dates_ss) - (int(24 / 3) + 1)):
            if (3 * date_i) % recalib_months == 0:
                dates.append(dates_ss[date_i + int(24 / 3)])

        print(f"Recalibrating Weights: {datetime.now()}")
        inputs = []
        for date_i in range(len(dates) - 1):
            inputs.append([date_i, dates, temp_og, ss_test, res_test, num_strategies, metric, recalib_months, dates_ss])

        weights = [optimize_weights_live(inputs[-1])]

        if not os.path.exists(f'{ticker}/weights'):
            os.makedirs(f'{ticker}/weights')

        with open(
                f'{ticker}/weights/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.pkl',
                'wb') as file:
            pickle.dump(weights, file)

        zipf = zipfile.ZipFile(f'{ticker}.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir(f"{ticker}/", zipf)
        zipf.close()