import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from CacheUpdater import *
from MCMC.MCMC import MCMC
from helper_functions import valid_dates
from datetime import datetime
import missingno as msno
import matplotlib.pyplot as plt

def prepare_portfolio_data(tickers, recalibrating_months, api, country):
    def get_data(ticker, api, country):
        if api == "yfinance":
            temp_og = yf.download(ticker, start='2007-07-01', end=str(date.today() + timedelta(1)))
            if len(temp_og) == 0:
                temp_og = yf.download(ticker, start='2007-07-01', end=str(date.today()))
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

        if api == "reuters":
            temp_og = ek.get_timeseries(ticker, start_date='2007-07-01', end_date=str(date.today() + timedelta(1)))
            temp_og.reset_index(inplace=True)
            temp_og.rename(
                columns={"HIGH": "High", "CLOSE": "Close", "LOW": "Low", "OPEN": "Open", "VOLUME": "Volume"},
                inplace=True)
            temp_og.drop(['COUNT'], axis=1, inplace=True)

        return temp_og

    data = pd.DataFrame()
    for ticker in tickers:
        try:
            temp_og = get_data(ticker, api, country)
            data = pd.concat([data,
                              temp_og["Close"].to_frame().astype(float).rename(columns={"Close": ticker}).set_index(
                                  temp_og["Date"])], axis=1)
            # data[f"{ticker}Return"] = np.log(data[ticker] / data[ticker].shift(1))
            # data[f"{ticker}ROC0.5"] = data[ticker].pct_change(10)
            # data[f"{ticker}ROC1"] = data[ticker].pct_change(21)
            # data[f"{ticker}ROC3"] = data[ticker].pct_change(63)
            # data[f"{ticker}ROC6"] = data[ticker].pct_change(126)
            # data[f"{ticker}ROC9"] = data[ticker].pct_change(189)
            # data[f"{ticker}ROC12"] = data[ticker].pct_change(252)
            # data[f"{ticker}SD12"] = data[ticker].pct_change().rolling(252).std()
            # data[f"{ticker}FReturn"] = data[ticker].shift(-recalibrating_months * 21) / data[ticker] - 1
        except:
            print(f"{ticker} not processed")
    data.reset_index(inplace=True)
    return data

def get_weights_stocks_live(constituents, topn, test_monthsf, train_monthsf, datesf, temp_ogf, save=True):
    inputs = []
    for date_i in range(len(datesf) - (int(train_monthsf / test_monthsf) + 1)):
        inputs.append([temp_ogf, topn, datesf, date_i, train_monthsf, test_monthsf, constituents])

    results = recalibrate_weights_stocks(inputs[-1])
    # res_test2 = [pd.DataFrame(columns=["Lookback", "Low Bound", "High Bound", "AvgWinLoss",  "Sortino", "Optimization_Years"])] * len(results2)
    res_test = [pd.DataFrame(
        columns=["Ticker", "WtROC0.5", "WtROC1", "WtROC3", "WtROC6", "WtROC9", "WtROC12", "Accelerating Momentum"])]
    res_test[0] = pd.concat(
        [res_test[0], results[1].reset_index().drop(['index'], axis=1)], axis=0)

    return res_test


def recalibrate_weights_stocks(inp):

    def alpha(*args):
        weights = pd.DataFrame(args)
        weights = weights / weights.sum()
        for ticker in tickers:
            df = data[[f"{ticker}ROC0.5", f"{ticker}ROC1", f"{ticker}ROC3", f"{ticker}ROC6", f"{ticker}ROC9",
                       f"{ticker}ROC12", f"{ticker}SD12", f"{ticker}FReturn"]]
            df[f"{ticker}AM"] = np.dot(df[[f"{ticker}ROC0.5", f"{ticker}ROC1", f"{ticker}ROC3", f"{ticker}ROC6",
                                           f"{ticker}ROC9", f"{ticker}ROC12"]], weights)
            data[f"{ticker}AM"] = df[f"{ticker}AM"] / df[f"{ticker}SD12"]
        return data[[f"{ticker}AM" for ticker in tickers]].to_numpy().ravel()

    def prior(params):
        if (params[0] < 0) | (params[0] > 1):
            return 0
        if (params[1] < 0) | (params[1] > 1):
            return 0
        if (params[2] < 0) | (params[2] > 1):
            return 0
        if (params[3] < 0) | (params[3] > 1):
            return 0
        if (params[4] < 0) | (params[4] > 1):
            return 0
        if (params[5] < 0) | (params[5] > 1):
            return 0
        return 1

    # Optimizing weights for entire portfolio
    temp_ogf = inp[0]
    top_n = inp[1]
    datesf = inp[2]
    date_i = inp[3]
    train_monthsf = inp[4]
    test_monthsf = inp[5]
    constituents = inp[6]

    # data_og = temp_ogf.loc[(temp_ogf["Date"] > str(datesf[date_i])) & (temp_ogf["Date"] < str(datesf[date_i + int(train_monthsf/test_monthsf) + int(test_monthsf/test_monthsf)]))].reset_index().drop(['index'], axis=1)

    # Adjustment made for forward returns
    data_og = temp_ogf.loc[(temp_ogf["Date"] > str(datesf[date_i])) & (
            temp_ogf["Date"] < str(datesf[date_i + int(train_monthsf / test_monthsf) - 1]))].reset_index().drop(
        ['index'], axis=1)

    tickers_in_index = constituents.loc[datesf[date_i]][0]
    data = data_og.dropna(axis=1, how='any')

    tickers1 = []
    for column in data.columns[1:]:
        if column.endswith("Return") | column.endswith("FReturn") | column.endswith("ROC0.5") | column.endswith(
                "ROC1") | column.endswith("ROC3") | \
                column.endswith("ROC6") | column.endswith("ROC9") | column.endswith("ROC12") | column.endswith(
            "SD12"):
            continue
        else:
            tickers1.append(column)

    tickers = []
    for ticker in tickers1:
        if ((f"{ticker}" in data.columns[1:]) & (f"{ticker}ROC0.5" in data.columns[1:]) & (
                f"{ticker}ROC1" in data.columns[1:]) & (
                f"{ticker}ROC3" in data.columns[1:]) & (f"{ticker}ROC6" in data.columns[1:]) &
                (f"{ticker}ROC9" in data.columns[1:]) & (f"{ticker}ROC12" in data.columns[1:]) & (
                        f"{ticker}SD12" in data.columns[1:]) & (f"{ticker}FReturn" in data.columns[1:])):
            # print(ticker)
            tickers.append(ticker)


    for ticker in tickers:
        if not ticker in tickers_in_index:
            tickers.remove(ticker)

    random_starts = 1
    iterations = 10
    guess_list = [np.random.dirichlet(np.ones(6), size=1).tolist()[0] for i in range(random_starts)]
    res = pd.DataFrame(columns=["WtROC0.5", "WtROC1", "WtROC3", "WtROC6", "WtROC9", "WtROC12", "NMIS"])

    try:
        for guess in guess_list:
            mc = MCMC(alpha_fn=alpha, alpha_fn_params_0=guess,
                      target=data[[f"{ticker}FReturn" for ticker in tickers]].to_numpy().ravel(),
                      num_iters=iterations,
                      prior=prior, optimize_fn=None, lower_limit=0, upper_limit=1)
            rs = mc.optimize()
            res_iter = [{"WtROC0.5": mc.analyse_results(rs, top_n=iterations)[0][i][0],
                         "WtROC1": mc.analyse_results(rs, top_n=iterations)[0][i][1],
                         "WtROC3": mc.analyse_results(rs, top_n=iterations)[0][i][2],
                         "WtROC6": mc.analyse_results(rs, top_n=iterations)[0][i][3],
                         "WtROC9": mc.analyse_results(rs, top_n=iterations)[0][i][4],
                         "WtROC12": mc.analyse_results(rs, top_n=iterations)[0][i][5],
                         "NMIS": mc.analyse_results(rs, top_n=iterations)[1][i]} \
                        for i in range(iterations)]
            res_iter = pd.DataFrame(res_iter)
            res = pd.concat([res, res_iter], axis=0)
        res = res.sort_values("NMIS", axis=0, ascending=False).reset_index(drop=True)
        chosen_weights = pd.DataFrame(
            [res.iloc[0]["WtROC0.5"], res.iloc[0]["WtROC1"], res.iloc[0]["WtROC3"], res.iloc[0]["WtROC6"],
             res.iloc[0]["WtROC9"], res.iloc[0]["WtROC12"]])
        chosen_weights = chosen_weights / chosen_weights.sum()
        am = []
        for ticker in tickers:
            am.append({"Ticker": ticker, "WtROC0.5": float(chosen_weights.iloc[0]),
                       "WtROC1": float(chosen_weights.iloc[1]),
                       "WtROC3": float(chosen_weights.iloc[2]), "WtROC6": float(chosen_weights.iloc[3]),
                       "WtROC9": float(chosen_weights.iloc[4]), "WtROC12": float(chosen_weights.iloc[5]),
                       "Accelerating Momentum": np.dot(
                           data_og.iloc[-1][[f"{ticker}ROC0.5", f"{ticker}ROC1", f"{ticker}ROC3",
                                             f"{ticker}ROC6", f"{ticker}ROC9", f"{ticker}ROC12"]],
                           chosen_weights)[0] / data_og.iloc[-1][f"{ticker}SD12"]})
        am = pd.DataFrame(am)
        am = am.sort_values("Accelerating Momentum", axis=0, ascending=False).reset_index(drop=True)
        am = am.iloc[:top_n]
    except:
        am = pd.DataFrame(columns=["Ticker", "WtROC0.5", "WtROC1", "WtROC3", "WtROC6", "WtROC9", "WtROC12",
                                   "Accelerating Momentum"])
    return date_i, am

if __name__ == '__main__':

    #Step1
    # NIFTY AM and Constituent rebalance
    # #Recalibrating AM Constituents
    print(f"Start Recalibrating constituents of Midcap at {datetime.now()}")
    index = ".NIMDCP50"
    with open(f'MIDCAP_CSV_Constituents.pkl', 'rb') as file:
        constituents = pickle.load(file)
    # print("Hi")
    dates = constituents.index
    dates_new = valid_dates(pd.date_range(start=dates[0],
                                 end="2050-06-15", freq=f'1M'))

    constituents = constituents.reindex(dates_new)
    constituents.fillna(method="ffill", inplace=True)
    print(f"End Recalibrating constituents of Midcap at {datetime.now()}")

    with open(f'MIDCAP_CSV_Constituents.pkl', 'wb') as file:
        pickle.dump(constituents, file)

    #Step2
    # #Recalibrating AM
    index = ".NIMDCP50"
    with open(f'MIDCAP_CSV_Constituents.pkl', 'rb') as file:
        constituents = pickle.load(file)
    constituents_all = []
    for i in range(len(constituents)):
        constituents_all = constituents_all + constituents.iloc[i][0]
    tickers = list(set(constituents_all))
    top_nassets = 8
    recalibrating_months = 1
    training_period = 24  # 24/48/96
    dates_recalibrating = valid_dates(
        pd.date_range(start="2007-01-01", end="2024-06-15", freq=f'{recalibrating_months}M'))

    data_inp = prepare_portfolio_data(tickers, recalibrating_months, "investpy", "india")

    data_inp = data_inp[(data_inp["Date"] < "2014-02-28") | (data_inp["Date"] > "2014-03-13")]
    data_inp = data_inp[data_inp['Date'] != '2020-10-16']
    data_inp = data_inp[data_inp["Date"] != "2020-11-14"]
    data_inp = data_inp[data_inp["Date"] != "2020-11-17"]

    # with open(f'MIDCAP_RecalibPeriod_{int(1)}.pkl', 'rb') as file:
    #     assets = pickle.load(file)

    print(f"Start Recalibrating AM of Nifty Midcap at {datetime.now()}")
    latest = get_weights_stocks_live(constituents, top_nassets, recalibrating_months, training_period, dates_recalibrating,data_inp, save=False)[0]

    with open(f'Z:/Algo/Live_emailer_updates/files_for_rebalance/MIDCAP_RecalibPeriod_latest.pkl', 'wb') as file:
        pickle.dump(latest, file)

    print(f"End Recalibrating AM of Nifty Midcap at {datetime.now()}")
