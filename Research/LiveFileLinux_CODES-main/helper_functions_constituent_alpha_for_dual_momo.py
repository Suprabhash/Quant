import pandas as pd
import numpy as np

def return_all_tickers_over_backtest(assets):
    alpha_tickers = []

    for asset in assets:
        alpha_tickers = alpha_tickers + list(asset["Ticker"])

    return list(set(alpha_tickers))

def backtest_Alpha_AM_Nifty_ConstituentAlphas(dates_rebalancing, data_inp, assetsb):
    current_balance = 6460589.485738462
    gold_allocation = 0
    nifty_allocation = 0
    cash_allocation = 0
    portfolio_value = pd.DataFrame()

    # data_inp["signal_nifty"] = data_inp["signal_nifty"].shift(1)
    # data_inp["signal_gold"] = data_inp["signal_gold"].shift(1)
    data_inp["signal_nifty"].iloc[0] = 1
    data_inp["signal_gold"].iloc[0] = 1

    for date_i in range(len(dates_rebalancing) - 1):
        try:
            test = data_inp.loc[
                (data_inp["Date"] >= str(dates_rebalancing[date_i])) & (
                        data_inp["Date"] < str(dates_rebalancing[date_i + 1]))].reset_index().drop(['index'], axis=1)
            test.set_index(test["Date"], inplace=True)
        except:
            test = data_inp.loc[(data_inp["Date"] > str(dates_rebalancing[date_i]))]
            test.set_index(test["Date"], inplace=True)
        tickers = assetsb[date_i]["Ticker"].to_list()

        for ticker in tickers:
            test[f"signal{ticker}"].fillna(0, inplace=True)
            test[f"AlphaFfillPrice{ticker}"] = test[f"signal{ticker}"].shift(1)
            test[f"AlphaFfillPrice{ticker}"].iloc[0] = 1
            for k in range(len(test[f"AlphaFfillPrice{ticker}"])):
                if ((test["signal_nifty"].iloc[k] == 1) & (test["signal_nifty"].shift(1).fillna(0).iloc[k] == 0)):
                    test[f"AlphaFfillPrice{ticker}"].iloc[k] = 1
            test[f"AlphaFfillPrice{ticker}"] = test[ticker] * test[f"AlphaFfillPrice{ticker}"].replace(0, np.nan)
            test[f"AlphaFfillPrice{ticker}"].fillna(method="ffill", inplace=True)

        percent_tracker_current_balance_ticker = {}
        percent_tracker_units_ticker = {}
        percent_ticker = {}
        for ticker in tickers:
            percent_tracker_current_balance_ticker[ticker] = current_balance / len(tickers)
            percent_tracker_units_ticker[ticker] = percent_tracker_current_balance_ticker[ticker] / test.iloc[0][ticker]

        current_balance_ticker = {}
        units_ticker = {}
        for ticker in tickers:
            current_balance_ticker[ticker] = current_balance / len(tickers)
            units_ticker[ticker] = current_balance_ticker[ticker] / test.iloc[0][ticker]

        units_gold = 0

        for i in range(len(test)):
            num_act_tickers = 0
            unallocated = 0
            for ticker in tickers:
                if (test[f"signal{ticker}"].iloc[i] == 1):
                    num_act_tickers = num_act_tickers + 1
                percent_tracker_current_balance_ticker[ticker] = percent_tracker_units_ticker[ticker] * test.iloc[i][ticker]
            for ticker in tickers:
                percent_ticker[ticker] = percent_tracker_current_balance_ticker[ticker] / sum(percent_tracker_current_balance_ticker.values())

            signal_nifty = test["signal_nifty"].iloc[i]
            signal_gold = test["signal_gold"].iloc[i]

            if signal_nifty == 1:
                nifty_allocation = current_balance
                gold_allocation = 0
                cash_allocation = 0
            if (signal_nifty == 0) & (signal_gold == 1):
                nifty_allocation = 0
                gold_allocation = current_balance / 2
                cash_allocation = current_balance / 2
            if (signal_nifty == 0) & (signal_gold == 0):
                nifty_allocation = 0
                gold_allocation = 0
                cash_allocation = current_balance

            if ((test["signal_nifty"].iloc[i] == 1) & (test["signal_nifty"].shift(1).fillna(0).iloc[i] == 0)):
                for ticker in tickers:
                    current_balance_ticker[ticker] = nifty_allocation * percent_ticker[ticker]
                    units_ticker[ticker] = current_balance_ticker[ticker] / test.iloc[i][ticker]

            if ((test["signal_nifty"].iloc[i] == 0) & (test["signal_nifty"].shift(1).fillna(1).iloc[i] == 1)):
                for ticker in tickers:
                    current_balance_ticker[ticker] = 0
                    units_ticker[ticker] = 0
                if signal_gold == 1:
                    units_gold = gold_allocation / test.iloc[i]["Close_gold"]

            if ((test["signal_gold"].iloc[i] == 1) & (test["signal_gold"].shift(1).fillna(0).iloc[i] == 0)):
                units_gold = gold_allocation / test.iloc[i]["Close_gold"]

            if ((test["signal_gold"].iloc[i] == 0) & (test["signal_gold"].shift(1).fillna(1).iloc[i] == 1)):
                units_gold = 0

            if signal_nifty == 1:
                units_gold = 0
                nifty_allocation = 0
                for ticker in tickers:
                    if (test[f"signal{ticker}"].iloc[i] == 1):
                        current_balance_ticker[ticker] = units_ticker[ticker] * test.iloc[i][ticker]
                    else:
                        current_balance_ticker[ticker] = 0
                        unallocated = unallocated + units_ticker[ticker] * test.iloc[i][f"AlphaFfillPrice{ticker}"]
                    nifty_allocation = nifty_allocation + current_balance_ticker[ticker]

            if (signal_nifty == 0) & (signal_gold == 1):
                gold_allocation = units_gold * test.iloc[i]["Close_gold"]

            unallocated = unallocated * (1 + 6 / 25200)
            cash_allocation = cash_allocation * (1 + 6 / 25200)
            current_balance = nifty_allocation + gold_allocation + cash_allocation + unallocated
            portfolio_day = {'Date': test.iloc[i]["Date"], 'signal_nifty': signal_nifty, 'signal_gold': signal_gold,
                             'units_gold': units_gold, 'nifty_allocation': nifty_allocation,
                             'gold_allocation': gold_allocation, 'cash_allocation': cash_allocation,
                             'Pvalue': current_balance, 'unallocated': unallocated, "Number of Active Tickers": num_act_tickers}
            portfolio_day[f"Gold_close"] = test.iloc[i]["Close_gold"]
            for ticker in tickers:
                portfolio_day[f"{ticker}_close"] = test.iloc[i][ticker]
                portfolio_day[f"{ticker}_signal"] = test.iloc[i][f"signal{ticker}"]
                portfolio_day[f"{ticker}_percent"] = percent_ticker[ticker]
                portfolio_day[f"{ticker}_units"] = units_ticker[ticker]
                portfolio_day[f"{ticker}_current_balance"] = current_balance_ticker[ticker]
            portfolio_day = pd.DataFrame([portfolio_day])
            portfolio_day = portfolio_day.set_index("Date")
            portfolio_value = pd.concat([portfolio_value, portfolio_day], axis=0, join="outer")

    # print(portfolio_value)
    # returns = pd.DataFrame(np.log(portfolio_value / portfolio_value.shift(1))).rename(columns={"Pvalue": "Return"})
    return portfolio_value, units_ticker, units_gold

# def backtest_Alpha_AM_Nifty_ConstituentAlphas(dates_rebalancing, data_inp, assetsb):
#     current_balance = 6460589.485738462
#     gold_allocation = 0
#     nifty_allocation = 0
#     cash_allocation = 0
#     portfolio_value = pd.DataFrame()
#
#     data_inp["signal_nifty"] = data_inp["signal_nifty"].shift(1)
#     data_inp["signal_gold"] = data_inp["signal_gold"].shift(1)
#     data_inp["signal_nifty"].iloc[0] = 1
#     data_inp["signal_gold"].iloc[0] = 1
#
#     for date_i in range(len(dates_rebalancing) - 1):
#         try:
#             test = data_inp.loc[
#                 (data_inp["Date"] >= str(dates_rebalancing[date_i])) & (
#                         data_inp["Date"] < str(dates_rebalancing[date_i + 1]))].reset_index().drop(['index'], axis=1)
#             test.set_index(test["Date"], inplace=True)
#         except:
#             test = data_inp.loc[(data_inp["Date"] > str(dates_rebalancing[date_i]))]
#             test.set_index(test["Date"], inplace=True)
#         tickers = assetsb[date_i]["Ticker"].to_list()
#
#         for ticker in tickers:
#             test[f"signal{ticker}"].fillna(0, inplace=True)
#             test[f"AlphaFfillPrice{ticker}"] = test[f"signal{ticker}"].shift(1)
#             test[f"AlphaFfillPrice{ticker}"].iloc[0] = 1
#             for k in range(len(test[f"AlphaFfillPrice{ticker}"])):
#                 if ((test["signal_nifty"].iloc[k] == 1) & (test["signal_nifty"].shift(1).fillna(0).iloc[k] == 0)):
#                     test[f"AlphaFfillPrice{ticker}"].iloc[k] = 1
#             test[f"AlphaFfillPrice{ticker}"] = test[ticker] * test[f"AlphaFfillPrice{ticker}"].replace(0, np.nan)
#             test[f"AlphaFfillPrice{ticker}"].fillna(method="ffill", inplace=True)
#
#         percent_tracker_current_balance_ticker = {}
#         percent_tracker_units_ticker = {}
#         percent_ticker = {}
#         for ticker in tickers:
#             percent_tracker_current_balance_ticker[ticker] = current_balance / len(tickers)
#             percent_tracker_units_ticker[ticker] = percent_tracker_current_balance_ticker[ticker] / test.iloc[0][ticker]
#
#         current_balance_ticker = {}
#         units_ticker = {}
#         for ticker in tickers:
#             current_balance_ticker[ticker] = current_balance / len(tickers)
#             units_ticker[ticker] = current_balance_ticker[ticker] / test.iloc[0][ticker]
#
#         units_gold = 0
#
#         for i in range(len(test)):
#             num_act_tickers = 0
#             unallocated = 0
#             for ticker in tickers:
#                 if (test[f"signal{ticker}"].iloc[i] == 1):
#                     num_act_tickers = num_act_tickers + 1
#                 percent_tracker_current_balance_ticker[ticker] = percent_tracker_units_ticker[ticker] * test.iloc[i][ticker]
#             for ticker in tickers:
#                 percent_ticker[ticker] = percent_tracker_current_balance_ticker[ticker] / sum(percent_tracker_current_balance_ticker.values())
#
#             signal_nifty = test["signal_nifty"].iloc[i]
#             signal_gold = test["signal_gold"].iloc[i]
#
#             if signal_nifty == 1:
#                 nifty_allocation = current_balance
#                 gold_allocation = 0
#                 cash_allocation = 0
#             if (signal_nifty == 0) & (signal_gold == 1):
#                 nifty_allocation = 0
#                 gold_allocation = current_balance / 2
#                 cash_allocation = current_balance / 2
#             if (signal_nifty == 0) & (signal_gold == 0):
#                 nifty_allocation = 0
#                 gold_allocation = 0
#                 cash_allocation = current_balance
#
#             if ((test["signal_nifty"].iloc[i] == 1) & (test["signal_nifty"].shift(1).fillna(0).iloc[i] == 0)):
#                 for ticker in tickers:
#                     current_balance_ticker[ticker] = nifty_allocation * percent_ticker[ticker]
#                     units_ticker[ticker] = current_balance_ticker[ticker] / test.iloc[i][ticker]
#
#             if ((test["signal_nifty"].iloc[i] == 0) & (test["signal_nifty"].shift(1).fillna(1).iloc[i] == 1)):
#                 for ticker in tickers:
#                     current_balance_ticker[ticker] = 0
#                     units_ticker[ticker] = 0
#                 if signal_gold == 1:
#                     units_gold = gold_allocation / test.iloc[i]["Close_gold"]
#
#             if ((test["signal_gold"].iloc[i] == 1) & (test["signal_gold"].shift(1).fillna(0).iloc[i] == 0)):
#                 units_gold = gold_allocation / test.iloc[i]["Close_gold"]
#
#             if ((test["signal_gold"].iloc[i] == 0) & (test["signal_gold"].shift(1).fillna(1).iloc[i] == 1)):
#                 units_gold = 0
#
#             if signal_nifty == 1:
#                 units_gold = 0
#                 nifty_allocation = 0
#                 for ticker in tickers:
#                     if (test[f"signal{ticker}"].iloc[i] == 1):
#                         current_balance_ticker[ticker] = units_ticker[ticker] * test.iloc[i][ticker]
#                     else:
#                         current_balance_ticker[ticker] = 0
#                         unallocated = unallocated + units_ticker[ticker] * test.iloc[i][f"AlphaFfillPrice{ticker}"]
#                     nifty_allocation = nifty_allocation + current_balance_ticker[ticker]
#
#             if (signal_nifty == 0) & (signal_gold == 1):
#                 gold_allocation = units_gold * test.iloc[i]["Close_gold"]
#
#             unallocated = unallocated * (1 + 6 / 25200)
#             cash_allocation = cash_allocation * (1 + 6 / 25200)
#             current_balance = nifty_allocation + gold_allocation + cash_allocation + unallocated
#             portfolio_day = {'Date': test.iloc[i]["Date"], 'signal_nifty': signal_nifty, 'signal_gold': signal_gold,
#                              'units_gold': units_gold, 'nifty_allocation': nifty_allocation,
#                              'gold_allocation': gold_allocation, 'cash_allocation': cash_allocation,
#                              'Pvalue': current_balance, 'unallocated': unallocated, "Number of Active Tickers": num_act_tickers}
#             portfolio_day[f"Gold_close"] = test.iloc[i]["Close_gold"]
#             for ticker in tickers:
#                 portfolio_day[f"{ticker}_close"] = test.iloc[i][ticker]
#                 portfolio_day[f"{ticker}_signal"] = test.iloc[i][f"signal{ticker}"]
#                 portfolio_day[f"{ticker}_percent"] = percent_ticker[ticker]
#                 portfolio_day[f"{ticker}_units"] = units_ticker[ticker]
#                 portfolio_day[f"{ticker}_current_balance"] = current_balance_ticker[ticker]
#             portfolio_day = pd.DataFrame([portfolio_day])
#             portfolio_day = portfolio_day.set_index("Date")
#             portfolio_value = pd.concat([portfolio_value, portfolio_day], axis=0, join="outer")
#
#     # print(portfolio_value)
#     # returns = pd.DataFrame(np.log(portfolio_value / portfolio_value.shift(1))).rename(columns={"Pvalue": "Return"})
#     return portfolio_value, units_ticker, units_gold

def backtest_Alpha_AM_Midcap_ConstituentAlphas(dates_rebalancing, data_inp, assetsb):
    current_balance = 4081757.322757147
    gold_allocation = 0
    midcap_allocation = 0
    cash_allocation = 0
    portfolio_value = pd.DataFrame()

    # data_inp["signal_midcap"] = data_inp["signal_midcap"].shift(1)
    # data_inp["signal_gold"] = data_inp["signal_gold"].shift(1)
    data_inp["signal_midcap"].iloc[0] = 1
    data_inp["signal_gold"].iloc[0] = 1

    for date_i in range(len(dates_rebalancing) - 1):
        try:
            test = data_inp.loc[
                (data_inp["Date"] >= str(dates_rebalancing[date_i])) & (
                        data_inp["Date"] < str(dates_rebalancing[date_i + 1]))].reset_index().drop(['index'], axis=1)
            test.set_index(test["Date"], inplace=True)
        except:
            test = data_inp.loc[(data_inp["Date"] > str(dates_rebalancing[date_i]))]
            test.set_index(test["Date"], inplace=True)
        tickers = assetsb[date_i]["Ticker"].to_list()

        for ticker in tickers:
            test[f"signal{ticker}"].fillna(0, inplace=True)
            test[f"AlphaFfillPrice{ticker}"] = test[f"signal{ticker}"].shift(1)
            test[f"AlphaFfillPrice{ticker}"].iloc[0] = 1
            for k in range(len(test[f"AlphaFfillPrice{ticker}"])):
                if ((test["signal_midcap"].iloc[k] == 1) & (test["signal_midcap"].shift(1).fillna(0).iloc[k] == 0)):
                    test[f"AlphaFfillPrice{ticker}"].iloc[k] = 1
            test[f"AlphaFfillPrice{ticker}"] = test[ticker] * test[f"AlphaFfillPrice{ticker}"].replace(0, np.nan)
            test[f"AlphaFfillPrice{ticker}"].fillna(method="ffill", inplace=True)

        percent_tracker_current_balance_ticker = {}
        percent_tracker_units_ticker = {}
        percent_ticker = {}
        for ticker in tickers:
            percent_tracker_current_balance_ticker[ticker] = current_balance / len(tickers)
            percent_tracker_units_ticker[ticker] = percent_tracker_current_balance_ticker[ticker] / test.iloc[0][ticker]

        current_balance_ticker = {}
        units_ticker = {}
        for ticker in tickers:
            current_balance_ticker[ticker] = current_balance / len(tickers)
            units_ticker[ticker] = current_balance_ticker[ticker] / test.iloc[0][ticker]

        units_gold = 0

        for i in range(len(test)):
            num_act_tickers = 0
            unallocated = 0
            for ticker in tickers:
                if (test[f"signal{ticker}"].iloc[i] == 1):
                    num_act_tickers = num_act_tickers + 1
                percent_tracker_current_balance_ticker[ticker] = percent_tracker_units_ticker[ticker] * test.iloc[i][ticker]
            for ticker in tickers:
                percent_ticker[ticker] = percent_tracker_current_balance_ticker[ticker] / sum(percent_tracker_current_balance_ticker.values())

            signal_midcap = test["signal_midcap"].iloc[i]
            signal_gold = test["signal_gold"].iloc[i]

            if signal_midcap == 1:
                midcap_allocation = current_balance
                gold_allocation = 0
                cash_allocation = 0
            if (signal_midcap == 0) & (signal_gold == 1):
                midcap_allocation = 0
                gold_allocation = current_balance / 2
                cash_allocation = current_balance / 2
            if (signal_midcap == 0) & (signal_gold == 0):
                midcap_allocation = 0
                gold_allocation = 0
                cash_allocation = current_balance

            if ((test["signal_midcap"].iloc[i] == 1) & (test["signal_midcap"].shift(1).fillna(0).iloc[i] == 0)):
                for ticker in tickers:
                    current_balance_ticker[ticker] = midcap_allocation * percent_ticker[ticker]
                    units_ticker[ticker] = current_balance_ticker[ticker] / test.iloc[i][ticker]

            if ((test["signal_midcap"].iloc[i] == 0) & (test["signal_midcap"].shift(1).fillna(1).iloc[i] == 1)):
                for ticker in tickers:
                    current_balance_ticker[ticker] = 0
                    units_ticker[ticker] = 0
                if signal_gold == 1:
                    units_gold = gold_allocation / test.iloc[i]["Close_gold"]

            if ((test["signal_gold"].iloc[i] == 1) & (test["signal_gold"].shift(1).fillna(0).iloc[i] == 0)):
                units_gold = gold_allocation / test.iloc[i]["Close_gold"]

            if ((test["signal_gold"].iloc[i] == 0) & (test["signal_gold"].shift(1).fillna(1).iloc[i] == 1)):
                units_gold = 0

            if signal_midcap == 1:
                units_gold = 0
                midcap_allocation = 0
                for ticker in tickers:
                    if (test[f"signal{ticker}"].iloc[i] == 1):
                        current_balance_ticker[ticker] = units_ticker[ticker] * test.iloc[i][ticker]
                    else:
                        current_balance_ticker[ticker] = 0
                        unallocated = unallocated + units_ticker[ticker] * test.iloc[i][f"AlphaFfillPrice{ticker}"]
                    midcap_allocation = midcap_allocation + current_balance_ticker[ticker]

            if (signal_midcap == 0) & (signal_gold == 1):
                gold_allocation = units_gold * test.iloc[i]["Close_gold"]

            unallocated = unallocated * (1 + 6 / 25200)
            cash_allocation = cash_allocation * (1 + 6 / 25200)
            current_balance = midcap_allocation + gold_allocation + cash_allocation + unallocated
            portfolio_day = {'Date': test.iloc[i]["Date"], 'signal_midcap': signal_midcap, 'signal_gold': signal_gold,
                             'units_gold': units_gold, 'midcap_allocation': midcap_allocation,
                             'gold_allocation': gold_allocation, 'cash_allocation': cash_allocation,
                             'Pvalue': current_balance, 'unallocated': unallocated, "Number of Active Tickers": num_act_tickers}
            portfolio_day[f"Gold_close"] = test.iloc[i]["Close_gold"]
            for ticker in tickers:
                portfolio_day[f"{ticker}_close"] = test.iloc[i][ticker]
                portfolio_day[f"{ticker}_signal"] = test.iloc[i][f"signal{ticker}"]
                portfolio_day[f"{ticker}_percent"] = percent_ticker[ticker]
                portfolio_day[f"{ticker}_units"] = units_ticker[ticker]
                portfolio_day[f"{ticker}_current_balance"] = current_balance_ticker[ticker]
            portfolio_day = pd.DataFrame([portfolio_day])
            portfolio_day = portfolio_day.set_index("Date")
            portfolio_value = pd.concat([portfolio_value, portfolio_day], axis=0, join="outer")

    # print(portfolio_value)
    # returns = pd.DataFrame(np.log(portfolio_value / portfolio_value.shift(1))).rename(columns={"Pvalue": "Return"})
    return portfolio_value, units_ticker, units_gold