import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from datetime import date,timedelta
import pickle
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import itertools
import multiprocessing
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import numpy as np
import math

class vol_mom_strategy:
    metrics = {'Sharpe': 'Sharpe Ratio',
               'CAGR': 'Compound Average Growth Rate',
               'MDD': 'Maximum Drawdown',
               'NHR': 'Normalized Hit Ratio',
               'OTS': 'Optimal trade size'}

    def __init__(self, data, kb, xb, levelb, lookbackb, ks, xs, levels, lookbacks, start=None, end=None):

        self.kb = kb
        self.xb = xb
        self.ks = ks
        self.xs = xs
        self.levelb = levelb
        self.nb = lookbackb
        self.levels = levels
        self.ns = lookbacks
        self.data = data[["Date", "Close", f"{levelb}_{lookbackb}", f"{levels}_{lookbacks}"]]  # the dataframe
        self.data['yr'] = self.data['Date'].dt.year
        self.n_days = (self.data.Date.iloc[-1] - self.data.Date.iloc[0])
        self.start = start  # the beginning date of the sample period
        self.end = end  # the ending date of the sample period

    def generate_signals(self, charts=False):
        self.data = self.data.loc[:, ~self.data.columns.duplicated()]
        self.data.dropna()
        self.data.reset_index(inplace=True)
        self.data = self.data.drop(['index'], axis=1)

        ## creating signal
        ##Upper and lower bounds
        buy_cond1 = (self.data[f"{self.levelb}_{self.nb}"].shift(self.kb+1).fillna(method="bfill") > self.data["Close"].shift(self.kb+1).fillna(method="bfill"))
        buy_cond2 = self.data["Close"] > math.inf
        for i in range(len(self.data)):
            buy_cond2.iloc[i] = all(self.data[f"{self.levelb}_{self.nb}"].shift(self.kb - j).fillna(method="bfill").iloc[i] <
                                    self.data["Close"].shift(self.kb - j).fillna(method="bfill").iloc[i] for j in range(self.kb + 1))
        buy_mask = (buy_cond1) & (buy_cond2) & (abs(self.data[f"{self.levelb}_{self.nb}"]-self.data["Close"])/self.data["Close"]>self.xb)

        sell_cond1 = (self.data[f"{self.levels}_{self.ns}"].shift(self.ks + 1).fillna(method="bfill") < self.data["Close"].shift(self.ks + 1).fillna(method="bfill"))
        sell_cond2 = self.data["Close"] > math.inf
        for i in range(len(self.data)):
            sell_cond2.iloc[i] = all(self.data[f"{self.levels}_{self.ns}"].shift(self.ks - j).fillna(method="bfill").iloc[i] >self.data["Close"].shift(self.ks - j).fillna(method="bfill").iloc[i] for j in range(self.ks + 1))
        sell_mask = (sell_cond1) & (sell_cond2) & (abs(self.data[f"{self.levels}_{self.ns}"]-self.data["Close"])/self.data["Close"]>self.xs)

        #buy_mask = ((self.data["fisher"] >= self.data["lb"]) & (self.data["fisher"]>=self.data["ub"]))
        #sell_mask = ((self.data["fisher"] < self.data["lb"])|(self.data["fisher"]<self.data["ub"])&(self.data["fisher"]>=self.data["lb"]))

        bval = +1
        sval = 0 # -1 if short selling is allowed, otherwise 0

        self.data['signal_bounds'] = np.nan
        self.data.loc[buy_mask, 'signal_bounds'] = bval
        self.data.loc[sell_mask, 'signal_bounds'] = sval
        # initialize with long
        self.data["signal_bounds"][0] = 1

        self.data.signal_bounds = self.data.signal_bounds.fillna(method="ffill")
        #self.data.signal_bounds = self.data.signal_bounds.fillna(0)

        self.data["signal"] = self.data.signal_bounds

        # Closing positions at end of time period
        self.data["signal"][-1:] = 0

        buy_plot_mask = ((self.data.signal.shift(-1) == bval) & (self.data.signal == sval))
        sell_plot_mask = ((self.data.signal.shift(-1) == sval) & (self.data.signal == bval))

        #initialize with a buy position
        buy_plot_mask[0] = True

        mask = ((self.data.signal == 1) & (self.data.signal.shift(1)==0)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

        #self.data.to_csv("Test.csv")

        ## display chart
        if charts:
            plt.plot(self.data['Date'].to_numpy(), self.data['Close'].to_numpy(), color='black', label='Price')
            plt.plot(self.data['Date'].to_numpy(), self.data[f"{self.level}_{self.n}"].to_numpy(), color='orange', label=f"{self.level}_{self.n}")
            plt.plot(self.data.loc[buy_plot_mask, 'Date'].to_numpy(), self.data.loc[buy_plot_mask, 'Close'].to_numpy(), r'^', ms=15,
                     label="Entry Signal", color='green', markeredgecolor='k', markeredgewidth=1)
            plt.plot(self.data.loc[sell_plot_mask, 'Date'].to_numpy(), self.data.loc[sell_plot_mask, 'Close'].to_numpy(), r'^', ms=15,
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

        return self.data#[["Date", "signal"]]

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
        self.data["S_Return"] = self.data["S_Return"] + (self.int/25200)*(1-self.data['signal'].shift(1))
        self.data.dropna(inplace=True)
        self.data['Market_Return'] = self.data['Return'].expanding().sum()
        self.data['Strategy_Return'] = self.data['S_Return'].expanding().sum()
        self.data['Portfolio Value'] = ((self.data['Strategy_Return'] + 1) * self.allocation)
        # self.data['Wins'] = np.where(self.data['S_Return'] > 0, 1, 0)
        # self.data['Losses'] = np.where(self.data['S_Return'] < 0, 1, 0)

        ## Daywise Performance
        d_perform = {}
        num_trades = self.data.iloc[-1]["trade_num"]

        sortino_of_trades =[]
        for i in range(1, num_trades + 1):
            try:
                if self.data.loc[(self.data["trade_num"] == i) & (self.data["S_Return"] < 0), "S_Return"].std() != 0:
                    sortino_of_trades.append((self.data.loc[self.data["trade_num"] == i, "S_Return"].mean() - 0.06 / 252) / self.data.loc[
                        (self.data["trade_num"] == i) & (self.data["S_Return"] < 0), "S_Return"].std() * (252 ** .5))
                else:
                    sortino_of_trades.append(5)
            except:
                sortino_of_trades.append(0)

        if len(sortino_of_trades)>0:
            d_perform['avg_sortino_of_trades'] = sum(sortino_of_trades) / len(sortino_of_trades)
        else:
            d_perform['avg_sortino_of_trades'] = 0

        # d_perform['TotalWins'] = self.data['Wins'].sum()
        # d_perform['TotalLosses'] = self.data['Losses'].sum()
        # d_perform['TotalTrades'] = d_perform['TotalWins'] + d_perform['TotalLosses']
        # if d_perform['TotalTrades']==0:
        #     d_perform['HitRatio'] = 0
        # else:
        #     d_perform['HitRatio'] = round(d_perform['TotalWins'] / d_perform['TotalTrades'], 2)
        # d_perform['SharpeRatio'] = (self.data["S_Return"].mean() -0.06/252)/ self.data["S_Return"].std() * (252 ** .5)
        # d_perform['StDev Annualized Downside Return'] = self.data.loc[self.data["S_Return"]<0, "S_Return"].std() * (252 ** .5)
        # #print(self.data["S_Return"])#.isnull().sum().sum())
        # if math.isnan(d_perform['StDev Annualized Downside Return']):
        #     d_perform['StDev Annualized Downside Return'] = 0.0
        # #print(d_perform['StDev Annualized Downside Return'])
        # if d_perform['StDev Annualized Downside Return'] != 0.0:
        #     d_perform['SortinoRatio'] = (self.data["S_Return"].mean()-0.06/252)*252/ d_perform['StDev Annualized Downside Return']
        # else:
        #     d_perform['SortinoRatio'] = 0
        # if len(self.data['Strategy_Return'])!=0:
        #     d_perform['CAGR'] = (1 + self.data['Strategy_Return']).iloc[-1] ** (365.25 / self.n_days.days) - 1
        # else:
        #     d_perform['CAGR'] = 0
        # d_perform['MaxDrawdown'] = (1.0 - self.data['Portfolio Value'] / self.data['Portfolio Value'].cummax()).max()
        self.daywise_performance = pd.Series(d_perform)
        #
        # ## Tradewise performance
        # ecdf = self.data[self.data["signal"] == 1]
        # trade_wise_results = []
        # if len(ecdf) > 0:
        #     for i in range(max(ecdf['trade_num'])):
        #         trade_num = i + 1
        #         entry = ecdf[ecdf["trade_num"] == trade_num].iloc[0]["Close"]
        #         exit = ecdf[ecdf["trade_num"] == trade_num].iloc[-1]["Close"]
        #         trade_wise_results.append({'Trade Number': trade_num, 'Entry': entry, 'Exit': exit})
        # trade_wise_results = pd.DataFrame(trade_wise_results)
        # d_tp = {}
        # if len(trade_wise_results) > 0:
        #     trade_wise_results["Win/Loss"] = np.where(trade_wise_results["Exit"] > trade_wise_results["Entry"], "Win",
        #                                               "Loss")
        #     trade_wise_results["Return on Trade"] = trade_wise_results["Exit"] / trade_wise_results["Entry"] - 1
        #     d_tp["TotalWins"] = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Win"])
        #     d_tp["TotalLosses"] = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"])
        #     d_tp['TotalTrades'] = d_tp["TotalWins"] + d_tp["TotalLosses"]
        #     if d_tp['TotalTrades'] == 0:
        #         d_tp['HitRatio'] = 0
        #     else:
        #         d_tp['HitRatio'] = round(d_tp['TotalWins'] / d_tp['TotalTrades'], 4)
        #     d_tp['AvgWinRet'] = np.round(
        #         trade_wise_results[trade_wise_results["Win/Loss"] == "Win"]["Return on Trade"].mean(), 4)
        #     if math.isnan(d_tp['AvgWinRet']):
        #         d_tp['AvgWinRet'] = 0.0
        #     d_tp['AvgLossRet'] = np.round(
        #         trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"]["Return on Trade"].mean(), 4)
        #     if math.isnan(d_tp['AvgLossRet']):
        #         d_tp['AvgLossRet'] = 0.0
        #     if d_tp['AvgLossRet'] != 0:
        #         d_tp['WinByLossRet'] = np.round(abs(d_tp['AvgWinRet'] / d_tp['AvgLossRet']), 2)
        #     else:
        #         d_tp['WinByLossRet'] = 0
        #     if math.isnan(d_tp['WinByLossRet']):
        #         d_tp['WinByLossRet'] = 0.0
        #     if math.isinf(d_tp['WinByLossRet']):
        #         d_tp['WinByLossRet'] = 0.0
        # else:
        #     d_tp["TotalWins"] = 0
        #     d_tp["TotalLosses"] = 0
        #     d_tp['TotalTrades'] = 0
        #     d_tp['HitRatio'] = 0
        #     d_tp['AvgWinRet'] = 0
        #     d_tp['AvgLossRet'] = 0
        #     d_tp['WinByLossRet'] = 0
        # self.tradewise_performance = pd.Series(d_tp)

        return self.data

    # @staticmethod
    # def kelly(p, b):
    #     """
    #     Static method: No object or class related arguments
    #     p: win prob, b: net odds received on wager, output(f*) = p - (1-p)/b
    #
    #     Spreadsheet example
    #         from sympy import symbols, solve, diff
    #         x = symbols('x')
    #         y = (1+3.3*x)**37 *(1-x)**63
    #         solve(diff(y, x), x)[1]
    #     Shortcut
    #         .37 - 0.63/3.3
    #     """
    #     return np.round(p - (1 - p) / b, 4)

    def plot_performance(self, allocation=1, interest_rate = 6):
        # intializing a variable for initial allocation
        # to be used to create equity curve
        self.signal_performance(allocation, interest_rate)

        # yearly performance
        #self.yearly_performance()

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

    # def yearly_performance(self):
    #     """
    #     Instance method
    #     Adds an instance attribute: yearly_df
    #     """
    #     _yearly_df = self.data.groupby(['yr', 'signal']).S_Return.sum().unstack()
    #     _yearly_df.rename(columns={-1.0: 'Sell', 1.0: 'Buy'}, inplace=True)
    #     _yearly_df['Return'] = _yearly_df.sum(1)
    #
    #     # yearly_df
    #     self.yearly_df = _yearly_df.style.bar(color=["#ffcccb", '#90ee90'], align='zero').format({
    #         'Sell': '{:,.2%}'.format, 'Buy': '{:,.2%}'.format, 'Return': '{:,.2%}'.format})

    # def update_metrics(self):
    #     """
    #     Called from the opt_matrix class method
    #     """
    #     d_field = {}
    #
    #     d_field['PortfolioValue'] = self.data['Portfolio Value']
    #     d_field['Sharpe'] = self.daywise_performance.SharpeRatio
    #     d_field['Sortino'] = self.daywise_performance.SortinoRatio
    #     d_field['CAGR'] = self.daywise_performance.CAGR
    #     d_field['MDD'] = self.daywise_performance.MaxDrawdown
    #     d_field['NHR'] = self.tradewise_performance.NormHitRatio
    #     #d_field['OTS'] = self.tradewise_performance.OptimalTradeSize
    #     d_field['AvgWinLoss'] = self.tradewise_performance.WinByLossRet
    #
    #     return d_field
    #
    # @classmethod
    # def opt_matrix(cls, data, buy_fish, sell_fish, metrics, optimal_sol=True):
    #     """
    #
    #     """
    #     c_green = sns.light_palette("green", as_cmap=True)
    #     c_red = sns.light_palette("red", as_cmap=True)
    #
    #     d_mats = {m: [] for m in metrics}
    #
    #
    #     for lows in buy_fish:
    #         d_row = {m: [] for m in metrics}
    #         for highs in sell_fish:
    #             # if highs>=lows:
    #             obj = cls(data, zone_high=highs, zone_low=lows)  ## object being created from the class
    #             obj.generate_signals(charts=False)
    #             obj.signal_performance(10000, 6)
    #             d_field = obj.update_metrics()
    #             for m in metrics: d_row[m].append(d_field.get(m, np.nan))
    #             # else:
    #             #     for m in metrics: d_row[m].append(0)
    #         for m in metrics: d_mats[m].append(d_row[m])
    #
    #     d_df = {m: pd.DataFrame(d_mats[m], index=buy_fish, columns=sell_fish) for m in metrics}
    #
    #     def optimal(_df):
    #
    #         _df = _df.stack().rank()
    #         _df = (_df - _df.mean()) / _df.std()
    #         return _df.unstack()
    #
    #     if optimal_sol:
    #         # d_df['Metric'] = 0
    #         # if 'Sortino' in metrics: d_df['Metric'] += optimal(d_df['Sortino'])
    #         # if 'PVal' in metrics: d_df['Metric'] += optimal(d_df['PortfolioValue'])
    #         # if 'Sharpe' in metrics: d_df['Metric'] += 2 * optimal(d_df['Sharpe'])
    #         # if 'NHR' in metrics: d_df['Metric'] += optimal(d_df['NHR'])
    #         # if 'CAGR' in metrics: d_df['Metric'] += optimal(d_df['CAGR'])
    #         # if 'MDD' in metrics: d_df['Metric'] -= 2 * optimal(d_df['MDD'])
    #         # d1 = pd.DataFrame(d_df['Metric'])
    #         # val = np.amax(d1.to_numpy())
    #         # bf = d1.index[np.where(d1 == val)[0][0]]
    #         # sf = d1.columns[np.where(d1 == val)[1][0]]
    #
    #         #******
    #         d2 = pd.DataFrame(d_df[metrics[0]])
    #         val = np.amax(d2.to_numpy())
    #         bf = d2.index[np.where(d2 == val)[0][0]]
    #         sf = d2.columns[np.where(d2 == val)[1][0]]
    #         #******
    #         #return d_df
    #
    #         #print(f"Most optimal pair is Lower Bound:{bf}, Upper Bound:{sf}, with max {metrics[0]}:{val}")
    #         #metrics.insert(0, 'Signal')
    #
    #     # for m in metrics:
    #     #     display(HTML(d_df[m].style.background_gradient(axis=None, cmap=
    #     #     c_red if m == "MDD" else c_green).format(
    #     #         ("{:,.2}" if m in ["Sharpe", "Signal"] else "{:.2%}")).set_caption(m).render()))
    #
    #
    #     return (bf, sf, val)

def BFO_vol_mom_strategy(temp_og, kmax=9, xmax=5):

    data = [temp_og]
    k_list = [i for i in range(kmax + 1)]
    x_list = [i / 100 for i in range(xmax + 1)]
    levels = ["vah", "val", "poc"]
    lookbacks = [2, 5, 10, 21, 42, 63, 126, 252, 504]
    inputs = list(itertools.product(data, k_list, x_list, levels, lookbacks, k_list, x_list, levels, lookbacks))

    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results = pool.map(backtest_mom_for_avg_sortino, inputs)
    finally:  # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    results = pd.DataFrame(results)
    results.sort_values("avg_sortino_of_trades", ascending=False, inplace=True)
    results.reset_index(drop=True, inplace=True)
    return results

def backtest_mom_for_avg_sortino(input):

    strat = vol_mom_strategy(data = input[0], kb = input[1], xb = input[2], levelb = input[3], lookbackb = input[4], ks = input[5], xs = input[6], levels = input[7], lookbacks = input[8])
    strat.generate_signals()
    strat.signal_performance(10000, 6)
    return {"kb" : input[1], "xb" : input[2], "levelb" : input[3], "lookbackb" : input[4], "ks" : input[5], "xs" : input[6], "levels" : input[7], "lookbacks" : input[8],
            "avg_sortino_of_trades": strat.daywise_performance['avg_sortino_of_trades']}

def valid_dates(dates_all):
    dates = []
    i = 0
    while True:
        dates.append(dates_all[i])
        #if dates_all[i] > pd.to_datetime(date.today()):
        if dates_all[i] > pd.to_datetime(date.today().replace(month=11, day=12)):
            break
        i = i + 1
    return dates

if __name__ == '__main__':

    train_months = 48
    date_i = 14
    ticker = '.NSEI'

    print(f"Processing {ticker}")

    # with open(f'NSEI_Volume_Momentum_Backtest_Azure_temp_og.pkl','rb') as file:
    #     temp_og = pickle.load(file)

    temp_og = pd.read_csv(
        "https://github.com/AcsysAlgo/AzureData/blob/main/NSEI_Volume_Momentum_Backtest_Azure_temp_og.csv?raw=true")
    temp_og.drop(columns=["Unnamed: 0"], inplace=True)
    temp_og["Date"] = pd.to_datetime(temp_og["Date"])

    dates_all_ss = pd.date_range(start=str(temp_og.iloc[1]['Date'] + timedelta(days=503))[:10], end="2024-06-15", freq=f'3M')
    dates = valid_dates(dates_all_ss)

    temp = temp_og.loc[(temp_og["Date"] > str(dates[date_i])) & (temp_og["Date"] < str(
        dates[date_i + int(train_months / 3)]))].reset_index().drop(['index'], axis=1)

    with open(f'{ticker}_TrainYrs_{int(train_months / 12)}_All_Strategies_{date_i}.pkl',
              'wb') as file:
        pickle.dump(BFO_vol_mom_strategy(temp), file)