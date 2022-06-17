"""
This file creates a backtest from the input parameters and a function that codeifies the strategy logic.
It creates metrics. The signal generator and plotting functions can be imported from the backtest resources in the strategy folder
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

from Metrics.Metrics import *

class backtester:

    def __init__(self, data, strategy, *params, start=None, end=None):
        """
        :param data: Inputs the OHLCV data
        :param params: Inputs the strategy parameters
        :param strategy_signal_creator: This function is used to create signals based on the logic of the strategy
        :param start: Start date of the backtest
        :param end:  End date of the backtest
        """
        # Try-except used if pre-built signals are used to compute performance
        self.strategy = strategy
        try:
            self.params = params[0]
        except:
            self.params = None
        self.data = data
        try:
            self.strategy_signal_creator = strategy.create_signals
            self.plotting_function = strategy.plotting_function
        except:
            pass
        self.start = start
        self.end = end
        if start is not None:
            self.data = self.data[(self.data["Datetime"]>start)]
        if end is not None:
            self.data = self.data[(self.data["Datetime"] <= end)]
        self.data.reset_index(drop=True, inplace=True)

    def generate_signals(self):
        """
        Creates signals based on the strategy logic and plots the backtest, if requires
        :param plot: Takes a Boolean. Creates plots, if Ta plotting function has been passed
        :return:
        """
        self.data = pd.concat([self.data.set_index("Datetime"), self.strategy_signal_creator(self.data, self.params).set_index("Datetime")], axis=1, join="inner").reset_index()
        return self.data[["Datetime", "signal"]]

    def signal_performance(self, allocation, interest_rate, data_frequency, stats_req=[]):
        """
        This function calculates the strategy metrics and includes daywise, yearwise and tradewise statistics. The list of stats that need to be calculated can be passed
        so that backtester calculates only necessary metrics to save computation. If no list is passed, all metrics are calculated.
        :param allocation: This is the initial value of the portfolio
        :param interest_rate: When neutral, the backtester invests the portfolio into fixed income, the interest rate for which can be provided here. Provide annual rate of interest as a percent.
        :param stats_req: Provides a list of statistics required
        :return: Returns a series of stats.
        """
        self.allocation = allocation
        self.int = interest_rate
        if data_frequency == 'H':
            annual_factor = 252*24
        else:
            annual_factor = 252

        # creating returns and portfolio value series #This code runs compulsorily
        self.data['Return'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['S_Return'] = self.data['signal'].shift(1) * self.data['Return']
        self.data['S_Return'] = self.data['S_Return'].fillna(0)
        self.data["S_Return"] = self.data["S_Return"] + (self.int/(100*annual_factor))*(1-self.data['signal'].shift(1))
        self.data['Market_Return'] = np.exp(self.data['Return'].expanding().sum())
        self.data['Strategy_Return'] = np.exp( self.data['S_Return'].expanding().sum())
        self.data['Portfolio_Value'] = ((self.data['Strategy_Return'] + 1) * self.allocation)

        mask = ((self.data.signal == 1) & (self.data.signal.shift(1) == 0)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

        stats = {}
        ## Daywise Performance
        ## Hit Ratio
        if ("DailyHitRatio" in stats_req) or (stats_req == ["all"]):
            stats['DailyHitRatio'] = DailyHitRatio(self.data)

        ## Sharpe Ratio
        if ("SharpeRatio" in stats_req) or (stats_req == ["all"]):
            if data_frequency == 'H':
                stats['SharpeRatio'] = SharpeRatio_hourly(self.data)
            else:
                stats['SharpeRatio'] = SharpeRatio(self.data)

        ## Sortino Ratio
        if ("SortinoRatio" in stats_req) or (stats_req == ["all"]):
            if data_frequency == 'H':
                stats['SortinoRatio'] = SortinoRatio_hourly(self.data)
            else:
                stats['SortinoRatio'] = SortinoRatio(self.data)

        ## CAGR
        if ("CAGR" in stats_req) or (stats_req == ["all"]):
            stats['CAGR'] = CAGR(self.data)

        if ("MaxDrawdown" in stats_req) or (stats_req == ["all"]):
            stats['MaxDrawdown'] = MaxDrawdown(self.data)

        ## Tradewise performance
        if ("HitRatio" in stats_req) or (stats_req == ["all"]):
            stats['HitRatio'] = MaxDrawdown(self.data)

        if ("WinByLossRet" in stats_req) or (stats_req == ["all"]):
            stats['WinByLossRet'] = WinByLossRet(self.data)

        self.metrics = pd.Series(stats)

        if stats_req == []:
            return self.data[['Datetime', 'Close', 'signal', 'Return', 'S_Return', 'trade_num', 'Market_Return', 'Strategy_Return', 'Portfolio_Value']]
        else:
            return self.data[['Datetime', 'Close', 'signal', 'Return', 'S_Return', 'trade_num', 'Market_Return','Strategy_Return', 'Portfolio_Value']], self.metrics

    def plot_performance(self, data_frequency, allocation=1, interest_rate = 6, save_to=None):

        if self.strategy is not None:
            if save_to is not None:
                try:
                    for i in range(len(self.params)):
                        if callable(self.params[i]):
                            self.params[i] = self.params[i].__name__
                    self.plotting_function(self.data, save_to=save_to+"/"+f"{str(self.start)[:11]}"+f"{str(self.end)[:11]}"+f"{self.params}"+"StrategyPlot.jpg")
                except Exception as e:
                    # print(e)
                    self.plotting_function(self.data, save_to=save_to)
            else:
                self.plotting_function(self.data)

        self.signal_performance(allocation, interest_rate, data_frequency)
        # Plotting the Performance of the strategy
        plt.plot(self.data['Datetime'], self.data['Market_Return'], color='black', label='Market Returns')
        plt.plot(self.data['Datetime'], self.data['Strategy_Return'], color='blue', label='Strategy Returns')
        # plt.plot(self.data['Date'],self.data['fisher_rate'],color='red', label= 'Fisher Rate')
        plt.title('Strategy Backtest')
        plt.legend(loc=0)
        plt.tight_layout()

        if save_to is not None:
            try:
                plt.savefig(save_to+"/"+f"{str(self.start)[:11]}"+f"{str(self.end)[:11]}"+f"{self.params}"+"Performance.jpg")
            except:
                plt.savefig(save_to[:-4]+"P.jpg")
        else:
            plt.show()
        plt.clf()
        # return(self.data)

    