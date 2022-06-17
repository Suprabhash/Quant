"""
Definition of the strategy class
"""
import pickle
import time

import pandas as pd
from Backtester.backtester import backtester
from Utils.add_features import feature_creator
from Data.data_retrieval import get_data as get_Data

class strategy:
    def __init__(self):
        """
        Initialises the strategy with the parameters
        """
        pass

    def initialize_ticker_frequency(self, ticker, frequency):
        self.ticker = ticker
        self.frequency = frequency

    def get_data(self, ticker, frequency):
        self.ticker = ticker
        self.frequency = frequency
        self.data = get_Data(ticker, frequency)
        return self.data

    def add_features(self, data, params):
        self.feature_creator = feature_creator

    def create_signals(self, df, params):
        """
        Contains the logic required for calculating the signals from the features created and the strategy parameters
        :param df: contains the dataframe with OHLCV and feature data
        :param params: contains the paraeters required to calculate signals
        :return:  returns the dataframe with the signals added
        """
        pass

    def do_backtest(self, *params, allocation, interest_rate, plot=False, start=None, end=None, save_plot_to=None): #
        strat = backtester(self.data, self, params[0], start=start, end=end)
        signals = strat.generate_signals()
        equity_curve = strat.signal_performance(allocation, interest_rate, self.frequency)
        if plot:
            strat.plot_performance(self.frequency, allocation=allocation, interest_rate = interest_rate, save_to=save_plot_to)
        return signals, equity_curve

    def lookup_closing_state(self, lookbacks, search_params, date, strategy_name, data_freq):
        """ This function looks up the Train end date along with the param space to find the closing signal at the end of the training date.
            Currently searches all the lookbacks file even though result might be same"""
        for lookback in lookbacks:
            with open(f'Caches/{self.ticker}/{data_freq}/{strategy_name}/SelectedStrategies/All_{lookback}.pkl', 'rb') as file:
                strategies = pickle.load(file)
            for strat in strategies:
                if strat['Train End Date'] == date:
                    df = strat['Strategies']
                    if search_params in list(df['params']):
                        closing_state = df[df['params'] == search_params].reset_index()['closing_state'][0]
                        # print(closing_state)
                        return closing_state
                    else:
                        print('Not present')
                        return None
                else:
                    continue
        return None

    def get_optimization_params():
        pass

    def plotting_function(self, df):
        """
        Contains the code to visualise trades from the dataframe obtained from the backtest
        :param df: Dataframe obtained from backtest
        :return: plots
        """
        pass




