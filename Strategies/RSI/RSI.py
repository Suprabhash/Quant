from .config import number_selected_filteredstrategies, recalib_periods
from ..strategy import strategy
from Strategies.RSI.config import *
from Utils.add_features import add_RSI
import numpy as np
import matplotlib.pyplot as plt

class RSI(strategy):
    def __init__(self):
        strategy.__init__(self)
        self.parameter_searchspace = [params_searchspace[key] for key in params_searchspace.keys()]
        self.metric_searchspace = [metrics]
        self.RSI_lookbacks = RSI_lookbacks
        self.strategy_lookbacks = strategy_lookbacks
        self.number_of_optimisation_periods = number_of_optimisation_periods
        self.recalib_periods = recalib_periods
        self.num_strategies = num_strategies
        self.metrics_opt = metrics_opt
        self.number_selected_filteredstrategies = number_selected_filteredstrategies
        self.consider_selected_strategies_over = consider_selected_strategies_over
        self.metric_threshold = metric_threshold

    def add_features(self, data):
        for RSI_lookbacks in self.RSI_lookbacks:
            data = add_RSI(data, RSI_lookbacks)
        return data

    def create_signals(self, df_input, final_signal_formation, *params):
        """
            Creates signals based on RSI values and its defined thresholds. A buy signal is created when RSI crosses over lower bound from bottom to top
            A sell signal is created when fisher crosses over upper bound from top to bottom
            :param df: Receives the OHLCV data frame with a column for Datetime
            :param params: A list of the lookback, lower bound and  upper bound
            :return: Returns a dataframe consisting of the Datetime and the signal
            """
        df = df_input.copy()
        params = params[0]
        lookback = params[0]
        up_threshold = params[1]
        low_threshold = params[2]

        df['RSI'] = df[f'RSI{lookback}']
        df = df.loc[(df.RSI != 0)]
        df['RSI_lag'] = df['RSI'].shift(1)  # This will be the PREVIOUS RSI. This is done so that on any given datetime value, we have both previous and current value in the same row
        df["lb"] = low_threshold
        df["ub"] = up_threshold
        df.dropna()
        df.reset_index(inplace=True, drop=True)

        # Creating buy/sell signals
        # Setting the conditions here
        buy_mask = (df['RSI'] < low_threshold) & (df['RSI_lag'] > low_threshold)
        sell_mask = (df['RSI'] > up_threshold) & (df['RSI_lag'] < up_threshold)
        bval = 1
        sval = 0
        df['signal_bounds'] = np.nan
        df.loc[buy_mask, 'signal_bounds'] = bval
        df.loc[sell_mask, 'signal_bounds'] = sval
        df.signal_bounds = df.signal_bounds.fillna(method="ffill")
        df.signal_bounds = df.signal_bounds.fillna(0)

        df["signal"] = df.signal_bounds
        # Making the final value 0 implying closing positions at end of time period
        if not final_signal_formation:
            df["signal"][-1:] = 0
        df1 = df.reset_index(drop=True)
        return df1[["Datetime", "signal", "lb", "ub", "RSI"]]


    def plotting_function(self, df):
        bval = 1
        sval = 0
        buy_plot_mask = ((df.signal.shift(-1) == bval) & (df.signal == sval))
        sell_plot_mask = ((df.signal.shift(-1) == sval) & (df.signal == bval))

        plt.plot(df['Datetime'], df['Close'], color='black', label='Price')
        plt.plot(df.loc[buy_plot_mask]['Datetime'], df.loc[buy_plot_mask]['Close'], r'^', ms=15,
                 label="Entry Signal", color='green', markeredgecolor='k', markeredgewidth=1)
        plt.plot(df.loc[sell_plot_mask]['Datetime'], df.loc[sell_plot_mask]['Close'], r'^', ms=15,
                 label="Exit Signal", color='red', markeredgecolor='k', markeredgewidth=1)
        plt.title('Strategy Backtest')
        plt.legend(loc=0)
        d_color = {}
        d_color[1] = '#90ee90'  ## light green
        d_color[-1] = "#ffcccb"  ## light red
        d_color[0] = '#ffffff'

        j = 0
        for i in range(1, df.shape[0]):
            if np.isnan(df.signal[i - 1]):
                j = i
            elif (df.signal[i - 1] == df.signal[i]) and (i < (df.shape[0] - 1)):
                continue
            else:
                plt.axvspan(df['Datetime'][j], df['Datetime'][i],
                            alpha=0.5, color=d_color[df.signal[i - 1]], label="interval")
                j = i
        plt.show()

        plt.plot(df['Datetime'], df['RSI'], color='black', label='RSI')

        plt.plot(df['Datetime'], df['lb'], color='green', label='Lower Bound')
        plt.plot(df['Datetime'], df['ub'], color='red', label='Upper Bound')

        plt.plot(df.loc[buy_plot_mask]['Datetime'].shift(1), df.loc[buy_plot_mask]['RSI'].shift(1), r'^',
                 ms=15,
                 label="Entry Signal", color='green', markeredgecolor='k', markeredgewidth=1)
        plt.plot(df.loc[sell_plot_mask]['Datetime'].shift(1), df.loc[sell_plot_mask]['RSI'].shift(1), r'^',
                 ms=15,
                 label="Exit Signal", color='red', markeredgecolor='k', markeredgewidth=1)
        plt.title('Strategy Backtest')
        plt.legend(loc=0)
        d_color = {}
        d_color[1] = '#90ee90'  ## light green
        d_color[-1] = "#ffcccb"  ## light red
        d_color[0] = '#ffffff'

        j = 0
        for i in range(1, df.shape[0]):
            if np.isnan(df.signal[i - 1]):
                j = i
            elif (df.signal[i - 1] == df.signal[i]) and (i < (df.shape[0] - 1)):
                continue
            else:
                plt.axvspan(df['Datetime'][j], df['Datetime'][i],
                            alpha=0.5, color=d_color[df.signal[i - 1]], label="interval")
                j = i
        plt.show()







