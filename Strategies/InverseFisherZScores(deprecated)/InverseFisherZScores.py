from ..strategy import strategy
from . import config
from Strategies.InverseFisher.config import *
from Utils.add_features import add_zscore, add_tanh
import numpy as np
import matplotlib.pyplot as plt

class InverseFisherZScores(strategy):
    def __init__(self):
        strategy.__init__(self)

    def add_features(self, data, params):
        data["ohlc4"] = (data["Open"] + data["Close"] + data["High"] + data["Low"])/4
        zscores_lookback = params[0]
        data = add_zscore(data, "ohlc4", zscores_lookback)
        data = add_tanh(data, f"z_score_ohlc4_{zscores_lookback}")
        data.drop(columns = [column for column in data.columns if ("z_score" in column) and ("tanh" not in column)], inplace=True)
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data

    def create_signals(self, df_input, closing_state,  *params):
        """
            Creates signals based on indicator values and its defined thresholds. A buy signal is created when indicator crosses over lower bound from bottom to top
            A sell signal is created when indicator crosses over upper bound from top to bottom
            :param df: Receives the OHLCV data frame with a column for Datetime and indicator values in columns of the format f"indicator{Lookback_value}"
            :param params: A list of the lookback, lower bound and upper bound
            :return: Returns a dataframe consisting of the Datetime and the signal
            """

        df = df_input.copy()
        params = params[0]
        zscore_lookback = params[0]
        lower_bound = params[1]
        upper_bound = params[2]

        df["indicator"] = df[f"tanh_z_score_ohlc4_{zscore_lookback}"]
        df = df.loc[(df.indicator != 0)]
        df["indicator_lag"] = df.indicator.shift(1)
        df["lb"] = lower_bound
        df["ub"] = upper_bound
        df.dropna()
        df.reset_index(inplace=True, drop=True)

        ## creating signal
        ##Upper and lower bounds
        buy_mask = (df["indicator"] > df["lb"]) & (df["indicator_lag"] < df["lb"])
        sell_mask = ((df["indicator"] < df["ub"]) & (df["indicator_lag"] > df["ub"]))

        bval = +1
        sval = 0  # -1 if short selling is allowed, otherwise 0

        df['signal_bounds'] = np.nan
        df.loc[buy_mask, 'signal_bounds'] = bval
        df.loc[sell_mask, 'signal_bounds'] = sval

        if closing_state:  # This portion is reached at the end while stitching together all the signals
            df["signal_bounds"][0] = closing_state
            df.signal_bounds = df.signal_bounds.fillna(method="ffill")
        else:
            df.signal_bounds = df.signal_bounds.fillna(method="ffill")
            df.signal_bounds = df.signal_bounds.fillna(
                0)  # This basically fills the first few NaNs with 0s. This will be skipped in the final signal stitching

        df["signal"] = df.signal_bounds

        df1 = df.dropna()
        df1 = df1.reset_index(drop=True)
        return df1[["Datetime", "signal", "lb", "ub", "indicator"]]


    def plotting_function(self, df, save_to=None):
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
        if save_to != None:
            plt.savefig(save_to)
        else:
            plt.show()
        plt.clf()

        plt.plot(df['Datetime'], df['indicator'], color='black', label='indicator')

        plt.plot(df['Datetime'], df['lb'], color='green', label='Lower Bound')
        plt.plot(df['Datetime'], df['ub'], color='red', label='Upper Bound')

        plt.plot(df.loc[buy_plot_mask]['Datetime'].shift(1), df.loc[buy_plot_mask]['indicator'].shift(1), r'^',
                 ms=15,
                 label="Entry Signal", color='green', markeredgecolor='k', markeredgewidth=1)
        plt.plot(df.loc[sell_plot_mask]['Datetime'].shift(1), df.loc[sell_plot_mask]['indicator'].shift(1), r'^',
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
        if save_to != None:
            plt.savefig(save_to)
        else:
            plt.show()
        plt.clf()

    @staticmethod
    def get_optimization_params():
        parameter_searchspace = [config.params_searchspace[key] for key in config.params_searchspace.keys()]
        metric_searchspace = [config.metrics]
        strategy_lookbacks = config.strategy_lookbacks
        number_of_optimisation_periods = config.number_of_optimisation_periods
        recalib_periods = config.recalib_periods
        num_strategies = config.num_strategies
        metrics_opt = config.metrics_opt
        number_selected_filteredstrategies = config.number_selected_filteredstrategies
        consider_selected_strategies_over = config.consider_selected_strategies_over
        number_selected_strategies = config.number_selected_strategies
        starting_points = config.starting_points
        return {"parameter_searchspace": parameter_searchspace, "metric_searchspace": metric_searchspace,
                "strategy_lookbacks": strategy_lookbacks,
                "number_of_optimisation_periods": number_of_optimisation_periods,
                "recalib_periods": recalib_periods, "num_strategies": num_strategies, "metrics_opt": metrics_opt,
                "number_selected_filteredstrategies": number_selected_filteredstrategies,
                "consider_selected_strategies_over": consider_selected_strategies_over,
                "number_selected_strategies": number_selected_strategies,
                "starting_points": starting_points}






