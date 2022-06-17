from Backtester.backtester import backtester
from . import config
from ..strategy import strategy
from Strategies.Alpha.config import *
from Utils.add_features import add_hurst, add_MA_hurst, add_ROC_MA_hurst
import numpy as np
import matplotlib.pyplot as plt
from hurst import *
import itertools
from pathos.multiprocessing import ProcessingPool

class RescaledRange(strategy):
    def __init__(self):
        strategy.__init__(self)

    def add_features(self, data, params):
        #Hurst Addition
        params1 = [[data]] + [params[0]]
        inputs = list(itertools.product(*params1))
        pool = ProcessingPool()
        results = pool.map(add_hurst, inputs)
        pool.clear()

        for result in results:
            data[result.columns[-1]] = result.filter(regex='H[0-9]+')

        # MA Addition
        params2 = [[data]] + [params[0], params[1]]
        inputs = list(itertools.product(*params2))
        pool = ProcessingPool()
        results = pool.map(add_MA_hurst, inputs)
        pool.clear()

        for result in results:
            data[result.columns[-1]] = result.filter(regex='MA[0-9]+_H[0-9]+')

        # ROC Addition
        params3 = [[data]] + [params[0], params[1], params[2]]
        inputs = list(itertools.product(*params3))
        pool = ProcessingPool()
        results = pool.map(add_ROC_MA_hurst, inputs)
        pool.clear()

        for result in results:
            data[result.columns[-1]] = result.filter(regex='ROC[0-9]+_MA[0-9]+_H[0-9]+')

        return data

    def create_signals(self, df_input, *params):

        params = params[0]
        df = df_input.copy()
        hurst_lookback = params[0]
        MA_lookback = params[1]
        ROC_lookback = params[2]
        level = params[3]

        df["ROC(MA(hurst))"] = df[f"ROC{ROC_lookback}_MA{MA_lookback}_H{hurst_lookback}"]
        df["level"] = level
        df["signal_bounds"] = np.where(df["ROC(MA(hurst))"] > df["level"], 1, 0)

        # if closing_state:                                 # This portion is reached at the end while stitching together all the signals
        #     df["signal_bounds"][0] = closing_state
        #     df.signal_bounds = df.signal_bounds.fillna(method="ffill")
        # else:
        df.signal_bounds = df.signal_bounds.fillna(method="ffill")
        df.signal_bounds = df.signal_bounds.fillna(0) # This basically fills the first few NaNs with 0s. This will be skipped in the final signal stitching

        df["signal"] = df.signal_bounds
        df1 = df.reset_index(drop=True)
        return df1[["Datetime", "signal", "ROC(MA(hurst))", "level"]]

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

        df.set_index("Datetime", inplace=True)
        df[["ROC(MA(hurst))", "signal", "level"]].plot(secondary_y="signal")
        plt.title('Strategy Backtest')
        plt.legend(loc=0)
        d_color = {}
        d_color[1] = '#90ee90'  ## light green
        d_color[-1] = "#ffcccb"  ## light red
        d_color[0] = '#ffffff'

        df.reset_index(inplace=True)

        if save_to != None:
            plt.savefig(save_to)
        else:
            plt.show()
        plt.clf()

    @staticmethod
    def get_optimization_params():
        feature_space = [config.params_searchspace["hurst_range"], config.params_searchspace["MA_range"], config.params_searchspace["ROC_range"]]
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
        return { "feature_space": feature_space, "parameter_searchspace": parameter_searchspace, "metric_searchspace": metric_searchspace,
                "strategy_lookbacks": strategy_lookbacks,
                "number_of_optimisation_periods": number_of_optimisation_periods,
                "recalib_periods": recalib_periods, "num_strategies": num_strategies, "metrics_opt": metrics_opt,
                "number_selected_filteredstrategies": number_selected_filteredstrategies,
                "consider_selected_strategies_over": consider_selected_strategies_over,
                "number_selected_strategies": number_selected_strategies,
                "starting_points": starting_points}


