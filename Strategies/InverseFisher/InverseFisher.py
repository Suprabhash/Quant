from pathos.multiprocessing import ProcessingPool

from ..strategy import strategy
from . import config
from Utils.add_features import add_zscore, add_tanh_zscores, add_F
import numpy as np
import matplotlib.pyplot as plt
import itertools
import Utils.add_features as function_lib
from Utils.Num_cores import num_cores
from Utils.utils import callable_functions_helper

class InverseFisher(strategy):
    def __init__(self):
        strategy.__init__(self)

    def add_features(self, data, params):
        data["ohlc4"] = (data["Open"] + data["Close"] + data["High"] + data["Low"]) / 4

        # Zscores Addition
        params1 = [[data]] + [params[2]]
        inputs = list(itertools.product(*params1))
        pool = ProcessingPool(num_cores)
        results = pool.map(add_zscore, inputs)
        pool.clear()

        for result in results:
            data[result.columns[-1]] = result.filter(regex='z_score.+')

        # Tanh Addition
        params2 = [[data]] + [params[2]]
        inputs = list(itertools.product(*params2))
        pool = ProcessingPool(num_cores)
        results = pool.map(add_tanh_zscores, inputs)
        pool.clear()

        for result in results:
            data[result.columns[-1]] = result.filter(regex='tanh_.+')

        # F1 Addition
        params3 = [[data]] + [params[2], list(set(params[0]+params[1])), list(set(params[3]+params[4]))]
        inputs = list(itertools.product(*params3))
        pool = ProcessingPool(num_cores)
        results = pool.map(add_F, inputs)
        pool.clear()

        for result in results:
            try:
                data[result.columns[-1]] = result.filter(regex='.+_tanh_.+')
            except:
                pass

        return data

    def create_signals(self, df_input, *params):
        """
        Creates signals based on indicator values and its defined thresholds. A buy signal is created when indicator crosses over lower bound from bottom to top
        A sell signal is created when indicator crosses over upper bound from top to bottom
        :param df: Receives the OHLCV data frame with a column for Datetime and indicator values in columns of the format f"indicator{Lookback_value}"
        :param params: A list of the lookback, lower bound and upper bound
        :return: Returns a dataframe consisting of the Datetime and the signal
        """

        df = df_input.copy()
        params = params[0]
        params = callable_functions_helper(params)[0]
        f1 = getattr(function_lib, params[0])
        f2 = getattr(function_lib, params[1])
        zscore_lookback = params[2]
        f1_lookback = params[3]
        f2_lookback = params[4]
        bound1 = params[5]
        bound2 = params[6]


        if f1.__name__ == "x":
            df["indicator1"] = df[f"tanh_z_score{zscore_lookback}_ohlc4"]
        else:
            df["indicator1"] = df[f"{f1.__name__ }{f1_lookback}_tanh_z_score{zscore_lookback}_ohlc4"]


        if f2.__name__ == "x":
            df["indicator2"] = df[f"tanh_z_score{zscore_lookback}_ohlc4"]
        else:
            df["indicator2"] = df[f"{f2.__name__}{f2_lookback}_tanh_z_score{zscore_lookback}_ohlc4"]


        df = df.loc[(df.indicator1 != 0)]
        df = df.loc[(df.indicator2 != 0)]
        df["indicator1_lag"] = df.indicator1.shift(1)
        df["indicator2_lag"] = df.indicator2.shift(1)
        df["b1"] = bound1
        df["b2"] = bound2
        df.dropna()
        df.reset_index(inplace=True, drop=True)

        ## creating signal
        buy_mask = (df["indicator1"] > df["b1"]) & (df["indicator1_lag"] < df["b1"])
        sell_mask = ((df["indicator2"] < df["b2"]) & (df["indicator2_lag"] > df["b2"]))

        bval = +1
        sval = 0  #

        df['signal_bounds'] = np.nan
        df.loc[buy_mask, 'signal_bounds'] = bval
        df.loc[sell_mask, 'signal_bounds'] = sval

        df.signal_bounds = df.signal_bounds.fillna(method="ffill")
        df.signal_bounds = df.signal_bounds.fillna(
            0)

        df["signal"] = df.signal_bounds
        df = df.reset_index(drop=True)
        return df[["Datetime", "signal", "indicator1", "indicator2", "b1", "b2"]]


    def plotting_function(self, df, save_to= None):
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

        plt.plot(df['Datetime'], df['indicator1'], color='black', label='indicator1')
        plt.plot(df['Datetime'], df['indicator2'], color='black', label='indicator2')
        plt.plot(df['Datetime'], df['b1'], color='red', label='Bound 1')
        plt.plot(df['Datetime'], df['b2'], color='green', label='Bound 2')
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
        feature_space = [config.params_searchspace["f1"], config.params_searchspace["f2"], config.params_searchspace["zscores_lookbacks"], config.params_searchspace["f1_lookbacks"], config.params_searchspace["f2_lookbacks"]]
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
        return {"feature_space": feature_space, "parameter_searchspace": parameter_searchspace, "metric_searchspace": metric_searchspace,
                "strategy_lookbacks": strategy_lookbacks,
                "number_of_optimisation_periods": number_of_optimisation_periods,
                "recalib_periods": recalib_periods, "num_strategies": num_strategies, "metrics_opt": metrics_opt,
                "number_selected_filteredstrategies": number_selected_filteredstrategies,
                "consider_selected_strategies_over": consider_selected_strategies_over,
                "number_selected_strategies": number_selected_strategies,
                "starting_points": starting_points}








