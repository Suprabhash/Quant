from Utils.add_features import add_accumulation_MB, add_F_MarketBreadthMB
from Utils.utils import add_pivot_Comparison_with_values
from ..strategy import strategy
from . import config
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from tqdm import tqdm
from Data.data_retrieval import get_data
import itertools
import Utils.add_features as function_lib
from pathos.multiprocessing import ProcessingPool
from Utils.Num_cores import num_cores

class MarketBreadthPivotsOverBoughtOversold(strategy):
    def __init__(self):
        strategy.__init__(self)

    def get_data(self, ticker, frequency):
        self.ticker = ticker
        self.frequency = frequency
        with open(f'Utils/{ticker}_Constituents.pkl', 'rb') as file:
            constituent_df = pickle.load(file)
        tickers = []
        for i in range(len(constituent_df)):
            tickers = tickers + constituent_df.iloc[i]["Tickers"]
        tickers = list(set(tickers))
        data = pd.DataFrame()
        tickers_selected = []
        for ticker in tqdm(tickers+[".NSEI"]):
            try:
                if ticker != ".NSEI":
                    data_today = get_data(ticker, "D").set_index("Datetime")[["Close"]].rename(columns={"Close": ticker})
                    data_today[f"{ticker}_isConstituent"] = np.nan
                    for i in range(len(data_today)):
                        if data_today.index[i] in list(constituent_df["Date"]):
                            if ticker in list(constituent_df[data_today.index[i] == constituent_df["Date"]]["Tickers"])[0]:
                                data_today.loc[data_today.index[i], f"{ticker}_isConstituent"] = 1
                    data_today.fillna(method="ffill", inplace=True)
                    tickers_selected.append(ticker)
                else:
                    data_today = get_data(ticker, "D").set_index("Datetime")
                data = pd.concat([data, data_today], axis=1)
            except Exception as e:
                pass

        data[[col for col in data.columns if col.endswith("_isConstituent")]] = data[
            [col for col in data.columns if col.endswith("_isConstituent")]].fillna(0)
        tickers = tickers_selected
        data = data[data['Close'].notna()]
        data.reset_index(inplace=True)

        self.tickers = tickers
        self.data = data
        return self.data

    def add_features(self, data, params):

        # Mday Accumulation of Nday Highs/Lows Addition
        params1 = [[data]] + [params[2], params[3]] + [[self.tickers]]
        inputs = list(itertools.product(*params1))
        pool = ProcessingPool(num_cores)
        results = pool.map(add_accumulation_MB, inputs)
        pool.clear()

        for result in results:
            data[result.columns[-1]] = result.filter(regex='[0-9]+DayAccumulation_[0-9]+dayMB')

        # F1 Addition
        params2 = [[data]] + [params[2], params[3], params[0], params[1]]
        inputs = list(itertools.product(*params2))
        pool = ProcessingPool(num_cores)
        results = pool.map(add_F_MarketBreadthMB, inputs)
        pool.clear()

        for result in results:
            try:
                data[result.columns[-1]] = result.filter(regex='.+[0-9]+_[0-9]+DayAccumulation_[0-9]+dayMB')
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
        f1 = getattr(function_lib, params[0])
        f1_lookback = params[1]
        M = params[2]
        N = params[3]
        n = params[4]

        col = f"{f1.__name__}{f1_lookback}_{M}DayAccumulation_{N}dayMB"
        df = add_pivot_Comparison_with_values(df, col, n)

        # Generate Signals
        buy_mask = (df[f"{col}_PreviousPivotValue"] < 2) & (df[f"{col}_PivotValue"] > 2) & (
                    df[f"{col}_PivotValue"] < df[f"{col}_PreviousPivotValue"])
        sell_mask = (df[f"{col}_PreviousPivotValue"] > 8) & (df[f"{col}_PivotValue"] < 8) & (
                    df[f"{col}_PivotValue"] > df[f"{col}_PreviousPivotValue"])

        bval = +1
        sval = 0  # -1 if short selling is allowed, otherwise 0

        df['signal'] = np.nan
        df.loc[buy_mask, 'signal'] = bval
        df.loc[sell_mask, 'signal'] = sval
        df.signal = df.signal.fillna(method="ffill")
        df.signal = df.signal.fillna(0)

        bval = +1
        sval = 0  #

        df['signal_bounds'] = np.nan
        df.loc[buy_mask, 'signal_bounds'] = bval
        df.loc[sell_mask, 'signal_bounds'] = sval

        df.signal_bounds = df.signal_bounds.fillna(method="ffill")
        df.signal_bounds = df.signal_bounds.fillna(0)

        df["signal"] = df.signal_bounds
        df = df.reset_index(drop=True)
        df["col"] = col

        return df[["Datetime", "signal", "col", f"{col}_IsHighPivot", f"{col}_IsLowPivot", f"{col}_PivotValue"]]


    def plotting_function(self, df_inp, save_to= None):

        df = df_inp.copy()
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

        col = df["col"].iloc[0]
        plt.plot(df['Datetime'], df[col], color='black', label=col)
        plt.plot(df[df[f"{col}_IsHighPivot"]==1]['Datetime'], df[df[f"{col}_IsHighPivot"]==1][f"{col}_PivotValue"], color='black', marker='o', ms=5, linestyle = 'None',mec='r')
        plt.plot(df[df[f"{col}_IsLowPivot"]==1]['Datetime'], df[df[f"{col}_IsLowPivot"]==1][f"{col}_PivotValue"], color='black', marker='o', ms=5, linestyle = 'None',mec='g')
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
        feature_space = [config.params_searchspace["f1"],
                         config.params_searchspace["f1_lookbacks"],
                         config.params_searchspace["M"], config.params_searchspace["N"], config.params_searchspace["n"]]

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
        return {"feature_space": feature_space, "parameter_searchspace": parameter_searchspace,
                "metric_searchspace": metric_searchspace,
                "strategy_lookbacks": strategy_lookbacks,
                "number_of_optimisation_periods": number_of_optimisation_periods,
                "recalib_periods": recalib_periods, "num_strategies": num_strategies, "metrics_opt": metrics_opt,
                "number_selected_filteredstrategies": number_selected_filteredstrategies,
                "consider_selected_strategies_over": consider_selected_strategies_over,
                "number_selected_strategies": number_selected_strategies,
                "starting_points": starting_points}








