from ..strategy import strategy
from Strategies.VolumeSVR.config import *
from Utils.add_features import add_fisher
import numpy as np
from . import config
import matplotlib.pyplot as plt
from Utils.utils import get_data_ETH_minute, resample_data
import os
import pickle
from Utils.add_features import return_volume_features_minute_hourly, prepare_volume_features, add_percentile_of_forward_returns
from sklearn.svm import SVR

class VolumeSVR(strategy):
    def __init__(self):
        strategy.__init__(self)

    def get_data(self, ticker, frequency):
        self.ticker = ticker
        self.frequency = frequency

        df_minute = get_data_ETH_minute(path=f'Caches/{self.ticker}/{self.frequency}/{__class__.__name__}/Data/MinuteOHLCV.pkl')
        self.data = resample_data(df_minute, 60)

        return self.data

    def add_features(self, data, params):

        if self.ticker != "ETH=BTSP":
            print("Strategy currently works with ETH only")

        # To be replaced by Azure Data Tables Later
        with open(f'Data/ETH_Temp/ETH_VolumeLevels_Hourly.pkl', 'rb') as file:
            vol_feat = pickle.load(file)

        print("Creating Volume Features")
        #Create volume features
        all_feat = prepare_volume_features(data, vol_feat)
        all_feat = add_percentile_of_forward_returns(all_feat, params[0], params[1], freturn=params[2][0])

        print("Scaling Features")
        #Scale features
        cols = [col for col in list(all_feat.columns) if ("FReturn" not in col) if (col != "Datetime")]
        scaled_all_feat = params[4][0](all_feat, params[1], cols)

        print("Selecting features")
        #Select features
        cols_inp = [col for col in list(scaled_all_feat.columns) if ("FReturn" not in col) if (col != "Datetime")]
        cols_out = [col for col in list(scaled_all_feat.columns) if ("FReturn" in col)]
        selected_features, _, _ = params[5][0](scaled_all_feat, cols_inp, cols_out, params[3][0])
        #Need to add more checks here
        scaled_selected_feat = scaled_all_feat[selected_features+cols_out+["Datetime"]]

        with open(f'Caches/{self.ticker}/{self.frequency}/{__class__.__name__}/Data/Scaled_SelectedFeatures.pkl','wb') as file:
            pickle.dump(selected_features, file)

        return scaled_selected_feat

    def create_signals(self, df_input, *params):
        """
            Creates signals based on fisher values and its defined thresholds. A buy signal is created when fisher crosses over lower bound from bottom to top
            A sell signal is created when fisher crosses over upper bound from top to bottom
            :param df: Receives the OHLCV data frame with a column for Datetime and Fisher values in columns of the format f"Fisher{Lookback_value}"
            :param params: A list of the lookback, lower bound and upper bound
            :return: Returns a dataframe consisting of the Datetime and the signal
            """


        [kernel, gamma, C, epsilon, return_lookforward, percentile_lookback, lb, ub] = params[0]


        with open(f'Caches/{self.ticker}/{self.frequency}/{__class__.__name__}/Data/Scaled_SelectedFeatures.pkl','rb') as file:
            cols = pickle.load(file)

        dates = df_input[['Datetime']]
        unscaled = df_input[cols]
        df_input.drop(columns=["Datetime"] + cols, inplace=True)

        vol_feats = df_input[[col for col in list(df_input.columns) if "FReturn" not in col]]
        vol_metrics = df_input[[col for col in list(df_input.columns) if "FReturn" in col]]

        regressor = SVR(kernel=kernel, gamma=gamma, C=C, epsilon=epsilon)
        regressor.fit(vol_feats[[col for col in vol_feats.columns if f"percentile_over_{percentile_lookback}" in col]].to_numpy(), vol_metrics[f"{return_lookforward}FReturn_percentile_over_{percentile_lookback}"].to_numpy())

        y_pred = regressor.predict(vol_feats[[col for col in vol_feats.columns if f"percentile_over_{percentile_lookback}" in col]].to_numpy())

        backtest = unscaled.copy()
        backtest["Predicted_Percentile"] = pd.DataFrame(y_pred)

        backtest["lb"] = lb
        backtest["ub"] = ub
        buy_mask = (backtest["Predicted_Percentile"] >= backtest["ub"])
        sell_mask = (backtest["Predicted_Percentile"] <= backtest["lb"])
        backtest["Datetime"] = dates
        bval = +1
        sval = 0

        backtest['signal'] = np.nan
        backtest.loc[buy_mask, 'signal'] = bval
        backtest.loc[sell_mask, 'signal'] = sval

        backtest.signal = backtest.signal.fillna(method="ffill")
        backtest.signal = backtest.signal.fillna(0)
        return backtest[["Datetime", "signal", "lb", "ub", "Actual", "Predicted"]]

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

        plt.plot(df['Datetime'], df['Actual'], color='black', label='Actual')
        plt.plot(df['Datetime'], df['Predicted'], color='blue', label='Predicted')
        plt.plot(df['Datetime'], df['lb'], color='green', label='Lower Bound')
        plt.plot(df['Datetime'], df['ub'], color='red', label='Upper Bound')

        plt.plot(df.loc[buy_plot_mask]['Datetime'].shift(1), df.loc[buy_plot_mask]['Predicted'].shift(1), r'^',
                 ms=15,
                 label="Entry Signal", color='green', markeredgecolor='k', markeredgewidth=1)
        plt.plot(df.loc[sell_plot_mask]['Datetime'].shift(1), df.loc[sell_plot_mask]['Predicted'].shift(1), r'^',
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

    @staticmethod
    def get_optimization_params():
        feature_space = [config.params_searchspace["return_lookforward"], config.params_searchspace["percentile_lookbacks"],
                         [config.percentile_type_for_freturns], [config.feature_selection_threshold], [config.scaler], [config.feature_selecter]]
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




