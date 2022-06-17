"""
Strategy parameters for InverseFisher
"""
from Metrics.Metrics import *
from Utils.add_features import max_over_lookback, min_over_lookback, sma, Fisher, Stochastic
from Utils.utils import frange


params_searchspace = {
    "f1": [Stochastic, Fisher],
    "f1_lookbacks": [15,30,90,250,500],
    "M": [1000],
    "N": [15,30,90,250,500],
    "n": [1,2,3,4,5,6,7]

}

metrics = [SharpeRatio]   #WinByLossRet
number_selected_strategies = 2000
number_selected_filteredstrategies = 10
strategy_lookbacks = [9,36]
number_of_optimisation_periods = [1, 2]
recalib_periods = [6]  #
num_strategies = [1, 3, 5, 7]    #
metrics_opt = [rolling_sharpe, rolling_sortino, rolling_cagr, maxdrawup_by_maxdrawdown, outperformance]
consider_selected_strategies_over = 1   #
starting_points = 10