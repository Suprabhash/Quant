"""
Strategy parameters for InverseFisher
"""
from Metrics.Metrics import *
from Utils.add_features import max_over_lookback, min_over_lookback, sma, x
from Utils.utils import frange


params_searchspace = {
    "f1": [x, max_over_lookback, min_over_lookback, sma], #x, max_over_lookback, min_over_lookback, sma
    "f2": [x, max_over_lookback, min_over_lookback, sma],
    "zscores_lookbacks": frange(100, 400, 50),   #50, 410, 20
    "f1_lookbacks": frange(1,11,3) + [21,30,60,250],#1,11,1+ [21,30,60,250],
    "f2_lookbacks": frange(1,11,3) + [21,30,60,250],
    "b1": frange(-7.0,7.0,0.5),  # (-7.0,7.0,0.25
    "b2": frange(-7.0,7.0,0.5)
}

metrics = [SharpeRatio]
number_selected_strategies = 2000
number_selected_filteredstrategies = 10
strategy_lookbacks = [9,36]
number_of_optimisation_periods = [1,2] #, 2
recalib_periods = [6]
num_strategies = [1] #,3,5,7
metrics_opt = [rolling_sharpe]   #, rolling_sortino, rolling_cagr, maxdrawup_by_maxdrawdown, outperformance
consider_selected_strategies_over = 1    #
starting_points = 10