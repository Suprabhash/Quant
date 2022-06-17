"""
Strategy parameters for InverseFisher(deprecated)
"""
from Metrics.Metrics import *
from Utils.utils import frange

params_searchspace = {
    "zscores_lookbacks": frange(48,50,1), #14,91,1
    "lb": frange(-1, 1, 1),  #0.1
    "ub": frange(-1, 1, 1)
}

metrics = [SharpeRatio]   #WinByLossRet
number_selected_strategies = 2000
number_selected_filteredstrategies = 10
strategy_lookbacks = [9,36]
number_of_optimisation_periods = [1, 2]
recalib_periods = [6]  #
num_strategies = [1, 3, 5, 7]    #
metrics_opt = [rolling_sharpe, rolling_sortino, rolling_cagr, maxdrawup_by_maxdrawdown, outperformance]
consider_selected_strategies_over = 8    #
starting_points = 10