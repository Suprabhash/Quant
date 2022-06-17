"""
Strategy parameters for Alpha
"""
from Metrics.Metrics import *

params_searchspace = {
    "lookbacks": [i for i in range(14, 20, 2)],
    "ub": [i for i in range(int(70), int(90), int(5))],
    "lb": [i for i in range(int(10), int(40), int(5))]
}

metrics = [WinByLossRet]
number_selected_filteredstrategies = 10
RSI_lookbacks = [RSI_lookback for RSI_lookback in range(14, 20, 2)]
strategy_lookbacks = [24, 48, 96]
number_of_optimisation_periods = [3]
recalib_periods = [6]
num_strategies = [5]
metrics_opt = [rolling_sharpe] #, rolling_sortino, rolling_cagr, maxdrawup_by_maxdrawdown, outperformance]
consider_selected_strategies_over = 8    #Considering all strategies over 2 years - 8 recalibration periods
metric_threshold = 10