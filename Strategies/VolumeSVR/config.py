"""
Strategy parameters for VolumeSVR
"""
from Metrics.Metrics import *
from Utils.utils import frange, rolling_percentile_parallelized, NMI_feature_selection

params_searchspace = {
    "kernel": ["rbf"],
    "gamma": [0.0001, 0.005],   #frange(0.0001, 1, 0.0001)
    "C": [0.1, 1, 10],  #frange(0.1, 1000, 0.01)
    "epsilon": [0.05, 0.1],  #frange(0, 1, 0.01)
    "return_lookforward": frange(5, 10, 5),    #frange(5, 50, 5)
    "percentile_lookbacks": frange(150, 175, 25),   #frange(150, 400, 25)
    "lb": [0.4], #frange( 1, 49, 1)
    "ub": [0.6], #frange(51, 99, 1)
}

metrics = [outperformance]
number_selected_strategies = 2000
number_selected_filteredstrategies = 1
percentile_type_for_freturns = "simple"
strategy_lookbacks = [24] #, 48, 96
number_of_optimisation_periods = [1]
recalib_periods = [3]  #3,6,12
num_strategies = [1]    #1,3,5,7
metrics_opt = [rolling_sharpe]  #, rolling_sortino, rolling_cagr, maxdrawup_by_maxdrawdown, outperformance
consider_selected_strategies_over = 1    #Considering all strategies over 2 years - 8 recalibration periods
scaler = rolling_percentile_parallelized
feature_selecter = NMI_feature_selection
feature_selection_threshold = 0
starting_points = 10
