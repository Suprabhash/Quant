"""
Strategy parameters for Alpha
"""
from Metrics.Metrics import *
from Utils.utils import frange

params_searchspace = {
    "lookbacks": frange(70, 80, 10),  #50, 400, 20
    "lb": frange(-1.0,1.0,1),  #-7,7,0.25
    "ub": frange(-1.0,1.0,1)
}
fisher_lookbacks = [f_look for f_look in range(70, 80, 10)]

metrics = [WinByLossRet]   #WinByLossRet
number_selected_strategies = 2000
number_selected_filteredstrategies = 10
strategy_lookbacks =[9] # 9,36
number_of_optimisation_periods = [1]#,2
recalib_periods = [6]  #
num_strategies = [1]    #,3,5,7
metrics_opt = [rolling_sharpe] #, rolling_sortino, rolling_cagr, maxdrawup_by_maxdrawdown, outperformance
consider_selected_strategies_over = 1#
starting_points = 10