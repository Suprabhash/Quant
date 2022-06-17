"""
Strategy parameters for RescaledRange
"""
from Metrics.Metrics import *
from Utils.utils import frange

params_searchspace = {
                    "hurst_range": [100,150,200,250,300,350,400,450,500],                       #100,150,200,250,300,350,400,450,500
                    "MA_range" : [5,10,20,30,50],               #5,10,20,30,50
                    "ROC_range" : [1,2,3,4,5,10,15,20,30,50],   #1,2,3,4,5,10,15,20,30,50
                    "levels" : frange(-0.028, 0.032, 0.001)    #frange(-0.028, 0.032, 0.001)
                    }

metrics = [SharpeRatio]   #WinByLossRet
number_selected_strategies = 2000
number_selected_filteredstrategies = 10
strategy_lookbacks = [9,36]
number_of_optimisation_periods = [1, 2]
recalib_periods = [6]  #
num_strategies = [1, 3, 5, 7]    #, 3, 5, 7
metrics_opt = [rolling_sharpe, rolling_sortino, rolling_cagr, maxdrawup_by_maxdrawdown, outperformance] #, rolling_sortino, rolling_cagr, maxdrawup_by_maxdrawdown, outperformance
consider_selected_strategies_over = 1    #
starting_points = 10

