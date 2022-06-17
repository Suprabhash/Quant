import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})


from Data.data_retrieval import get_data
from hurst import *
from Utils.utils import frange
from Strategies.RescaledRange.RescaledRange import RescaledRange
from Deployer.Deployer import Deployer
from Metrics.Metrics import SharpeRatio
from Utils.utils import correlation_filter

if __name__ == "__main__":
    deployer = Deployer(strategy = RescaledRange, ticker = ".NSEI", data_frequency = 'D')

    print("Getting data")
    deployer.get_data()

    print("Creating Dates")
    deployer.create_dates("3_Months")

    print("Adding features")
    deployer.add_features()

    # print("Running backtests")
    # deployer.run_backtests(use_optimiser="BruteForce")
    #
    # print("Selecting Strategies")
    # deployer.select_strategies(use_optimiser = "BruteForce", parallelize=True)
    #
    print("Checking Selected Strategies")
    deployer.check_selected_strategies(forward_months = 6)
    #
    # print("Filtering Strategies")
    # deployer.filter_strategies(filter_function = correlation_filter)
    #
    # print("Optimizing weights")
    # deployer.optimize_weights()

    # print("Selecting best and mailing results")
    # deployer.select_best_and_mail_results(list = ["Suprabhash", "Aditya"])
