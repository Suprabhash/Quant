"""
This file contains the exhasutive framework to perform optimization, create signals and deploy strategies
"""
import datetime
import os
import sys
from datetime import timedelta
import pickle
from os import path
#from pathos.multiprocessing import ProcessingPool
import multiprocessing
import numpy as np
import multiprocessing
from Data.data_retrieval import get_data
import pandas as pd
from matplotlib import pyplot as plt

#from Data.data_retrieval import get_data as get_Data
from Utils.utils import *
from Backtester.backtester import backtester
from Metrics.Metrics import *
from Optimisers.Optimiser import Optimiser
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from PIL import Image
import time
import Metrics.Metrics as function_lib

class Deployer():
    def __init__(self, strategy, ticker, data_frequency):
        """
        The deployer object requires the user to provide the name of the strategy. All strategy code is imported from the Folder of the same name.
        Creates a directory for the strategy caches. Each ticker has a cache folder, each of which has subdirectories based on the granlarity of the data that the strategy uses.
        :param strategy_name: Name of the strategy
        :param ticker: Ticker as on Reuters. If using a ticker from yfinance or investpy, please use the lookup table in Data.tickers.py to get the corresponding Reuters ticker
        :param data_frequency: Granularity of the data. Currently supports "D": daily and "H": Hourly
        """
        self.strategy = strategy()
        self.strategy_name = strategy.__name__
        self.ticker = ticker
        self.data_frequency = data_frequency
        self.opt_params = self.strategy.get_optimization_params()
        self.feature_space = self.opt_params["feature_space"]
        self.params_searchspace = self.opt_params["parameter_searchspace"]
        self.metrics_searchspace = self.opt_params["metric_searchspace"]
        self.lookbacks = self.opt_params["strategy_lookbacks"]
        self.number_of_optimisation_periods = self.opt_params["number_of_optimisation_periods"]
        self.recalib_periods = self.opt_params["recalib_periods"]
        self.num_strategies = self.opt_params["num_strategies"]
        self.metrics_opt = self.opt_params["metrics_opt"]
        self.number_selected_filteredstrategies = self.opt_params["number_selected_filteredstrategies"]
        self.consider_selected_strategies_over = self.opt_params["consider_selected_strategies_over"]
        self.number_selected_strategies = self.opt_params["number_selected_strategies"]
        self.starting_points = self.opt_params["starting_points"]
        if not os.path.exists(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}'):
            os.makedirs(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}')

    def get_data(self, save_cache = False, import_cache = False):
        """
        Uses the data retrieval code from Data module to import data from the datatables
        :return: Pandas dataframe of OHLCV data along with a column of datetimes for each datapoint
        """
        if not os.path.exists(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/Data'):
            os.makedirs(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/Data')

        if import_cache:
            self.strategy.initialize_ticker_frequency(self.ticker, self.data_frequency)
            with open(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/Data/OHLCV.pkl','rb') as file:
                self.data = pickle.load(file)
            self.strategy.data = self.data
        else:
            self.data =self.strategy.get_data(self.ticker, self.data_frequency)
            if save_cache:
                with open(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/Data/OHLCV.pkl', 'wb') as file:
                    pickle.dump(self.data, file)

        self.start_date = self.data.iloc[0]['Datetime']
        if save_cache:
            with open(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/Data/Start_Date.pkl','wb') as file:
                pickle.dump(self.start_date, file)

    def add_features(self, save_cache = False, import_cache = False):
        """
        Adds features relevant to the strategy. Code for adding features can be added in the the Data.add_features module
        :param func_add_feature: The function for adding features, imported from Data.add_features
        :param save_cache: Saves the OHLCV+Features dataframe, if true
        :param import_cache: skips feature creation and directly imports from the saved cache.
        :return:
        """

        if import_cache:
            with open(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/Data/Features.pkl','rb') as file:
                self.data = pickle.load(file)
            self.strategy.data=self.data
        else:
            self.data = self.strategy.add_features(self.data, self.feature_space)
            if save_cache:
                with open(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/Data/Features.pkl', 'wb') as file:
                    pickle.dump(self.data, file)

    def create_dates(self, time_unit):
        self.time_unit = interpret_time_unit(time_unit)
        self.dates = valid_dates(pd.date_range(start=str(self.data.iloc[0]['Datetime'] + timedelta(days=365)), end="2034-06-15", freq=f'{self.time_unit[0]}{self.time_unit[1]}'))

    def backtest(self, args):
        _, ec = self.strategy.do_backtest(list(args[0:-1]), start = self.data.iloc[0]["Datetime"], end = self.data.iloc[-1]["Datetime"],allocation=10000, interest_rate=6, plot=False, save_plot_to=None)
        result = {"params": args[0:-1], "equity_curve": ec[['Datetime', 'Close', 'signal', 'Return', 'S_Return']]}
        with open(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/SelectedStrategies/Backtests/{tuple(callable_functions_helper(list(result["params"]))[0])}.pkl','wb') as file:
            pickle.dump(result, file)

    def run_backtests(self, use_optimiser, parallelize=True):

        if not os.path.exists(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/SelectedStrategies/Backtests'):
            os.makedirs(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/SelectedStrategies/Backtests')

        data = self.data.copy()
        optimiser = Optimiser(method=use_optimiser)
        optimiser.define_parameter_searchspace(self.params_searchspace)
        optimiser.define_metrics_searchspace(self.metrics_searchspace)
        optimiser.define_alpha_function(self.backtest)
        optimiser.optimise(parallelize=parallelize)

    def calc_metric_from_backtest(self, args):

        data = self.data.copy()
        strategies={}
        with open(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/SelectedStrategies/Backtests/{tuple(callable_functions_helper(list(args[0:-1]))[0])}.pkl','rb') as file:
            e_curve = pickle.load(file)
        for lookback in self.lookbacks:
            strategies[lookback] = []
            for date_i in (range(len(self.dates))):
                if date_i- (int(lookback/self.time_unit[0]))>=0:
                    strategies[lookback].append({"Train Start Date": self.dates[date_i - (int(lookback / self.time_unit[0]))],"Train End Date": self.dates[date_i], "Lookback": lookback})
                    df = data.loc[(data["Datetime"] >= self.dates[date_i- (int(lookback/self.time_unit[0]))]) & (data["Datetime"] <= self.dates[date_i])].reset_index(drop=True)
                    ec = e_curve["equity_curve"]
                    ec = ec.loc[(ec["Datetime"] > df.iloc[0]["Datetime"]) & (ec["Datetime"] <= df.iloc[-1]["Datetime"])].reset_index(drop=True)
                    mask = ((ec.signal == 1) & (ec.signal.shift(1) == 0)) & (ec.signal.notnull())
                    if (ec.signal.iloc[0] == 1):
                        mask[0] = True
                    ec['trade_num'] = np.where(mask, 1, 0).cumsum()
                    strategies[lookback][date_i]["metric_val"] = getattr(function_lib, args[-1])(ec)
                    strategies[lookback][date_i]['Lookback']=lookback
                else:
                    strategies[lookback].append({"Train Start Date": None, "Train End Date": self.dates[date_i], "Strategies": None, "Lookback": lookback})
        return {"params":tuple(args[0:-1]),"strategies":strategies}

    def select_helper(self,args):
        lookback=args[0]
        strategies=self.strategies
        count = 0
        select_strategies = pd.DataFrame()
        for index, strat in strategies.iterrows():
            if count == 0:
                select_strategies["Train Start Date"] = pd.DataFrame(strat["strategies"][lookback])["Train Start Date"]
                select_strategies["Train End Date"] = pd.DataFrame(strat["strategies"][lookback])["Train End Date"]
                select_strategies[index] = pd.DataFrame(strat["strategies"][lookback])["metric_val"]
                count += 1
            else:
                select_strategies[index] = pd.DataFrame(strat["strategies"][lookback])["metric_val"]
        select_strategies["Strategies"] = ''
        for i in range(len(select_strategies.index)):
            select_strategies["Strategies"].iloc[i] = (((select_strategies.iloc[:, 3:-1].loc[i]).reset_index()).rename(
                                                        columns={'index': 'params', i: 'metric_val'})).sort_values(by="metric_val",ascending=False).reset_index(drop=True)
            select_strategies["Strategies"].iloc[i]["metric"] = self.metrics_searchspace[-1][-1]
            select_strategies["Strategies"].iloc[i]["Lookback"] = lookback
        select_strategies["Lookback"] = lookback
        select_strategies = (select_strategies[['Strategies', 'Train Start Date', 'Train End Date','Lookback']]).to_dict('records')
        return select_strategies

    def select_strategies(self, use_optimiser, parallelize=True):
        optimiser = Optimiser(method=use_optimiser)
        optimiser.define_parameter_searchspace(self.params_searchspace)
        optimiser.define_metrics_searchspace(self.metrics_searchspace)
        optimiser.define_alpha_function(self.calc_metric_from_backtest)
        strategies=(optimiser.optimise(parallelize=parallelize))

        self.strategies=(pd.DataFrame(strategies)).set_index("params")

        optimiser = Optimiser(method=use_optimiser)
        optimiser.define_parameter_searchspace([[i for i in self.lookbacks]])
        optimiser.define_alpha_function(self.select_helper)
        select_strategies = (optimiser.optimise(parallelize=parallelize))
        count=0
        for lookback in (self.lookbacks):
            with open(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/SelectedStrategies/All_{lookback}.pkl','wb') as file:
                pickle.dump(select_strategies[count], file)
            count+=1
            print(f"{lookback}_Done")

    def check_selected_strategies(self, forward_months):
        if not os.path.exists(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/CheckSelectedStrategies'):
            os.makedirs(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/CheckSelectedStrategies')

        for lookback in tqdm(self.lookbacks):
            with open(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/SelectedStrategies/All_{lookback}.pkl', 'rb') as file:
                strategies = pickle.load(file)
            for date_i in (range(len(self.dates))):
                if date_i - (int(lookback / self.time_unit[0])) >= 0:
                    _, ec = self.strategy.do_backtest(list(strategies[date_i]["Strategies"]['params'].iloc[0]), start= strategies[date_i]["Train Start Date"], end= strategies[date_i]["Train End Date"],  allocation=10000, interest_rate=6, plot=True,
                                                     save_plot_to=f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/CheckSelectedStrategies')
                    _, ec = self.strategy.do_backtest(list(strategies[date_i]["Strategies"]['params'].iloc[0]), start= strategies[date_i]["Train End Date"], end= strategies[date_i]["Train End Date"] + relativedelta(months=forward_months),  allocation=10000, interest_rate=6, plot=True,
                                                     save_plot_to=f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/CheckSelectedStrategies'+"/"+f"{str(strategies[date_i]['Train Start Date'])[:11]}"+f"{str(strategies[date_i]['Train End Date'])[:11]}"+f"{list(strategies[date_i]['Strategies']['params'].iloc[0])}"+"Test.jpg")


                    images = [Image.open(x) for x in [f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/CheckSelectedStrategies'+"/"+f"{str(strategies[date_i]['Train Start Date'])[:11]}"+f"{str(strategies[date_i]['Train End Date'])[:11]}"+f"{list(strategies[date_i]['Strategies']['params'].iloc[0])}"+"StrategyPlot.jpg",
                            f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/CheckSelectedStrategies'+"/"+f"{str(strategies[date_i]['Train Start Date'])[:11]}"+f"{str(strategies[date_i]['Train End Date'])[:11]}"+f"{list(strategies[date_i]['Strategies']['params'].iloc[0])}"+"Test.jpg"]]
                    widths, heights = zip(*(i.size for i in images))
                    total_width = sum(widths)
                    max_height = max(heights)
                    new_im = Image.new('RGB', (total_width, max_height))
                    x_offset = 0
                    for im in images:
                        new_im.paste(im, (x_offset, 0))
                        x_offset += im.size[0]

                    new_im.save(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/CheckSelectedStrategies'+"/"+f"{str(strategies[date_i]['Train Start Date'])[:11]}"+f"{str(strategies[date_i]['Train End Date'])[:11]}"+f"{list(strategies[date_i]['Strategies']['params'].iloc[0])}"+"StrategyPlot.jpg")
                    os.remove(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/CheckSelectedStrategies'+"/"+f"{str(strategies[date_i]['Train Start Date'])[:11]}"+f"{str(strategies[date_i]['Train End Date'])[:11]}"+f"{list(strategies[date_i]['Strategies']['params'].iloc[0])}"+"Test.jpg")

                    images = [Image.open(x) for x in [
                        f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/CheckSelectedStrategies' + "/" + f"{str(strategies[date_i]['Train Start Date'])[:11]}" + f"{str(strategies[date_i]['Train End Date'])[:11]}" + f"{list(strategies[date_i]['Strategies']['params'].iloc[0])}" + "Performance.jpg",
                        f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/CheckSelectedStrategies' + "/" + f"{str(strategies[date_i]['Train Start Date'])[:11]}" + f"{str(strategies[date_i]['Train End Date'])[:11]}" + f"{list(strategies[date_i]['Strategies']['params'].iloc[0])}" + "TestP.jpg"]]
                    widths, heights = zip(*(i.size for i in images))
                    total_width = sum(widths)
                    max_height = max(heights)
                    new_im = Image.new('RGB', (total_width, max_height))
                    x_offset = 0
                    for im in images:
                        new_im.paste(im, (x_offset, 0))
                        x_offset += im.size[0]

                    new_im.save(
                        f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/CheckSelectedStrategies' + "/" + f"{str(strategies[date_i]['Train Start Date'])[:11]}" + f"{str(strategies[date_i]['Train End Date'])[:11]}" + f"{list(strategies[date_i]['Strategies']['params'].iloc[0])}" + "Performance.jpg")
                    os.remove(
                        f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/CheckSelectedStrategies' + "/" + f"{str(strategies[date_i]['Train Start Date'])[:11]}" + f"{str(strategies[date_i]['Train End Date'])[:11]}" + f"{list(strategies[date_i]['Strategies']['params'].iloc[0])}" + "TestP.jpg")

    @staticmethod
    def filter_helper(itrr, start, end, all_strategies, strategies):
        for n in range(start, end):
            all_strategies[itrr]["Train Start Date"].append(strategies[n]["Train Start Date"])
            all_strategies[itrr]["Train End Date"].append(strategies[n]["Train End Date"])
            if not (isinstance(strategies[n]["Strategies"], type(None))):
                all_strategies[itrr]["Strategies"] = pd.concat(
                    [all_strategies[itrr]["Strategies"], strategies[n]["Strategies"]])
            all_strategies[itrr]["Strategies"].sort_values(by="metric_val", ascending=False, inplace=True)
            all_strategies[itrr]["Strategies"].reset_index(drop=True, inplace=True)
            all_strategies[itrr]["Lookback"].append(strategies[itrr]["Lookback"])
        return all_strategies

    def filter_strategies(self, filter_function):
        for number_of_optimisation_period in self.number_of_optimisation_periods:
            all_strategies = [{"Train Start Date": [], "Train End Date": [], "Strategies": pd.DataFrame(columns=["params", "metric", "metric_val", "Lookback"]), "Lookback": []} for i in range(len(self.dates))]
            lookbacks = self.lookbacks[:number_of_optimisation_period]
            print("Collecting metrics")
            for lookback in lookbacks:
                with open(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/SelectedStrategies/All_{lookback}.pkl', 'rb') as file:
                    strategies = pickle.load(file)

                for i in range(len(self.dates)):
                    if i >= self.consider_selected_strategies_over - 1:
                        start = i - (self.consider_selected_strategies_over - 1)
                        end = i + 1
                        all_strategies = self.filter_helper(i, start, end, all_strategies,strategies)  # static method
                    else:
                        start = 0
                        end = i + 1
                        all_strategies = self.filter_helper(i, start, end, all_strategies, strategies)  # static method

            print("Calculating sharpe of collected metrics")
            for i in tqdm(range(len(self.dates))):
                param_func_dict = {}
                for row in range(len(all_strategies[i]["Strategies"]["params"])):
                    all_strategies[i]["Strategies"]["params"].iloc[row], dict_strats = callable_functions_helper(list(all_strategies[i]["Strategies"]["params"].iloc[row]))
                    param_func_dict.update(dict_strats)
                    all_strategies[i]["Strategies"]["params"].iloc[row] = tuple(all_strategies[i]["Strategies"]["params"].iloc[row])

                df = pd.concat([all_strategies[i]["Strategies"].groupby(['params'])["metric_val"].apply(list),all_strategies[i]["Strategies"].groupby(['params'])["Lookback"].apply(list)], axis=1).reset_index()
                try:
                    all_strategies[i]["Strategies"] = df
                    for row in range(len(all_strategies[i]["Strategies"]["params"])):
                        all_strategies[i]["Strategies"]["params"].iloc[row] = list(all_strategies[i]["Strategies"]["params"].iloc[row])
                        for num in range(len(all_strategies[i]["Strategies"]["params"].iloc[row])):
                            try:
                                if callable(param_func_dict[all_strategies[i]["Strategies"]["params"].iloc[row][num]]):
                                    all_strategies[i]["Strategies"]["params"].iloc[row][num] = param_func_dict[all_strategies[i]["Strategies"]["params"].iloc[row][num]]
                            except:
                                pass
                        all_strategies[i]["Strategies"]["params"].iloc[row] = tuple(all_strategies[i]["Strategies"]["params"].iloc[row])


                    if len(df[["metric_val"]].reset_index(drop=True).iloc[0].to_list()[0])==1:
                        all_strategies[i]["Strategies"]["metric"] = self.metrics_searchspace[0][0].__name__
                        all_strategies[i]["Strategies"]["metric_val"] = all_strategies[i]["Strategies"]["metric_val"].apply(lambda x: x[0])
                        all_strategies[i]["Strategies"].replace([np.inf, -np.inf], np.nan, inplace=True)
                    else:
                        if len(df[["metric_val"]].reset_index(drop=True).iloc[0].to_list()[0])>=8:
                            all_strategies[i]["Strategies"]["metric"] = "Sharpe of "+self.metrics_searchspace[0][0].__name__
                            all_strategies[i]["Strategies"]["metric_val"] = all_strategies[i]["Strategies"]["metric_val"].apply(lambda x: np.mean(x) / np.std(x))
                            all_strategies[i]["Strategies"].replace([np.inf, -np.inf], np.nan, inplace=True)
                        else:
                            all_strategies[i]["Strategies"]["metric"] = "Median of " + self.metrics_searchspace[0][0].__name__
                            all_strategies[i]["Strategies"]["metric_val"] = all_strategies[i]["Strategies"]["metric_val"].apply(lambda x: np.median(x))
                        all_strategies[i]["Strategies"].dropna(inplace=True)
                        all_strategies[i]["Strategies"] = all_strategies[i]["Strategies"].sort_values(by="metric_val", ascending=False)
                    try:
                        all_strategies[i]["Strategies"] = all_strategies[i]["Strategies"].iloc[:self.number_selected_strategies].reset_index(drop=True)
                    except Exception as e:
                        # print(e)
                        pass
                except Exception as e:
                    # print(e)
                    continue

            selected_strategies = [{"Strategies Selected from Data from Date": None, "Strategies Selected from Data till Date": None, "Train Start Date": None, "Train End Date": None, "Strategies": pd.DataFrame(columns=["params", "metric", "metric_val", "Lookback"]), "Lookback": min(lookbacks)} for i in range(len(self.dates))]
            print("Using filter")
            for i in tqdm(range(len(self.dates))):
                try:
                    selected_strategies[i]["Strategies Selected from Data from Date"] = (min([i for i in all_strategies[i]["Train Start Date"] if i]))
                    selected_strategies[i]["Strategies Selected from Data till Date"] = (max([i for i in all_strategies[i]["Train End Date"] if i]))
                    selected_strategies[i]["Train Start Date"] = (max([i for i in all_strategies[i]["Train Start Date"] if i]))
                    selected_strategies[i]["Train End Date"] = (max([i for i in all_strategies[i]["Train End Date"] if i]))
                    selected_strategies[i]["Strategies"] = filter_function(all_strategies[i]["Strategies"], self.strategy, self.strategy_name,self.number_selected_filteredstrategies, start = selected_strategies[i]["Train Start Date"], end = selected_strategies[i]["Train End Date"])
                except Exception as e:
                    # print(e)
                    selected_strategies[i]["Train Start Date"] = None
                    selected_strategies[i]["Strategies"] = None
                    selected_strategies[i]["Train End Date"] = (max([i for i in all_strategies[i]["Train End Date"] if i]))

            with open(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/SelectedStrategies/Filtered_Top{self.number_selected_filteredstrategies}_NumOptPeriods{number_of_optimisation_period}.pkl', 'wb') as file:
                pickle.dump(selected_strategies, file)

    @staticmethod
    def get_weights(df, strategies,strategy,strategy_name, num_strategies, metric, data_frequency, starting_points):

        if len(strategies) > num_strategies:
            selected_strategies = strategies[:num_strategies]
        else:
            selected_strategies = strategies

        for i in range(len(selected_strategies)):
            with open(f'Caches/{strategy.ticker}/{strategy.frequency}/{strategy_name}/SelectedStrategies/Backtests/{tuple(callable_functions_helper(list(selected_strategies.iloc[i]["params"]))[0])}.pkl','rb') as file:
                signal = pickle.load(file)
            #signal = backtests["Strategies"][tuple(strategies.iloc[i]["params"])]
            signal=signal["equity_curve"]
            signal = signal.loc[(signal["Datetime"] > df.iloc[0]["Datetime"]) & (signal["Datetime"] <= df.iloc[-1]["Datetime"])].reset_index(drop=True)

            if i == 0:
                signals = signal['signal'].to_frame().rename(columns={'signal': f'Strategy{i + 1}'}).set_index(
                    signal["Datetime"])
            else:
                signals = pd.merge(signals, (
                    signal['signal'].to_frame().rename(columns={'signal': f'Strategy{i + 1}'}).set_index(
                        signal["Datetime"])), left_index=True, right_index=True)

        def alpha(*args):
            weights = []
            for weight in args:
                weights.append(weight)
            weights = pd.DataFrame(weights)
            weights = weights / weights.sum()
            signal_final = pd.DataFrame(np.dot(np.where(np.isnan(signals), 0, signals), weights))
            signal_final = pd.DataFrame(np.where(signal_final > 0.5, 1, 0)).set_index(signals.index).rename(
                columns={0: 'signal'})
            inp = pd.merge(df.set_index(df["Datetime"]).drop(columns=["Datetime"]).astype(float),
                           signal_final[["signal"]].astype(float), left_index=True, right_index=True)
            inp = inp.reset_index()
            strat = backtester(data=inp, strategy=None)
            ec = strat.signal_performance(10000, 6, data_frequency).dropna()
            return ec

        def prior(params):
            weights = params
            weights = [weight / sum(weights) for weight in weights]
            # for weight in weights:
            #     if weight > 0.3:
            #         return 0
            return 1

        if len(selected_strategies) > 1:
            opt = Optimiser(method="MCMC")
            opt.define_alpha_function(alpha)
            opt.define_optim_function(metric)
            opt.define_prior(prior)
            guesses = [(list(np.random.dirichlet(np.ones(len(selected_strategies))))) for i in range(starting_points)]
            if len(guesses) == 1:
                guesses = guesses[0]
            opt.define_guess(guess=guesses)
            opt.define_iterations(200)
            opt.define_lower_and_upper_limit(0, 1)
            opt.optimise(parallelize=True)
            res = opt.return_results()
            weights = pd.DataFrame(res.iloc[0]["params"])
            weights = weights / weights.sum(axis=0)
        else:
            weights = pd.DataFrame([1])

        selected_strategies["weights"] = weights
        return selected_strategies

    def optimize_weights(self):
        if not os.path.exists(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/equity_curves'):
            os.makedirs(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/equity_curves')

        if not os.path.exists(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/csv_files'):
            os.makedirs(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/csv_files')

        if not os.path.exists(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/weighted_strategies'):
            os.makedirs(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/weighted_strategies')

        for number_of_optimization_period in self.number_of_optimisation_periods:
            for recalib_periods in self.recalib_periods:
                for num_strategies in self.num_strategies:
                    for metric in self.metrics_opt:

                        if path.exists(f"Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/weighted_strategies\Results_LP{number_of_optimization_period}_Recal{recalib_periods}_NS{num_strategies}_M{metric.__name__}.pkl"):
                            print("Already processed")
                            continue

                        with open(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/SelectedStrategies/Filtered_Top{self.number_selected_filteredstrategies}_NumOptPeriods{number_of_optimization_period}.pkl','rb') as file:
                            strategies = pickle.load(file)

                        selected_strategies = []
                        for date_i in range(len(self.dates) - (int(recalib_periods / self.time_unit[0]))):
                            if (self.time_unit[0] * date_i) % recalib_periods == 0:
                                strategies[date_i]["Test End Date"] = strategies[date_i+int(recalib_periods/self.time_unit[0])]["Train End Date"]
                                selected_strategies.append(strategies[date_i])

                        for i, strategy in enumerate(selected_strategies):
                            try:
                                df = self.data[(self.data["Datetime"]>=selected_strategies[i]["Train Start Date"])&(self.data["Datetime"]<=selected_strategies[i]["Train End Date"])]
                                selected_strategies[i]["Strategies"] = self.get_weights(df,selected_strategies[i]["Strategies"],self.strategy, self.strategy_name, num_strategies, metric, self.data_frequency, self.starting_points)
                            except Exception as e:
                                # print(e)
                                pass

                        with open(f"Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/weighted_strategies\Results_LP{number_of_optimization_period}_Recal{recalib_periods}_NS{num_strategies}_M{metric.__name__}.pkl",'wb') as file:
                            pickle.dump(selected_strategies, file)


    def backtest_weighted_strategy(self, args):
        number_of_optimization_period = args[0]
        recalib_periods = args[1]
        num_strategies = args[2]
        metric = args[3]

        with open(f"Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/weighted_strategies\Results_LP{number_of_optimization_period}_Recal{recalib_periods}_NS{num_strategies}_M{metric}.pkl",'rb') as file:
            selected_strategies = pickle.load(file)

        inp_all = pd.DataFrame()
        for i, strategy in enumerate(selected_strategies):
            try:
                df = self.data[(self.data["Datetime"]>=selected_strategies[i]["Train End Date"])&(self.data["Datetime"]<=selected_strategies[i]["Test End Date"])]
                strategies = strategy["Strategies"]

                for j in range(len(strategies)):

                    with open(f'Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/SelectedStrategies/Backtests/{tuple(callable_functions_helper(list(strategies.iloc[j]["params"]))[0])}.pkl','rb') as file:
                        signal = pickle.load(file)
                    signal = signal["equity_curve"]
                    signal = signal.loc[(signal["Datetime"] > selected_strategies[i]["Train End Date"]) & (
                                signal["Datetime"] <= selected_strategies[i]["Test End Date"])].reset_index(drop=True)
                    if j == 0:
                        signals = signal['signal'].to_frame().rename(columns={'signal': f'Strategy{j + 1}'}).set_index(signal["Datetime"])
                    else:
                        signals = pd.merge(signals, (signal['signal'].to_frame().rename(columns={'signal': f'Strategy{j + 1}'}).set_index(signal["Datetime"])),left_index=True, right_index=True)

                signal_final = pd.DataFrame(np.dot(np.where(np.isnan(signals), 0, signals), pd.DataFrame(list(strategies["weights"]))))
                signal_final = pd.DataFrame(np.where(signal_final > 0.5, 1, 0)).set_index(signals.index).rename(columns={0: 'signal'})
                inp = pd.merge(df.set_index(df["Datetime"]).drop(columns=["Datetime"]).astype(float), signal_final[["signal"]].astype(float), left_index=True, right_index=True)
                inp_all = pd.concat([inp_all, inp])
            except Exception as e:
                # print(e)
                pass

        inp_all = inp_all[["Close", "signal"]].reset_index().fillna(method="ffill")
        strat = backtester(data = inp_all, strategy=None)
        ec = strat.signal_performance(10000, 6, self.data_frequency).dropna()
        strat.plot_performance(self.data_frequency, save_to=f"Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/equity_curves\Results_LP{number_of_optimization_period}_Recal{recalib_periods}_NS{num_strategies}_M{metric}.jpg")
        ec.to_csv(f"Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/csv_files\Results_LP{number_of_optimization_period}_Recal{recalib_periods}_NS{num_strategies}_M{metric}.csv")

    def select_best_and_mail_results(self, list):
        optimiser = Optimiser(method="BruteForce")
        optimiser.define_parameter_searchspace([self.number_of_optimisation_periods, self.recalib_periods, self.num_strategies, self.metrics_opt])
        optimiser.define_alpha_function(self.backtest_weighted_strategy)
        optimiser.optimise(parallelize=True)
        res = []
        for number_of_optimization_period in self.number_of_optimisation_periods:
            for recalib_periods in self.recalib_periods:
                for num_strategies in self.num_strategies:
                    for metric in self.metrics_opt:
                        try:
                            temp_res = pd.read_csv(
                                f"Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/csv_files/Results_LP{number_of_optimization_period}_Recal{recalib_periods}_NS{num_strategies}_M{metric}.csv",
                                parse_dates=True)
                            res.append({'Ticker': self.ticker, "Optimization Periods": number_of_optimization_period,
                                        "Recalibration Months": recalib_periods, "Number of Strategies": num_strategies,
                                        "Metric": metric,
                                        "Sharpe": SharpeRatio(temp_res) if self.data_frequency == 'D' else SharpeRatio_hourly(temp_res),
                                        "MaxDrawupByMaxDrawdown": maxdrawup_by_maxdrawdown(temp_res),
                                        "Outperformance": outperformance(temp_res),
                                        "Total_Ret": total_return(temp_res),
                                        "Median_1yr_Ret": rolling_yearly_return_median(temp_res)})
                        except Exception as e:
                            print(f"Not processed: {e}")
        pd.DataFrame(res).to_csv(f"Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/csv_files/Results_Parametric.csv")

        # Emailer for top3 strategies
        for sortby in ["Total_Ret", "Median_1yr_Ret", "Outperformance", "Sharpe", "MaxDrawupByMaxDrawdown"]:  # "Outperformance", "Sharpe", "MaxDrawupByMaxDrawdown"
            res_sorted = pd.DataFrame(res).sort_values(sortby, ascending=False)
            topn = 3
            if len(res_sorted)<3:
                topn = len(res_sorted)
            for i in range(topn):  
                number_of_optimization_period = res_sorted.iloc[i]["Optimization Periods"]
                recalib_periods = res_sorted.iloc[i]["Recalibration Months"]
                num_strategies = res_sorted.iloc[i]["Number of Strategies"]
                metric = res_sorted.iloc[i]["Metric"]
                temp_res = pd.read_csv(
                    f"Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/csv_files\Results_LP{number_of_optimization_period}_Recal{recalib_periods}_NS{num_strategies}_M{metric}.csv",
                                parse_dates=True)
                temp_res['Datetime'] = pd.to_datetime(temp_res['Datetime'])
                plt.plot(temp_res['Datetime'], temp_res['Market_Return'], color='black', label='Market Returns')
                plt.plot(temp_res['Datetime'], temp_res['Strategy_Return'], color='blue', label='Strategy Returns')
                plt.title('Strategy Backtest')
                plt.legend(loc=0)
                plt.tight_layout()
                plt.savefig(
                    f"Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}/SortedBy_{sortby}_{(i + 1)}_Results_Ticker{self.ticker}_LP{number_of_optimization_period}_Recal{recalib_periods}_NS{num_strategies}_M{metric}.jpg")
                plt.clf()
            path_mail = f"Caches/{self.ticker}/{self.data_frequency}/{self.strategy_name}"
            files = os.listdir(path_mail)
            images = []
            for file in files:
                if ((file.startswith(f"SortedBy_{sortby}")) & (file.endswith('.jpg'))):
                    img_path = path_mail + '/' + file
                    images.append(img_path)
            emails = []
            if "Suprabhash" in list:
                emails.append("suprabhashsahu@acsysindia.com")
            if "Aditya" in list:
                emails.append("aditya@shankar.biz")
            if "divakarank" in list:
                emails.append("divakarank@acsysindia.com")
            if "Sandesh" in list:
                emails.append("saisandesh@acsysindia.com")
            SendMail(emails, self.strategy_name, self.ticker, sortby, images)


















