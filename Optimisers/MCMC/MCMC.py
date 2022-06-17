import sys

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from functools import partial

from sklearn.metrics import silhouette_score
import vix_utils
#import pandas_datareader as pdr
from datetime import datetime
from datetime import timedelta
from Optimisers.MCMC.DataframeManipulator import DataframeManipulator
from Optimisers.MCMC.Misc import Misc
from Optimisers.MCMC.Indicator import *
from scipy.stats import percentileofscore
from Optimisers.MCMC.Metrics import *
import numpy as np
from numba import njit
from numba.typed import List
# from fast_histogram import histogram1d, histogram2d
from Optimisers.MCMC import config as mcmc_config

# define an example,
# define a method to define alpha functions bound with dataframe


class MCMC():

    def __init__(self, alpha_fn, alpha_fn_params_0, target, num_iters, prior, burn = 0.00, optimize_fn = None, lower_limit = -10000,
                 upper_limit = 10000):
        self.alpha_fn = alpha_fn
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.initial_params = alpha_fn_params_0
        self.target = target
        if optimize_fn is not None:
            self.optimize_fn = optimize_fn
        else:
            self.optimize_fn = MCMC.nmi
        self.num_iters = num_iters
        self.burn = burn
        self.prior = prior
        # The below 2 attributes are read from the config file. These serve as conditions for the 'next' guess
        self.min_max_zip = list(zip(mcmc_config.min_max_lookup.keys(), mcmc_config.min_max_lookup.values()))
        self.sum_zip = list(zip(mcmc_config.sum_lookup.keys(), mcmc_config.sum_lookup.values()))

    def transition_fn(self, cur, iter):
        std_guesses = self.std_guess(iter, List(self.initial_params), self.num_iters)
        if self.sum_zip == []:
            sum_zip = None
        else:
            sum_zip = List(self.sum_zip)
        return self.transition_np(List(cur), List(self.min_max_zip), sum_zip, List(std_guesses))

    @staticmethod
    @njit
    def transition_np(cur, min_max_zip, sum_zip, std):
        if sum_zip == None:
            new_guesses = []
            for index_config in min_max_zip:
                idx = index_config[0]
                lb = index_config[1][0]
                ub = index_config[1][1]
                loop = True
                while loop:
                    new_guess = np.random.normal(cur[idx], std[idx], (1,))
                    if new_guess[0] >= lb and new_guess[0] <= ub:
                        new_guesses.append(new_guess[0])
                        loop = False
            return new_guesses
        else:
            new_guesses = [-1.0 for i in range(len(min_max_zip))]
            for key_zip in sum_zip:
                done = False
                while not done:
                    key_sum = key_zip[1]
                    for idx in key_zip[0]:
                        lb = min_max_zip[idx][1][0]
                        ub = min_max_zip[idx][1][1]
                        if key_zip[0].index(idx) == len(key_zip[0]) - 1:
                            if key_sum >= lb and key_sum <= ub:
                                # print('Filling last index')
                                new_guesses[idx] = key_sum
                                done = True
                            else:
                                # print('Failed to satisfy the last element\'s condtn')
                                pass
                            break

                        loop = True
                        while loop:
                            new_guess = np.random.normal(cur[idx], std[idx], (1,))
                            if new_guess[0] >= lb and new_guess[0] <= ub:
                                new_guesses[idx] = new_guess[0]
                                key_sum = key_sum - new_guess[0]
                                loop = False
                            else:
                                continue
            return new_guesses

    @staticmethod
    @njit
    def std_guess(itr, cur_guesses, num_iters):
        stds = []
        for guess in cur_guesses:
            num_digits = len(str(round(guess)))
            std = np.power(10, float(num_digits - 2))
            if itr < 0.5 * num_iters:
                std_factor = 2
            elif itr < 0.65 * num_iters:
                std_factor = 1
            elif itr < 0.85 * num_iters:
                std_factor = 0.75
            elif itr < 0.95 * num_iters:
                std_factor = 0.5
            elif itr < 0.99 * num_iters:
                std_factor = 0.1
            elif itr < num_iters:
                std_factor = 0.01
            # std_factor = 0.1
            stds.append(std * std_factor)
        return stds

    @staticmethod
    def __to_percentile( arr ):
        pct_arr = []
        for idx in range( 0, len(arr)):
            pct_arr.append( round( percentileofscore( np.array( arr ), arr[ idx ]  ) ) )
        return pct_arr

    @staticmethod
    def __shan_entropy(c):
        c_normalized = c / float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = -sum(c_normalized * np.log2(c_normalized))
        return H

    @staticmethod
    def nmi( X, Y, bins ):
        # if np.percentile(np.abs(X), 5) < (np.percentile(np.abs(X), 95)/25):
        #     print('Lognormaling')
        #     X = X - min(X) + 1
        #     X = np.log(X)
        # if np.percentile(np.abs(Y), 5) < (np.percentile(np.abs(Y), 95)/25):
        #     print('Lognormaling')
        #     Y = Y - min(Y) + 1
        #     Y = np.log(Y)
        bin_sizes = [20, 40, 80, 160]
        NMIs = []
        for bins in bin_sizes:
            c_XY = np.histogram2d(X, Y, bins)[0]
            c_X = np.histogram(X, bins)[0]
            c_Y = np.histogram(Y, bins)[0]

            H_X = MCMC.__shan_entropy(c_X)
            H_Y = MCMC.__shan_entropy(c_Y)
            H_XY = MCMC.__shan_entropy(c_XY)

            NMI = 2*(H_X + H_Y - H_XY)/(H_X+H_Y)
            NMIs.append(NMI)
        return max(NMIs)

    # @staticmethod
    # @njit
    # def nmi(X, Y, bins):
    #     bins = min(30, bins)
    #     h_X = np.histogram(X, bins)
    #     h_Y = np.histogram(Y, bins)
    #
    #     # Clustering according to the histogram's indices NOTE: h_X[1] will have number of elements = bin size + 1
    #     c_X = np.array([-1] * len(X))
    #     for i in range(len(X)):
    #         for j in range(len(h_X[1]) - 1):
    #             if X[i] >= h_X[1][j] and X[i] < h_X[1][j + 1]:
    #                 c_X[i] = int(list(h_X[1]).index(h_X[1][j])) + 1
    #             elif j == len(h_X[1]) - 2 and X[i] >= h_X[1][j + 1]:
    #                 c_X[i] = int(list(h_X[1]).index(h_X[1][j])) + 1
    #             else:
    #                 continue
    #
    #     c_Y = np.array([-1] * len(Y))
    #     for i in range(len(Y)):
    #         for j in range(len(h_Y[1]) - 1):
    #             if Y[i] >= h_Y[1][j] and Y[i] < h_Y[1][j + 1]:
    #                 c_Y[i] = int(list(h_Y[1]).index(h_Y[1][j])) + 1
    #             elif j == len(h_Y[1]) - 2 and Y[i] >= h_Y[1][j + 1]:
    #                 c_Y[i] = int(list(h_Y[1]).index(h_Y[1][j])) + 1
    #             else:
    #                 continue
    #
    #     corr = np.corrcoef(c_X, c_Y)[0][1]
    #     if np.isnan(corr):
    #         corr = 0
    #     return corr

    def do_step(self, iter, prev_params, prev_nmi ):

        #print("Inside do_step")

        next_params = self.transition_fn( prev_params, iter )

        if self.prior( next_params ) != 0:

            #y_pred = MCMC.__to_percentile( self.alpha_fn( *next_params ) )
            #print( y_pred )
            #y_true = MCMC.__to_percentile( self.target )
            #print( y_true )

            # X = self.alpha_fn( *next_params )
            X = self.alpha_fn( np.array(next_params) )
            Y = self.target
            next_nmi = self.optimize_fn( X , Y, round( len( X )/5 ) )

            #print("Iter:", iter)
            #print( "Next MI:" + str( next_nmi ))

            if next_nmi > prev_nmi:
                # print( "Exploit:")
                # print( next_nmi )
                # print( next_params )
                #print( self.std_guess(iter, self.num_iters))
                #print( self.explore_factor(iter, self.num_iters))
                return [ next_params, next_nmi ]
            else:
                ratio = next_nmi/prev_nmi

                uniform = np.random.uniform(0,1 )
                if ratio > uniform * self.explore_factor( iter, self.num_iters ):
                    # print("Explore:")
                    # print(next_nmi)
                    # print(next_params)
                    #print(self.std_guess(iter, self.num_iters))
                    #print(self.explore_factor(iter, self.num_iters))
                    return [ next_params, next_nmi ]
                else:
                    return [ prev_params, prev_nmi ]
        else:
            return [ prev_params, prev_nmi ]

    def optimize(self):

        prev_params = self.initial_params
        [ prev_params, prev_nmi] = self.do_step( 0, prev_params, -1 )
        all_results = []

        for i in range( 0, self.num_iters):
            # print( i )
            # if round( i / 100 ) == i/100:
            #     print( "Current: "  + str( i ) + " of " + str( self.num_iters ))
            [next_params, next_nmi] = self.do_step( i, prev_params, prev_nmi)
            all_results.append( [next_params, next_nmi, i ])
            prev_params = next_params
            prev_nmi = next_nmi

        return all_results

    def explore_factor( self, iter, num_iters ):
        if iter < 0.1 * num_iters:
            return 0.5
        if iter < 0.3 * num_iters:
            return 0.8
        if iter < 0.5 * num_iters:
            return 1
        if iter < 0.75 * num_iters:
            return 1.5
        if iter < 0.8 * num_iters:
            return 2
        if iter < 0.9 * num_iters:
            return 3
        if iter < 1 * num_iters:
            return 4
        return 5
        #return 0.1



    def analyse_results(self, all_results, top_n = 5 ):
        params = [ x[0] for x in all_results[round(self.burn*len(all_results)):]]
        nmis = [ x[1] for x in all_results[round(self.burn*len(all_results)):]]
        iteration = [x[2] for x in all_results[round(self.burn * len(all_results)):]]
        best_nmis = sorted( nmis, reverse=True)
        best_nmis = best_nmis[:top_n]

        best_params = []
        best_nmi = []
        best_iteration = []

        for p, n, it in zip( params, nmis, iteration ):
            if n >= best_nmis[-1]:
                best_params.append( p )
                best_nmi.append( n )
                best_iteration.append(it)
            if len( best_nmi ) == top_n:
                break

        return best_params, best_nmi, best_iteration


class MCMC_Indicator( MCMC ):


    def __init__(self, indicator, initial_args, feature, target_col, df, num_iters, prior, fltr ):
        self.target_col = target_col
        self.filter = fltr
        self.indicator = indicator
        self.feature = feature
        self.df = df
        MCMC.__init__( self, alpha_fn=self.create_alpha_fn(), alpha_fn_params_0=self.create_alpha_args( initial_args ),
                       target = self.create_target(),
                       num_iters = num_iters, prior=prior )


    def transition_fn(self, cur, iter ):
        std = self.std_guess( iter, self.num_iters )
        return [ round( x ) for x in np.random.normal( cur, std, ( len(cur),) ) ]


    def create_alpha_fn(self):
        indicator = self.indicator
        def alpha_fn( *args_to_optimize ):
            feature = self.feature
            df = self.df
            ind_args = list( args_to_optimize )
            print( ind_args)
            ind_args.append( feature )
            print( "Indicator initialization args")
            print( ind_args )
            id = indicator(*ind_args)
            print( "Indicator application args" )
            modified_df = id.apply( df )

            modified_df = modified_df.drop([ self.target_col ], axis = 1)
            modified_df = pd.concat( [ modified_df, self.df[ self.target_col ] ], axis = 1, join="inner")
            modified_df = self.filter( modified_df, id.describe(), self.target_col )
            modified_df = modified_df.dropna()

            self.target = modified_df[ self.target_col ].values
            return modified_df[ id.describe() ].values

        return alpha_fn

    def create_target(self):
        target = self.df[ self.target_col ]
        print( target.tail(10))
        return target

    def create_alpha_args(self, args ):

        all_args = args

        print( "Alpha args")
        print( all_args )
        return all_args





simple_test = True
if __name__ == "__main__":

    if simple_test:

        m = np.random.normal( 0, 1, 5000 )
        obs = [ 3 ** x for x in m ]

        def optim_fn( x, y, bins ):
            diff = [ (a-b)*(a-b) for a, b in zip( x,y)]
            return -sum( diff )

        def alpha_fn( a, b ):
    #        u = np.random.normal( 0, 1, 5000 )
            y = [( a * b ) ** x for x in m ]
            return y


        def prior( params ):
            if params[ 0 ] < 0 or params[1] < 0:
                return 0
            return 1



    #    from functools import partial
        guess = [ 3, 2 ]
    #    b_std = partial( std_guess, guess )

        mc = MCMC( alpha_fn, guess, obs, 10000, prior =  prior, optimize_fn=optim_fn )
        rs = mc.optimize()
        print( mc.analyse_results( rs, top_n=2 ))

    else:

        def prior( params ):
            if params[ 0 ] < 5:
                return 0
            return 1

        def fltr( df, indicator_col, target_col ):
            p = Pivot( 1, indicator_col )
            new_df = p.apply( df )
            new_df = new_df[new_df[p.describe()] == 1]
            #mask = (new_df[target_col] >= 1) | (new_df[target_col] <= -1)
            #new_df = new_df[ mask ]

            return new_df

        from Backtester import Backtester

        b = Backtester(["FCEL", "MSTR"], {"FCEL": "united states", "MSTR": "united states"}, {}, 0.001, 0.002, None)
        b.insert_rebal_set("1/1/2021", "FCEL", 0.5)
        b.insert_rebal_set("1/1/2021", "MSTR", 0.5)
        b.insert_rebal_set("21/2/2021", "FCEL", 0)
        b.insert_rebal_set("21/2/2021", "MSTR", 0)
        b.insert_rebal_set("1/5/2021", "FCEL", 1)
        b.insert_rebal_set("1/5/2021", "MSTR", 0)
        b.do_work("1/1/2013", "6/5/2021")
        df = b.full_data
        rr = RollingFSharpe( 30, "FCEL_Close" )
        df = rr.apply( df )
        dfm = DataframeManipulator( df )
        dfm.add_lookback_func( rr.describe(), "percentile", 260, new_column_name=rr.describe() + "_P" )
        df = dfm.df
        df = df.dropna()
        mcmc = MCMC_Indicator( Fisher, [ 100 ], "FCEL_Close", rr.describe(), df, 1000, prior, fltr )
        rs = mcmc.optimize()
        print( mcmc.analyse_results( rs, top_n=10 ))





