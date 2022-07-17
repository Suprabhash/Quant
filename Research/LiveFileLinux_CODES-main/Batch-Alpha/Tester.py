import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
from datetime import timedelta

import os.path
from os import path
import warnings
warnings.filterwarnings('ignore')

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import investpy
from datetime import date
import math
import numpy as np
import pandas as pd
import multiprocessing
import pickle
import eikon as ek
import yfinance as yf
from scipy.stats import percentileofscore
import scipy
import os
import zipfile


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

    def transition_fn(self, cur, iter ):

        #print("Inside transition_fn")


        std = self.std_guess( iter, self.num_iters )
        new_guesses = []
        for c, s in zip( cur, std):

            #print("Inside for loop")

            loop = True
            while loop:

                #print("Inside while loop")

                new_guess = np.random.normal( c, s, (1,))

                #print(f"New guess {new_guess}")
                #print(f"c: {c}")
                #print(f"s: {s}")

                if new_guess[0] <= self.upper_limit and new_guess[0] >= self.lower_limit:
                    new_guesses.append( new_guess[0] )
                    loop = False
        return list( new_guesses )

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

        c_XY = np.histogram2d(X, Y, bins)[0]
        c_X = np.histogram(X, bins)[0]
        c_Y = np.histogram(Y, bins)[0]

        H_X = MCMC.__shan_entropy(c_X)
        H_Y = MCMC.__shan_entropy(c_Y)
        H_XY = MCMC.__shan_entropy(c_XY)

        NMI = 2*(H_X + H_Y - H_XY)/(H_X+H_Y)
        return NMI

    def do_step(self, iter, prev_params, prev_nmi ):

        #print("Inside do_step")

        next_params = self.transition_fn( prev_params, iter )

        if self.prior( next_params ) != 0:

            #y_pred = MCMC.__to_percentile( self.alpha_fn( *next_params ) )
            #print( y_pred )
            #y_true = MCMC.__to_percentile( self.target )
            #print( y_true )

            X = self.alpha_fn( *next_params )
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

    def std_guess( self, iter, num_iters ):
        stds = []
        guesses = self.initial_params
        for guess in guesses:
            num_digits = len( str( round(guess) ))
            std = (10 ** ( num_digits-2 ))
            if iter < 0.5 * num_iters:
                std_factor = 2
            elif iter < 0.65 * num_iters:
                std_factor = 1
            elif iter < 0.85 * num_iters:
                std_factor = 0.75
            elif iter < 0.95 * num_iters:
                std_factor = 0.5
            elif iter < 0.99 * num_iters:
                std_factor = 0.1
            elif iter < num_iters:
                std_factor = 0.01
            #std_factor = 0.1
            stds.append( std * std_factor )
        return stds

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

class DataframeManipulator:

    def __init__(self, df ):
        self.df = df

    def look_back( self, column_name, num_rows, new_column_name = None ):
        if new_column_name is None:
            new_column_name = column_name + "_T-" + str( num_rows )
        self.df[ new_column_name ] = self.df[ column_name ].shift( num_rows )

    def look_forward( self, column_name, num_rows, new_column_name = None ):
        if new_column_name is None:
            new_column_name = column_name + "_T+" + str( num_rows )
        self.df[ new_column_name ] = self.df[ column_name ].shift( -num_rows )

    def extend_explicit(self, values, new_column_name ):
        self.df[ new_column_name ] = values

    def delete_cols(self, column_names ):
        if column_names != []:
            self.df = self.df.drop( column_names, axis = 1)

    def make_hl2(self, high, low ):
        self.df[ "HL2" ] = (self.df[ high ] + self.df[low])/2

    def extend_with_func(self, func, new_column_name, args = () ):
        self.df[ new_column_name ] = self.df.apply( func, axis = 1, args = args )

    def drop_na(self):
        self.df =  self.df.dropna().copy()

    def add_lookback_func(self, column_name, lookback_fn, lookback_dur, new_column_name = None, adjust = False ):
        df_temp = self.df[column_name]
        if new_column_name is None:
            new_column_name = column_name + "_" + lookback_fn + "_" + str( lookback_dur )
        if lookback_fn == "max":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[ new_column_name ] = r.max()
        elif lookback_fn == "rma":
            r = df_temp.ewm( min_periods=lookback_dur, adjust=adjust, alpha = 1/lookback_dur)
            self.df[new_column_name] = r.mean()
        elif lookback_fn == "ema":
            r = df_temp.ewm( com = lookback_dur - 1, min_periods=lookback_dur, adjust = adjust)
            self.df[new_column_name] = r.mean()
        elif lookback_fn == "sma":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.mean()
        elif lookback_fn == "max":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.max()
        elif lookback_fn == "min":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.min()
        elif lookback_fn == "percentile":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.apply( lambda x: scipy.stats.percentileofscore( x, x[-1]))
        elif lookback_fn == "std":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.std()
        elif lookback_fn == "sum":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.sum()


    def reverse_column(self, column_name, new_column_name ):
        df_temp = self.df[ column_name ]
        df_temp = df_temp.iloc[::-1].values
        if new_column_name is None:
            self.df[ column_name] = df_temp
        else:
            self.df[ new_column_name] = df_temp


    def find_filter(self, column_name, filter_mask ):
        df_temp = self.df[ filter_mask ]
        return df_temp[ column_name ]

class Misc:

    BLANK = "<BLANK>"

    def __init__(self):
        pass


    @staticmethod
    def apply_if_not_present( df, cls, to_delete ):
        try:
            idx = df.columns.get_loc( cls.describe() )
        except:
            print( "Could not find " + str( cls.describe() ))
            df = cls.apply(df)
            to_delete.append( cls.describe() )
        return [ df, to_delete ]

    @staticmethod
    def roc_pct(row, horizon, feature ):
        change = row[ feature ] - row[ feature + "_T-" + str( horizon ) ]
        change_pct = change/row[feature + "_T-" + str( horizon )]
        return change_pct

    @staticmethod
    def change(row, horizon, feature):
        chg = row[feature] - row[feature + "_T-" + str(horizon)]
        return chg

    @staticmethod
    def rsi( row, rma_adv, rma_dec, sum_n_adv, sum_n_dec ):
        sum_n_adv_v = abs(row[sum_n_adv ])
        sum_n_dec_v = abs(row[sum_n_dec])

        rma_adv_v = abs( row[rma_adv])
        rma_dec_v = abs( row[rma_dec])

        mean_adv_v = rma_adv_v
        mean_dec_v = rma_dec_v

        if mean_dec_v == 0:
            ratio = 0
        else:
            ratio = 100/(1+(mean_adv_v/mean_dec_v))

        r = 100 - ratio

        return r

class Indicator:


    def __init__(self, feature ):
        self.feature = feature

    def apply(self, df ):
        pass

    def describe(self):
        return ( "SHELL_INDICATOR" )

class ROC(Indicator):

    def __init__(self, period, feature):
        self.period = period
        Indicator.__init__(self, feature)

    def describe(self):
        return ("ROC_" + self.feature + "_" + str(self.period))


    def apply(self, df):
        dfm = DataframeManipulator(df)
        dfm.look_back(self.feature, self.period )
        dfm.extend_with_func( Misc.roc_pct, self.describe(), ( self.period, self.feature, ) )
        return dfm.df

class Metrics():

    def __init__(self,feature ):
        self.feature = feature
        pass

    def describe(self):
        return ( "Empty" )

    def apply(self, df ):
        return None

class RollingSharpe( Metrics ):
    def __init__(self, lookback, feature ):
        self.lookback = lookback
        Metrics.__init__( self, feature )

    def describe(self):
        return ( "RSharpe_" + str( self.lookback ) + "_" + self.feature)

    @staticmethod
    def __make_sharpe( row, roc_col, std_col ):
        roc = row[ roc_col ]
        std = row[ std_col ]
        return roc/std * math.sqrt( 252 )

    def apply(self, df):
        roc = ROC( self.lookback, self.feature )
        df = roc.apply( df )
        df[ self.describe() ] = df[ roc.describe() ]
        dfm = DataframeManipulator( df )
        dfm.add_lookback_func( self.feature, "std", self.lookback )
        dfm.extend_with_func( RollingSharpe.__make_sharpe, self.describe(), (roc.describe(), self.feature + "_std_" + str( self.lookback ),) )
        dfm.delete_cols( [ roc.describe(), self.feature + "_std_" + str( self.lookback ) ])
        return dfm.df

class RollingFSharpe(Metrics):
    def __init__(self, lookfwd, feature):
        self.lookfwd = lookfwd
        Metrics.__init__(self, feature)

    def describe(self):
        return ("RFSharpe_" + str(self.lookfwd) + "_" + self.feature)


    def apply(self, df):
        delete_cols = []
        sharpe = RollingSharpe( self.lookfwd, self.feature )
        try:
            idx = df.columns.get_loc( sharpe.describe())
        except:
            df = sharpe.apply( df )
            delete_cols.append( sharpe.describe() )
        dfm = DataframeManipulator( df )
        dfm.look_forward( sharpe.describe(), self.lookfwd - 1, self.describe())
        if delete_cols != []:
            dfm.delete_cols( delete_cols )
        return dfm.df

class RollingMax( Metrics ):
    def __init__(self, lookback, feature ):
        self.lookback = lookback
        Metrics.__init__( self, feature )


    def describe(self):
        return ( "RMX_" + str( self.lookback ) + "_" + self.feature)

    @staticmethod
    def __Max( row, du_col, feature ):
        return ( max( row[ feature ], row[ du_col ] ) )

    def apply(self, df):
        dfm = DataframeManipulator( df )
        dfm.add_lookback_func( self.feature, "max", self.lookback )
        dfm.extend_with_func( RollingMax.__Max, self.describe(), ( self.feature + "_max_" + str( self.lookback ), self.feature) )
        dfm.delete_cols( [ self.feature + "_max_" + str( self.lookback ) ])
        return dfm.df

class RollingFMax( Metrics ):
    def __init__(self, lookfwd, feature ):
        self.lookfwd = lookfwd
        Metrics.__init__( self, feature )


    def describe(self):
        return ( "RFMX_" + str( self.lookfwd ) + "_" + self.feature)

    def apply(self, df):
        to_delete = []
        rmx = RollingMax( self.lookfwd, self.feature )
        [ df, to_delete ] = Misc.apply_if_not_present( df, rmx, to_delete )
        dfm = DataframeManipulator( df )
        dfm.look_forward( rmx.describe(), self.lookfwd - 1, self.describe() )
        dfm.delete_cols( to_delete )
        return dfm.df

class RollingMin( Metrics ):
    def __init__(self, lookback, feature ):
        self.lookback = lookback
        Metrics.__init__( self, feature )


    def describe(self):
        return ( "RMN_" + str( self.lookback ) + "_" + self.feature)

    @staticmethod
    def __Min( row, dd_col, feature ):
        return ( min( row[ feature ], row[ dd_col ] ) )

    def apply(self, df):
        dfm = DataframeManipulator( df )
        dfm.add_lookback_func( self.feature, "min", self.lookback )
        dfm.extend_with_func( RollingMin.__Min, self.describe(), ( self.feature + "_min_" + str( self.lookback ), self.feature) )
        dfm.delete_cols( [ self.feature + "_min_" + str( self.lookback ) ])
        return dfm.df

class RollingFMin( Metrics ):
    def __init__(self, lookfwd, feature ):
        self.lookfwd = lookfwd
        Metrics.__init__( self, feature )


    def describe(self):
        return ( "RFMN_" + str( self.lookfwd ) + "_" + self.feature)

    def apply(self, df):
        to_delete = []
        rmn = RollingMin( self.lookfwd, self.feature )
        [ df, to_delete ] = Misc.apply_if_not_present( df, rmn, to_delete )

        dfm = DataframeManipulator( df )
        dfm.look_forward( rmn.describe(), self.lookfwd - 1, self.describe() )
        dfm.delete_cols( to_delete )
        return dfm.df

class RollingFDD( Metrics ):
    def __init__(self, lookback, feature ):
        self.lookback = lookback
        Metrics.__init__( self, feature )

    def describe(self):
        return ( "RFDD_" + str( self.lookback ) + "_" + self.feature)
    @staticmethod
    def __DD( row, dd_col, feature ):
        return ( row[ dd_col ]  - row[ feature ] )/row[ feature ]

    def apply(self, df):
        to_delete = []
        fmn = RollingFMin(self.lookback, self.feature)
        [df, to_delete] = Misc.apply_if_not_present(df, fmn, to_delete)
        dfm = DataframeManipulator( df )
        dfm.extend_with_func( RollingFDD.__DD, self.describe(), ( fmn.describe(), self.feature  ))

        dfm.delete_cols( to_delete )
        return dfm.df

class RollingFDU( Metrics ):
    def __init__(self, lookback, feature ):
        self.lookback = lookback
        Metrics.__init__( self, feature )
    def describe(self):
        return ( "RFDU_" + str( self.lookback ) + "_" + self.feature)
    @staticmethod
    def __DU( row, du_col, feature ):
        return ( row[ du_col ] - row[ feature ] )/row[ feature ]
    def apply(self, df):
        to_delete = []
        fmx = RollingFMax( self.lookback, self.feature )
        [df, to_delete] = Misc.apply_if_not_present(df, fmx, to_delete)

        dfm = DataframeManipulator( df )
        dfm.extend_with_func( RollingFDU.__DU, self.describe(), ( fmx.describe(), self.feature  ))
        dfm.delete_cols( to_delete )
        return dfm.df

class RollingReturn( Metrics ):
    def __init__(self, lookback, feature ):
        self.lookback = lookback
        Metrics.__init__( self, feature )


    def describe(self):
        return ( "RRet_" + str( self.lookback ) + "_" + self.feature)

    def apply(self, df):
        roc = ROC( self.lookback, self.feature )
        df = roc.apply( df )
        df[ self.describe() ] = df[ roc.describe() ].copy()
        dfm = DataframeManipulator( df )
        dfm.delete_cols( [ roc.describe() ])
        return dfm.df

class RollingFReturn(Metrics):
    def __init__(self, lookfwd, feature):
        self.lookfwd = lookfwd
        Metrics.__init__( self, feature)

    def describe(self):
        return ("RFRet_" + str(self.lookfwd) + "_" + self.feature)

    def apply(self, df):
        to_delete = []
        rr = RollingReturn( self.lookfwd, self.feature )
        [ df, to_delete ] = Misc.apply_if_not_present( df, rr, to_delete )

        dfm = DataframeManipulator( df )
        dfm.look_forward( rr.describe(), self.lookfwd - 1, self.describe() )
        dfm.delete_cols( to_delete )
        df = dfm.df
        return df

class RollingFRR(Metrics):
    def __init__(self, lookfwd, feature):
        self.lookfwd = lookfwd
        Metrics.__init__( self, feature)

    @staticmethod
    def __do_rr( row, du_col, dd_col ):
        if row[ dd_col ] == 0:
            return 100000
        else:
            return abs( row[ du_col ]/row[ dd_col ] )

    def describe(self):
        return ("RFRR_" + str(self.lookfwd) + "_" + self.feature)

    def apply(self, df):
        to_delete = []
        du = RollingFDU( self.lookfwd, self.feature )
        dd = RollingFDD( self.lookfwd, self.feature )

        [df, to_delete] = Misc.apply_if_not_present(df, du, to_delete)
        [df, to_delete] = Misc.apply_if_not_present(df, dd, to_delete)

        dfm = DataframeManipulator( df )
        dfm.extend_with_func( RollingFRR.__do_rr, self.describe(), ( du.describe(), dd.describe(), ) )

        dfm.delete_cols( to_delete )
        df = dfm.df

        return df

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

def get_data_investpy( symbol, country, from_date, to_date ):
  find = investpy.search.search_quotes(text=symbol, products =["stocks", "etfs", "indices"] )
  for f in find:
    #print( f )

    if f.symbol.lower() == symbol.lower() and f.country.lower() == country.lower():
      break
  if f.symbol.lower() != symbol.lower():
    return None
  ret = f.retrieve_historical_data(from_date=from_date, to_date=to_date )
  if ret is None:
    try:
      ret = investpy.get_stock_historical_data(stock=symbol,
                                      country=country,
                                      from_date=from_date,
                                      to_date=to_date)
    except:
      ret = None
  if ret is None:
    try:
      ret = investpy.get_etf_historical_data(etf=symbol,
                                      country=country,
                                      from_date=from_date,
                                      to_date=to_date)
    except:
      ret = None

  if ret is None:
    try:
      ret = investpy.get_index_historical_data(index=symbol,
                                      country=country,
                                      from_date=from_date,
                                      to_date=to_date)
    except:
      ret = None
  ret.drop(["Change Pct"], axis=1, inplace=True)
  return ret

def get_data(ticker, api):

    if api == "yfinance":
        temp_og = yf.download(ticker, start = '2007-07-01', end= str(date.today()+timedelta(1)))
        temp_og.reset_index(inplace=True)
        temp_og.drop(['Adj Close'], axis=1, inplace=True)
        temp_og = temp_og.loc[temp_og["Close"]>1]
        temp_og = add_fisher(temp_og)

    if api =="investpy":
        temp_og = get_data_investpy(symbol=ticker, country='united states', from_date="01/07/2007",to_date=(date.today()+timedelta(1)).strftime("%d/%m/%Y"))
        temp_og.reset_index(inplace=True)
        temp_og = add_fisher(temp_og)

    if api == "reuters":
        temp_og = ek.get_timeseries(ticker, start_date='2007-07-01', end_date=str(date.today() + timedelta(1)))
        temp_og.reset_index(inplace=True)
        temp_og.rename(columns={"HIGH": "High", "CLOSE": "Close", "LOW": "Low", "OPEN": "Open", "VOLUME": "Volume"},
                       inplace=True)
        temp_og.drop(['COUNT'], axis=1, inplace=True)

    return temp_og

def add_fisher(temp):
    for f_look in range(50, 400, 20):
        temp[f'Fisher{f_look}'] = fisher(temp, f_look)
    return temp

def fisher(ohlc, period):
    def __round(val):
        if (val > .99):
            return .999
        elif val < -.99:
            return -.999
        return val

    from numpy import log, seterr
    seterr(divide="ignore")
    med = (ohlc["High"] + ohlc["Low"]) / 2
    ndaylow = med.rolling(window=period).min()
    ndayhigh = med.rolling(window=period).max()
    med = [0 if math.isnan(x) else x for x in med]
    ndaylow = [0 if math.isnan(x) else x for x in ndaylow]
    ndayhigh = [0 if math.isnan(x) else x for x in ndayhigh]
    raw = [0] * len(med)
    for i in range(0, len(med)):
        try:
            raw[i] = 2 * ((med[i] - ndaylow[i]) / (ndayhigh[i] - ndaylow[i]) - 0.5)
        except:
            ZeroDivisionError
    value = [0] * len(med)
    value[0] = __round(raw[0] * 0.33)
    for i in range(1, len(med)):
        try:
            value[i] = __round(0.33 * raw[i] + 0.67 * value[i - 1])
        except:
            ZeroDivisionError
    _smooth = [0 if math.isnan(x) else x for x in value]
    fish1 = [0] * len(_smooth)
    for i in range(1, len(_smooth)):
        fish1[i] = ((0.5 * (np.log((1 + _smooth[i]) / (1 - _smooth[i]))))) + (0.5 * fish1[i - 1])
    fish2 = fish1[1:len(fish1)]
    # plt.figure(figsize=(18, 8))
    # plt.plot(ohlc.index, fish1, linewidth=1, label="Fisher_val")
    # plt.legend(loc="upper left")
    # plt.show()
    return fish1

def valid_dates(dates_all):
    dates = []
    i = 0
    while True:
        dates.append(dates_all[i])
        if dates_all[i] > pd.to_datetime(date.today()):
            break
        i = i + 1
    return dates

def create_final_signal_weights(signal, params, weights, nos):
    params = params[:nos]
    for i in range(len(params)):
        if i==0:
            signals =  signal[params.iloc[i]["Name"]].to_frame().rename(columns={'params.iloc[i]["Name"]': f'Signal{i + 1}'})
        else:
            signals = pd.merge(signals, signal[params.iloc[i]["Name"]].to_frame().rename(columns={'params.iloc[i]["Name"]': f'Signal{i + 1}'}), left_index=True, right_index=True)
            #signalsg = pd.concat([signalsg, signalg[paramsg.iloc[i]["Name"]].to_frame().rename(columns={'paramsg.iloc[i]["Name"]': f'Signal{i + 1}'})],axis=1)

    sf = pd.DataFrame(np.dot(np.where(np.isnan(signals),0,signals), weights))
    #return sf.set_index(signals.index).rename(columns={0: 'signal'})

    return pd.DataFrame(np.where(sf > 0.5, 1, 0)).set_index(signals.index).rename(columns={0: 'signal'})

    #return pd.DataFrame(np.where(signalsg.mean(axis=1, skipna=True) > 0.5, 1, 0)).set_index(signalsg.index).rename(columns={0:'signal'}), \
    #        pd.DataFrame(np.where(signalsn.mean(axis=1, skipna=True) > 0.5, 1, 0)).set_index(signalsn.index).rename(columns={0: 'signal'})

    #portfolio scaling
    #return pd.DataFrame(signalsg.mean(axis=1, skipna=True)).rename(columns={0:'signal'}), \
    #       pd.DataFrame(signalsn.mean(axis=1, skipna=True)).rename(columns={0:'signal'})

def optimize_weights_and_backtest(input):

    def get_equity_curve(*args):
        weights = []
        for weight in args:
            weights.append(weight)
        # weights.append((1-sum(weights)))
        weights = pd.DataFrame(weights)
        weights = weights / weights.sum()
        signal_final = create_final_signal_weights(signal, params, weights, num_strategies)
        inp = pd.merge(train.set_index(train["Date"]), signal_final, left_index=True, right_index=True)
        test_strategy = FISHER_MCMC(inp.drop(['signal', 'Date'], axis=1), inp['signal'])
        _ = test_strategy.generate_signals()
        ec = test_strategy.signal_performance(10000, 6)

        return ec

    def sortino(x, y, bins):
        ecdf = x[["S_Return"]]
        stdev_down = ecdf.loc[ecdf["S_Return"] < 0, "S_Return"].std() * (252 ** .5)
        if math.isnan(stdev_down):
            stdev_down = 0.0
        if stdev_down != 0.0:
            sortino = (ecdf["S_Return"].mean() - 0.06 / 252) * 252 / stdev_down
        else:
            sortino = 0
        return np.float64(sortino)

    def sharpe(x, y, bins):
        ecdf = x[["S_Return"]]

        stdev = ecdf["S_Return"].std() * (252 ** .5)
        if math.isnan(stdev):
            stdev = 0.0
        if stdev != 0.0:
            sharpe = (ecdf["S_Return"].mean() - 0.06 / 252) * 252 / stdev
        else:
            sharpe = 0
        return np.float64(sharpe)

    def rolling_sharpe(x, y, bins):
        r_window = 252
        ecdf = x[["Date", "S_Return"]]
        ecdf['RSharpeRatio_Series'] = (ecdf["S_Return"].rolling(window=r_window, min_periods=1).mean() - 0.06 / 252) / ecdf[
            "S_Return"].rolling(window=r_window, min_periods=1).std() * (252 ** .5)
        ecdf['RSharpeRatio_Series'] = ecdf['RSharpeRatio_Series'].replace(to_replace=math.inf, value=0)
        ecdf['RSharpeRatio_Series'] = ecdf['RSharpeRatio_Series'].replace(to_replace=math.nan, value=0)
        RSharpeRatio = ecdf['RSharpeRatio_Series'].mean()
        return np.float64(RSharpeRatio)

    def rolling_sortino(x, y, bins):
        r_window = 252
        ecdf = x[["Date", "S_Return"]]
        ecdf["S_Return_stdev"] = ecdf["S_Return"].copy()
        ecdf["S_Return_stdev"] = np.where(ecdf["S_Return_stdev"] >= 0, ecdf["S_Return_stdev"], np.nan)
        ecdf['RStDev Annualized Downside Return_Series'] = ecdf["S_Return_stdev"].rolling(window=r_window,
                                                                                          min_periods=1).std() * (
                                                                       252 ** .5)
        ecdf['RStDev Annualized Downside Return_Series'] = ecdf['RStDev Annualized Downside Return_Series'].replace(
            to_replace=math.inf, value=0)
        ecdf['RStDev Annualized Downside Return_Series'] = ecdf['RStDev Annualized Downside Return_Series'].replace(
            to_replace=math.nan, value=0)
        ecdf['RSortinoRatio_Series'] = np.where(ecdf['RStDev Annualized Downside Return_Series'] != 0.0,
                                                (ecdf["S_Return"].rolling(window=r_window, min_periods=1).mean() - 0.06 / 252) * 252 /
                                                ecdf['RStDev Annualized Downside Return_Series'], 0)
        RSortinoRatio = ecdf['RSortinoRatio_Series'].mean()
        return np.float64(RSortinoRatio)

    def rolling_cagr(x, y, bins):
        ecdf = x[["Date", "S_Return"]]
        ecdf["Date"] = pd.to_datetime(ecdf["Date"])
        ecdf = ecdf['S_Return'].to_frame().set_index(ecdf["Date"])
        ecdf['Portfolio'] = 1 + ecdf['S_Return'].expanding().sum()
        ecdf['Portfolio'] = ecdf['Portfolio']
        ecdf['Portfolio_1yr'] = ecdf['Portfolio'].to_frame().shift(365, freq='D')
        ecdf['Portfolio_1yr'] = ecdf['Portfolio_1yr'].fillna(method="ffill")
        ecdf['RCAGR_Strategy_Series'] = ecdf['Portfolio'] / ecdf['Portfolio_1yr'] - 1
        RCAGR_Strategy = ecdf['RCAGR_Strategy_Series'].mean()
        return np.float64(RCAGR_Strategy)

    def maxdrawup_by_maxdrawdown(x, y, bins):
        r_window = 252
        ecdf = x[["Date", "S_Return"]]
        ecdf['Strategy_Return'] = ecdf['S_Return'].expanding().sum()
        ecdf['Portfolio Value'] = ((ecdf['Strategy_Return'] + 1) * 1)
        ecdf['Portfolio Value'][0] = 1
        ecdf['Drawdown_Series'] = 1 - ecdf['Portfolio Value'] / np.maximum.accumulate(ecdf['Portfolio Value'])
        ecdf['Max_Drawdown_Series'] = ecdf['Drawdown_Series'].rolling(window=r_window, min_periods=1).max()
        ecdf['Drawup_Series'] = ecdf['Portfolio Value'] / np.minimum.accumulate(ecdf['Portfolio Value']) - 1
        ecdf['Max_Drawup_Series'] = ecdf['Drawup_Series'].rolling(window=r_window, min_periods=1).max()
        ecdf['Drawup/Drawdown_Series'] = ecdf['Max_Drawup_Series'] / ecdf['Max_Drawdown_Series']
        ecdf['Drawup/Drawdown_Series'] = ecdf['Drawup/Drawdown_Series'].replace(math.inf, 100)
        RDrawupDrawdown = ecdf['Drawup/Drawdown_Series'].mean()
        # ecdf['Drawdown_Series'] = 1 - ecdf['Portfolio Value'] / np.maximum.accumulate(ecdf['Portfolio Value'])
        # ecdf['Drawup_Series'] = ecdf['Portfolio Value'] / np.minimum.accumulate(ecdf['Portfolio Value']) - 1
        # RDrawupDrawdown = ecdf['Drawup_Series'].max() / ecdf['Drawdown_Series'].max()
        return np.float64(RDrawupDrawdown)

    def outperformance(x, y, bins):
        r_window = 252
        x["Date"] = pd.to_datetime(x["Date"])
        ecdf1 = x['S_Return'].to_frame().set_index(x["Date"])
        ecdf2 = x['Return'].to_frame().set_index(x["Date"])
        ecdf1['Portfolio'] = 1 + ecdf1['S_Return'].expanding().sum()
        ecdf1['Portfolio_1yr'] = ecdf1['Portfolio'].to_frame().shift(365, freq='D')
        ecdf2['Portfolio'] = 1 + ecdf2['Return'].expanding().sum()
        ecdf2['Portfolio_1yr'] = ecdf2['Portfolio'].to_frame().shift(365, freq='D')
        ecdf1['Portfolio_1yr'] = ecdf1['Portfolio_1yr'].fillna(method="ffill")
        ecdf1['RCAGR_Strategy_Series'] = ecdf1['Portfolio'] / ecdf1['Portfolio_1yr'] - 1
        ecdf2['Portfolio_1yr'] = ecdf2['Portfolio_1yr'].fillna(method="ffill")
        ecdf2['RCAGR_Market_Series'] = ecdf2['Portfolio'] / ecdf2['Portfolio_1yr'] - 1
        RCAGR_Strategy = ecdf1['RCAGR_Strategy_Series'].mean()
        RCAGR_Market = ecdf2['RCAGR_Market_Series'].mean()
        ROutperformance = RCAGR_Strategy - RCAGR_Market
        return np.float64(ROutperformance)

    def prior(params):
        return 1

    date_i = input[0]
    dates = input[1]
    temp_og = input[2]
    ss_test = input[3]
    res_test = input[4]
    num_strategies = input[5]
    metric = input[6]
    recalib_months = input[7]
    dates_all = input[8]

    #print(f"Training period begins: {str(dates[date_i])}")
    if (date_i - int(24 / recalib_months)) < 0:
        temp = temp_og.loc[
            (temp_og["Date"] > str(dates_all[dates_all.index(dates[date_i]) - int(24 / 3)])) & (
                        temp_og["Date"] < str(dates[date_i + 1]))].reset_index().drop(
            ['index'], axis=1)
    else:
        temp = temp_og.loc[
            (temp_og["Date"] > str(dates[date_i - int(24 / recalib_months)])) & (
                        temp_og["Date"] < str(dates[date_i + 1]))].reset_index().drop(
            ['index'], axis=1)


    train = temp.loc[(temp["Date"] < str(dates[date_i]))].reset_index().drop(['index'], axis=1)
    test = temp.loc[(temp["Date"] >= str(dates[date_i]))].reset_index().drop(['index'], axis=1)

    if len(ss_test[date_i]) > 0:
        if len(ss_test[date_i]) > num_strategies:
            selected_strategies = ss_test[date_i][:num_strategies]
        else:
            selected_strategies = ss_test[date_i]

        if len(res_test[date_i]) > num_strategies:
            res = res_test[date_i][:num_strategies]
        else:
            res = res_test[date_i]

        # print("Optimizing weights")
        strategies = ["Strategy" + str(i) for i in range(1, len(res) + 1)]
        params = selected_params(strategies, res)

        _, signal = top_n_strat_params_rolling(temp, res, to_train=True, num_of_strat=len(selected_strategies),
                                               split_date=str(dates[date_i]))
        #print("Running MCMC")
        guess = (0.5 * np.ones([1, len(selected_strategies)])).tolist()[0]

        iters = 20

        if len(guess)>1:
            if metric == 'rolling_sharpe':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=iters, prior=prior,
                          optimize_fn=rolling_sharpe, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'rolling_sortino':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=iters, prior=prior,
                          optimize_fn=rolling_sortino, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'rolling_cagr':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=iters, prior=prior,
                          optimize_fn=rolling_cagr, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'maxdrawup_by_maxdrawdown':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=iters, prior=prior,
                          optimize_fn=maxdrawup_by_maxdrawdown, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'outperformance':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=iters, prior=prior,
                          optimize_fn=outperformance, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            # Printing results:
            weights = []
            for weight in mc.analyse_results(rs, top_n=1)[0][0]:
                weights.append(weight)
            weights = pd.DataFrame(weights)
            weights = weights / weights.sum(axis=0)

        else:
            weights = pd.DataFrame([1])

        #print(pd.concat([params.drop(["Name"], axis=1), weights.rename(columns={0: 'Weights'})], axis=1))

        # signal_final = create_final_signal_weights(signal, params, weights, len(selected_strategies))
        # inp = pd.merge(train.set_index(train["Date"]), signal_final, left_index=True, right_index=True)
        #
        # test_strategy = FISHER_MCMC(inp.drop(['signal', 'Date'], axis=1), inp['signal'])
        # _ = test_strategy.generate_signals()
        # _ = test_strategy.signal_performance(1, 6)

        # print(f"Training period ends: {str(dates[date_i + 2])}")
        # print(f"Sortino for training period is: {test_strategy.daywise_performance['SortinoRatio']}")
        # print(f"Testing period begins: {str(dates[date_i + 2])}")

        _, signal = top_n_strat_params_rolling(temp, res, to_train=False, num_of_strat=len(selected_strategies),
                                               split_date=str(dates[date_i]))

        # Weights
        signal_final = create_final_signal_weights(signal, params, weights, len(selected_strategies))

        # equi-weighted
        # signal_final = create_final_signal(signal, params, len(selected_strategies))

        inp = pd.merge(test.set_index(test["Date"]), signal_final, left_index=True, right_index=True)

    else:
        inp = test.set_index(test["Date"])
        inp['signal'] = 0
        weights = pd.DataFrame()

    test_strategy = FISHER_MCMC(inp.drop(['signal', 'Date'], axis=1), inp['signal'])
    _ = test_strategy.generate_signals()
    tt = test_strategy.signal_performance(1, 6)
    return date_i, tt.set_index(pd.to_datetime(tt["Date"])).drop(columns="Date"), weights
    # print(f"Testing period ends: {str(dates[date_i + 3])}")
    # print(f"Sortino for testing period is: {test_strategy.daywise_performance['SortinoRatio']}")

def get_strategies_brute_force(inp):
    def get_equity_curve_embeddings(*args):
        f_look = args[0]
        f_look = 1 * round(f_look / 1)
        lb = round(10 * args[1]) / 10
        ub = round(10 * args[2]) / 10

        temp["fisher"] = temp[f'Fisher{f_look}']

        test_strategy = FISHER_bounds_strategy_opt(temp, lb, ub)
        _ = test_strategy.generate_signals()
        ec = test_strategy.signal_performance(10000, 6)
        return ec

    def AvgWinLoss(x, y, bins):
        ecdf = x[["S_Return", "Close", "signal", "trade_num"]]
        ecdf = ecdf[ecdf["signal"] == 1]
        trade_wise_results = []
        for i in range(max(ecdf['trade_num'])):
            trade_num = i + 1
            entry = ecdf[ecdf["trade_num"] == trade_num].iloc[0]["Close"]
            exit = ecdf[ecdf["trade_num"] == trade_num].iloc[-1]["Close"]
            trade_wise_results.append({'Trade Number': trade_num, 'Entry': entry, 'Exit': exit})
        trade_wise_results = pd.DataFrame(trade_wise_results)
        d_tp = {}
        if len(trade_wise_results) > 0:
            trade_wise_results["Win/Loss"] = np.where(trade_wise_results["Exit"] > trade_wise_results["Entry"], "Win",
                                                      "Loss")
            trade_wise_results["Return on Trade"] = trade_wise_results["Exit"] / trade_wise_results["Entry"] - 1
            d_tp["TotalWins"] = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Win"])
            d_tp["TotalLosses"] = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"])
            d_tp['TotalTrades'] = d_tp["TotalWins"] + d_tp["TotalLosses"]
            if d_tp['TotalTrades'] == 0:
                d_tp['HitRatio'] = 0
            else:
                d_tp['HitRatio'] = round(d_tp['TotalWins'] / d_tp['TotalTrades'], 4)
            d_tp['AvgWinRet'] = np.round(
                trade_wise_results[trade_wise_results["Win/Loss"] == "Win"]["Return on Trade"].mean(), 4)
            if math.isnan(d_tp['AvgWinRet']):
                d_tp['AvgWinRet'] = 0.0
            d_tp['AvgLossRet'] = np.round(
                trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"]["Return on Trade"].mean(), 4)
            if math.isnan(d_tp['AvgLossRet']):
                d_tp['AvgLossRet'] = 0.0
            if d_tp['AvgLossRet'] != 0:
                d_tp['WinByLossRet'] = np.round(abs(d_tp['AvgWinRet'] / d_tp['AvgLossRet']), 2)
            else:
                d_tp['WinByLossRet'] = 0
            if math.isnan(d_tp['WinByLossRet']):
                d_tp['WinByLossRet'] = 0.0
            if math.isinf(d_tp['WinByLossRet']):
                d_tp['WinByLossRet'] = 0.0
        else:
            d_tp["TotalWins"] = 0
            d_tp["TotalLosses"] = 0
            d_tp['TotalTrades'] = 0
            d_tp['HitRatio'] = 0
            d_tp['AvgWinRet'] = 0
            d_tp['AvgLossRet'] = 0
            d_tp['WinByLossRet'] = 0

        return np.float64(d_tp['WinByLossRet'])

    date_i = inp[0]
    dates = inp[1]
    temp_og = inp[2]
    train_months = inp[3]

    temp = temp_og.loc[(temp_og["Date"] > str(dates[date_i])) & (temp_og["Date"] < str(
        dates[date_i + int(train_months / 3)]))].reset_index().drop(['index'], axis=1)
    res = pd.DataFrame(columns=["Lookback", "Low Bound", "High Bound", "AvgWinLoss"])

    for f_look in range(110, 130, 20):  #50, 410, 20
        max_metric = 0
        for lb in np.round(np.arange(-1, 1, 2), decimals=1):
            for ub in np.round(np.arange(-1, 1, 2), decimals=1):#-7, 7, 0.25
                metric = AvgWinLoss(get_equity_curve_embeddings(f_look, lb, ub), 0, 0)
                if metric > max_metric:
                    max_metric = metric
                    res_iter = pd.DataFrame(
                        [{"Lookback": f_look, "Low Bound": lb, "High Bound": ub, "AvgWinLoss": metric}])
                    res = pd.concat([res, res_iter], axis=0)

    res.drop_duplicates(keep='first', inplace=True, ignore_index=True)
    res.sort_values("AvgWinLoss", axis=0, ascending=False, inplace=True)
    res["Optimization_Years"] = train_months / 12
    res = res.reset_index().drop(['index'], axis=1)
    return (date_i, res)

def backtest_sortino(x, y, bins):
    ecdf = x[["S_Return"]]
    stdev_down = ecdf.loc[ecdf["S_Return"] < 0, "S_Return"].std() * (252 ** .5)
    if math.isnan(stdev_down):
        stdev_down = 0.0
    if stdev_down != 0.0:
        sortino = (ecdf["S_Return"].mean() - 0.06 / 252) * 252 / stdev_down
    else:
        sortino = 0
    return np.float64(sortino)
def backtest_sharpe(x, y, bins):
    ecdf = x[["S_Return"]]

    stdev = ecdf["S_Return"].std() * (252 ** .5)
    if math.isnan(stdev):
        stdev = 0.0
    if stdev != 0.0:
        sharpe = (ecdf["S_Return"].mean() - 0.06 / 252) * 252 / stdev
    else:
        sharpe = 0
    return np.float64(sharpe)
def backtest_rolling_sharpe(x, y, bins):
    r_window = 252
    ecdf = x[["Date", "S_Return"]]
    ecdf['RSharpeRatio_Series'] = (ecdf["S_Return"].rolling(window=r_window, min_periods=1).mean() - 0.06 / 252) / ecdf[
        "S_Return"].rolling(window=r_window, min_periods=1).std() * (252 ** .5)
    ecdf['RSharpeRatio_Series'] = ecdf['RSharpeRatio_Series'].replace(to_replace=math.inf, value=0)
    ecdf['RSharpeRatio_Series'] = ecdf['RSharpeRatio_Series'].replace(to_replace=math.nan, value=0)
    RSharpeRatio = ecdf['RSharpeRatio_Series'].mean()
    return np.float64(RSharpeRatio)
def backtest_rolling_sortino(x, y, bins):
    r_window = 252
    ecdf = x[["Date", "S_Return"]]
    ecdf["S_Return_stdev"] = ecdf["S_Return"].copy()
    ecdf["S_Return_stdev"] = np.where(ecdf["S_Return_stdev"] >= 0, ecdf["S_Return_stdev"], np.nan)
    ecdf['RStDev Annualized Downside Return_Series'] = ecdf["S_Return_stdev"].rolling(window=r_window,
                                                                                      min_periods=1).std() * (
                                                               252 ** .5)
    ecdf['RStDev Annualized Downside Return_Series'] = ecdf['RStDev Annualized Downside Return_Series'].replace(
        to_replace=math.inf, value=0)
    ecdf['RStDev Annualized Downside Return_Series'] = ecdf['RStDev Annualized Downside Return_Series'].replace(
        to_replace=math.nan, value=0)
    ecdf['RSortinoRatio_Series'] = np.where(ecdf['RStDev Annualized Downside Return_Series'] != 0.0,
                                            (ecdf["S_Return"].rolling(window=r_window,
                                                                      min_periods=1).mean() - 0.06 / 252) * 252 /
                                            ecdf['RStDev Annualized Downside Return_Series'], 0)
    RSortinoRatio = ecdf['RSortinoRatio_Series'].mean()
    return np.float64(RSortinoRatio)
def backtest_rolling_cagr(x, y, bins):
    ecdf = x[["Date", "S_Return"]]
    ecdf = ecdf['S_Return'].to_frame().set_index(ecdf["Date"])
    ecdf['Portfolio'] = 1 + ecdf['S_Return'].expanding().sum()
    ecdf['Portfolio'] = ecdf['Portfolio']
    ecdf['Portfolio_1yr'] = ecdf['Portfolio'].to_frame().shift(365, freq='D')
    ecdf['Portfolio_1yr'] = ecdf['Portfolio_1yr'].fillna(method="ffill")
    ecdf['RCAGR_Strategy_Series'] = ecdf['Portfolio'] / ecdf['Portfolio_1yr'] - 1
    RCAGR_Strategy = ecdf['RCAGR_Strategy_Series'].mean()
    return np.float64(RCAGR_Strategy)
def backtest_rolling_maxdrawup_by_maxdrawdown(x, y, bins):
    r_window = 252
    ecdf = x[["Date", "S_Return"]]
    ecdf['Strategy_Return'] = ecdf['S_Return'].expanding().sum()
    ecdf['Portfolio Value'] = ((ecdf['Strategy_Return'] + 1) * 1)
    ecdf['Portfolio Value'][0] = 1
    ecdf['Drawdown_Series'] = 1 - ecdf['Portfolio Value'] / np.maximum.accumulate(ecdf['Portfolio Value'])
    ecdf['Max_Drawdown_Series'] = ecdf['Drawdown_Series'].rolling(window=r_window, min_periods=1).max()
    ecdf['Drawup_Series'] = ecdf['Portfolio Value'] / np.minimum.accumulate(ecdf['Portfolio Value']) - 1
    ecdf['Max_Drawup_Series'] = ecdf['Drawup_Series'].rolling(window=r_window, min_periods=1).max()
    ecdf['Drawup/Drawdown_Series'] = ecdf['Max_Drawup_Series'] / ecdf['Max_Drawdown_Series']
    ecdf['Drawup/Drawdown_Series'] = ecdf['Drawup/Drawdown_Series'].replace(math.inf, 100)
    RDrawupDrawdown = ecdf['Drawup/Drawdown_Series'].mean()
    # ecdf['Drawdown_Series'] = 1 - ecdf['Portfolio Value'] / np.maximum.accumulate(ecdf['Portfolio Value'])
    # ecdf['Drawup_Series'] = ecdf['Portfolio Value'] / np.minimum.accumulate(ecdf['Portfolio Value']) - 1
    # RDrawupDrawdown = ecdf['Drawup_Series'].max() / ecdf['Drawdown_Series'].max()
    return np.float64(RDrawupDrawdown)
def backtest_maxdrawup_by_maxdrawdown(x, y, bins):
    r_window = 252
    ecdf = x[["Date", "S_Return"]]
    ecdf['Strategy_Return'] = ecdf['S_Return'].expanding().sum()
    ecdf['Portfolio Value'] = ((ecdf['Strategy_Return'] + 1) * 1)
    ecdf['Portfolio Value'][0] = 1
    # ecdf['Drawdown_Series'] = 1 - ecdf['Portfolio Value'] / np.maximum.accumulate(ecdf['Portfolio Value'])
    # ecdf['Max_Drawdown_Series'] = ecdf['Drawdown_Series'].rolling(window=r_window, min_periods=1).max()
    # ecdf['Drawup_Series'] = ecdf['Portfolio Value'] / np.minimum.accumulate(ecdf['Portfolio Value']) - 1
    # ecdf['Max_Drawup_Series'] = ecdf['Drawup_Series'].rolling(window=r_window, min_periods=1).max()
    # ecdf['Drawup/Drawdown_Series'] = ecdf['Max_Drawup_Series'] / ecdf['Max_Drawdown_Series']
    # ecdf['Drawup/Drawdown_Series'] = ecdf['Drawup/Drawdown_Series'].replace(math.inf, 100)
    # RDrawupDrawdown = ecdf['Drawup/Drawdown_Series'].mean()
    ecdf['Drawdown_Series'] = 1 - ecdf['Portfolio Value'] / np.maximum.accumulate(ecdf['Portfolio Value'])
    ecdf['Drawup_Series'] = ecdf['Portfolio Value'] / np.minimum.accumulate(ecdf['Portfolio Value']) - 1
    DrawupDrawdown = ecdf['Drawup_Series'].max() / ecdf['Drawdown_Series'].max()
    return np.float64(DrawupDrawdown)
def backtest_rolling_outperformance(x, y, bins):
    r_window = 252
    ecdf1 = x['S_Return'].to_frame().set_index(x["Date"])
    ecdf2 = x['Return'].to_frame().set_index(x["Date"])
    ecdf1['Portfolio'] = 1 + ecdf1['S_Return'].expanding().sum()
    ecdf1['Portfolio_1yr'] = ecdf1['Portfolio'].to_frame().shift(365, freq='D')
    ecdf2['Portfolio'] = 1 + ecdf2['Return'].expanding().sum()
    ecdf2['Portfolio_1yr'] = ecdf2['Portfolio'].to_frame().shift(365, freq='D')
    ecdf1['Portfolio_1yr'] = ecdf1['Portfolio_1yr'].fillna(method="ffill")
    ecdf1['RCAGR_Strategy_Series'] = ecdf1['Portfolio'] / ecdf1['Portfolio_1yr'] - 1
    ecdf2['Portfolio_1yr'] = ecdf2['Portfolio_1yr'].fillna(method="ffill")
    ecdf2['RCAGR_Market_Series'] = ecdf2['Portfolio'] / ecdf2['Portfolio_1yr'] - 1
    RCAGR_Strategy = ecdf1['RCAGR_Strategy_Series'].mean()
    RCAGR_Market = ecdf2['RCAGR_Market_Series'].mean()
    ROutperformance = RCAGR_Strategy - RCAGR_Market
    return np.float64(ROutperformance)
def backtest_outperformance(x, y, bins):
    r_window = 252
    ecdf1 = x['S_Return'].to_frame().set_index(x["Date"])
    ecdf2 = x['Return'].to_frame().set_index(x["Date"])
    ecdf1['Portfolio'] = 1 + ecdf1['S_Return'].expanding().sum()
    ecdf2['Portfolio'] = 1 + ecdf2['Return'].expanding().sum()
    RCAGR_Strategy = ecdf1['Portfolio'][-1]/ecdf1['Portfolio'][1]-1
    RCAGR_Market = ecdf2['Portfolio'][-1]/ecdf2['Portfolio'][1]-1
    Outperformance = RCAGR_Strategy - RCAGR_Market
    return np.float64(Outperformance)

def corr_sortino_filter(inp):
    date_i = inp[0]
    dates = inp[1]
    temp_og = inp[2]
    res_total = inp[3]
    num_strategies = inp[4]
    train_monthsf = inp[5]

    temp = temp_og.loc[(temp_og["Date"] > str(dates[date_i])) & (temp_og["Date"] < str(dates[date_i + (int(train_monthsf/3)+1)]))].reset_index().drop(['index'], axis=1)
    res = res_total[date_i]
    x, y = corr_filter(temp, res, dates, date_i, num_strategies, train_monthsf)
    return date_i,x,y
def corr_filter(temp, res, dates, date_i, num_strategies, train_monthsf):
    res.sort_values("AvgWinLoss", axis=0, ascending=False, inplace=True)
    res.reset_index().drop(['index'], axis=1)
    returns, _ = top_n_strat_params_rolling(temp, res, to_train=True, num_of_strat=len(res), split_date =str(dates[date_i+int(train_monthsf/3)]))
    if returns.empty:
        return [], pd.DataFrame( columns=["Lookback", "Low Bound", "High Bound", "AvgWinLoss", "Optimization_Years"])
    corr_mat = returns.corr()
    first_selected_strategy = 'Strategy1'
    selected_strategies = strategy_selection(returns, corr_mat, num_strategies, first_selected_strategy)
    params = selected_params(selected_strategies, res)
    res = params.drop(["Name"], axis=1)
    return (selected_strategies, res)

def top_n_strat_params_rolling(temp, res, to_train, num_of_strat, split_date):
    if len(res)>0:
        for i in range(num_of_strat):
            f_look = res.iloc[i, 0]
            bf = res.iloc[i, 1]
            sf = res.iloc[i, 2]
            temp["fisher"] = temp[f'Fisher{int(f_look)}']
            if to_train:
                train = temp.loc[(temp["Date"] <= split_date)].reset_index().drop(['index'], axis=1)
            else:
                train = temp.loc[temp["Date"] > split_date].reset_index().drop(['index'], axis=1)
            test_strategy = FISHER_bounds_strategy_opt(train, zone_low=bf, zone_high=sf)
            dummy_signal = test_strategy.generate_signals()
            dummy = test_strategy.signal_performance(10000, 6)
            if i==0:
                strat_sig_returns = dummy['S_Return'].to_frame().rename(columns={'S_Return': f'Strategy{i + 1}'}).set_index(dummy["Date"])
                strat_sig = dummy_signal['signal'].to_frame().rename(columns={'signal': f'Strategy{i + 1}'}).set_index(dummy_signal["Date"])
                #fisher_test = temp["fisher"].to_frame().rename(columns={"fisher": f'Fisher{asset}{i + 1}'}).set_index(temp["Date"])
            else:
                strat_sig_returns = pd.merge(strat_sig_returns, (dummy['S_Return'].to_frame().rename(columns={'S_Return': f'Strategy{i + 1}'}).set_index(dummy["Date"])), left_index=True, right_index=True)
                strat_sig = pd.concat([strat_sig, (dummy_signal['signal'].to_frame().rename(columns={'signal': f'Strategy{i + 1}'}).set_index(dummy_signal["Date"]))], axis = 1)
                #fisher_test = pd.concat([fisher_test, (temp["fisher"].to_frame().rename(columns={'fisher': f'Fisher{asset}{i + 1}'}).set_index(temp["Date"]))], axis = 1)
            #strat_sig_returns = pd.merge(strat_sig_returns,dummy['S_Return'].to_frame().rename(columns = {'S_Return':f'Strategy{i + 1}'}).set_index(dummy["Date"]), left_index=True, right_index=True)
        #return dummy
        return strat_sig_returns, strat_sig#, fisher_test
    else:
        return pd.DataFrame(), pd.DataFrame()

def strategy_selection(returns, corr_mat, num_strat, first_selected_strategy):
    strategies = [column for column in returns]
    selected_strategies = [first_selected_strategy]
    strategies.remove(first_selected_strategy)
    last_selected_strategy = first_selected_strategy

    while len(selected_strategies) < num_strat:
        corrs = corr_mat.loc[strategies][last_selected_strategy]
        corrs = corrs.loc[corrs>0.9]
        strategies = [st for st in strategies if st not in corrs.index.to_list()]

        if len(strategies)==0:
            break

        strat = strategies[0]

        selected_strategies.append(strat)
        strategies.remove(strat)
        last_selected_strategy = strat

    return selected_strategies

def selected_params(selected_strategies, res):
    selected_params = []
    for strategy in selected_strategies:
        selected_params.append(
            {"Name": strategy, "Lookback": res.iloc[int(strategy[8:])-1]["Lookback"],
             "Low Bound": res.iloc[int(strategy[8:])-1]["Low Bound"],
             "High Bound": res.iloc[int(strategy[8:])-1]["High Bound"],
             #"Sortino": res.iloc[int(strategy[8:])-1]["Sortino"],
             "AvgWinLoss": res.iloc[int(strategy[8:])-1]["AvgWinLoss"],
             "Optimization_Years": res.iloc[int(strategy[8:])-1]["Optimization_Years"]})
    selected_params = pd.DataFrame(selected_params)
    return selected_params

class FISHER_bounds_strategy_opt:

    def __init__(self, data, zone_low, zone_high, start=None, end=None):
        self.zl = zone_low
        self.zh = zone_high
        self.data = data  # the dataframe
        #self.data['yr'] = self.data['Date'].dt.year
        #self.n_days = (self.data.Date.iloc[-1] - self.data.Date.iloc[0])
        self.start = start  # the beginning date of the sample period
        self.end = end  # the ending date of the sample period

    def generate_signals(self, charts=True):
        self.data = self.data.loc[(self.data.fisher != 0)]
        self.data["fisher_lag"] = self.data.fisher.shift(1)
        self.data["lb"] = self.zl
        self.data["ub"] = self.zh
        self.data.dropna()
        self.data.reset_index(inplace=True)
        self.data = self.data.drop(['index'], axis=1)

        ## creating signal
        ##Upper and lower bounds
        buy_mask = (self.data["fisher"] > self.data["lb"]) & (self.data["fisher_lag"] < self.data["lb"])
        sell_mask = ((self.data["fisher"] < self.data["ub"]) & (self.data["fisher_lag"] > self.data["ub"])) | (
                    self.data["fisher"] < np.minimum(self.data["lb"], self.data["ub"]))

        bval = +1
        sval = 0  # -1 if short selling is allowed, otherwise 0

        self.data['signal_bounds'] = np.nan
        self.data.loc[buy_mask, 'signal_bounds'] = bval
        self.data.loc[sell_mask, 'signal_bounds'] = sval
        # initialize with long
        self.data["signal_bounds"][0] = 1

        self.data.signal_bounds = self.data.signal_bounds.fillna(method="ffill")

        self.data["signal"] = self.data.signal_bounds

        # Closing positions at end of time period
        self.data["signal"][-1:] = 0

        mask = ((self.data.signal == 1) & (self.data.signal.shift(1) == 0)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

        return self.data[["Date", "signal"]]

    def signal_performance(self, allocation, interest_rate):
        """
        Another instance method
        """
        self.allocation = allocation
        self.int = interest_rate

        # creating returns and portfolio value series
        self.data['Return'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['S_Return'] = self.data['signal'].shift(1) * self.data['Return']
        self.data['S_Return'] = self.data['S_Return'].fillna(0)
        self.data["S_Return"] = self.data["S_Return"] + (self.int / 25200) * (1 - self.data['signal'].shift(1))
        self.data['Market_Return'] = self.data['Return'].expanding().sum()
        self.data['Strategy_Return'] = self.data['S_Return'].expanding().sum()
        self.data['Portfolio Value'] = ((self.data['Strategy_Return'] + 1) * self.allocation)
        self.data['Wins'] = np.where(self.data['S_Return'] > 0, 1, 0)
        self.data['Losses'] = np.where(self.data['S_Return'] < 0, 1, 0)

        return self.data[['Date', 'Close', 'signal', 'S_Return', 'trade_num', 'Return']]

class FISHER_MCMC:

    def __init__(self, data, signals, start=None, end=None):

        self.signals = signals
        self.data = data  # the dataframe
        self.start = start  # the beginning date of the sample period
        self.end = end  # the ending date of the sample period

    def generate_signals(self, charts=True):

        self.data["signal"] = self.signals

        # Closing positions at end of time period
        self.data["signal"][-1:] = 0

        mask = ((self.data.signal == 1) & (self.data.signal.shift(1)==0)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

    def signal_performance(self, allocation, interest_rate):

        self.allocation = allocation
        self.int = interest_rate
        self.data = self.data.reset_index().rename(columns={'index':'Date'})
        # self.data['yr'] = self.data['Date'].dt.year
        # self.n_days = (self.data.Date.iloc[-1] - self.data.Date.iloc[0])

        # creating returns and portfolio value series
        self.data['Return'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['S_Return'] = self.data['signal'].shift(1) * self.data['Return']
        self.data['S_Return'] = self.data['S_Return'].fillna(0)
        self.data['Return'] = self.data['Return'].fillna(0)
        self.data["S_Return"] = self.data["S_Return"] + (self.int/25200)*(1-self.data['signal'].shift(1))
        self.data['Market_Return'] = self.data['Return'].expanding().sum()
        self.data['Strategy_Return'] = self.data['S_Return'].expanding().sum()
        self.data['Portfolio Value'] = ((self.data['Strategy_Return'] + 1) * self.allocation)
        self.data['Wins'] = np.where(self.data['S_Return'] > 0, 1, 0)
        self.data['Losses'] = np.where(self.data['S_Return'] < 0, 1, 0)

        return self.data[['Date', 'Close', 'signal', 'S_Return', 'trade_num', 'Return']]

def select_all_strategies(train_monthsf, datesf, temp_ogf, ticker, save=True):
    inputs =[]
    for date_i in range(len(datesf)-(int(train_monthsf/3)+1)):
        inputs.append([date_i, datesf, temp_ogf, train_monthsf])
    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results = pool.map(get_strategies_brute_force, inputs)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    res_test = [pd.DataFrame(columns=["Lookback", "Low Bound", "High Bound", "AvgWinLoss",\
                                       "Optimization_Years"])] * (len(datesf)-(int(24/3)+1))
    for i in range(len(results)):
        res_test[results[i][0]+int((train_monthsf-24)/3)] = pd.concat([res_test[results[i][0]],results[i][1].reset_index().drop(['index'], axis=1)], axis=0)

    if save==True:
        with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_{int(train_monthsf/12)}_All_Strategies.pkl', 'wb') as file:
            pickle.dump(res_test, file)

    return res_test

def select_strategies_from_corr_filter(res_testf2,res_testf4,res_testf8, datesf, temp_ogf, num_opt_periodsf,num_strategiesf, ticker, save=True):
    train_monthsf = 24  #minimum optimization lookback
    res_total = [None]*(len(datesf)-(int(train_monthsf/3)+1))
    for i in range(len(datesf)-(int(train_monthsf/3)+1)):
        if num_opt_periodsf==1:
            res_total[i] = pd.concat([res_testf2[i]], axis = 0)
        if num_opt_periodsf==2:
            res_total[i] = pd.concat([res_testf2[i],res_testf4[i]], axis=0)
        if num_opt_periodsf==3:
            res_total[i] = pd.concat([res_testf2[i],res_testf4[i],res_testf8[i]], axis=0)
        res_total[i] = res_total[i].reset_index().drop(['index'], axis=1)

    ss_test = [None]*(len(datesf)-(int(train_monthsf/3)+1))
    res_test = [None]*(len(datesf)-(int(train_monthsf/3)+1))
    inputs = []
    for date_i in range(len(datesf)-(int(train_monthsf/3)+1)):
        inputs.append([date_i, datesf, temp_ogf,res_total, num_strategiesf,train_monthsf])
    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results_filtered = pool.map(corr_sortino_filter, inputs)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    for i in range(len(datesf)-(int(train_monthsf/3)+1)):
        ss_test[results_filtered[i][0]] = results_filtered[i][1]
        res_test[results_filtered[i][0]] = results_filtered[i][2]

    if save==True:
        with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{num_opt_periodsf}_Selected_Strategies_ss.pkl', 'wb') as file:
            pickle.dump(ss_test, file)
        with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{num_opt_periodsf}_Selected_Strategies_res.pkl', 'wb') as file:
            pickle.dump(res_test, file)

    return ss_test, res_test

def SendMail(ticker, sortby, ImgFileNameList):
    msg = MIMEMultipart()
    msg['Subject'] = f'{ticker} Top 3 strategies sorted by {sortby}'
    msg['From'] = 'algo_notifications@acsysindia.com'
    msg['Cc'] = 'suprabhashsahu@acsysindia.com'   #, aditya@shankar.biz
    msg['To'] = 'algo_notifications@acsysindia.com'

    text = MIMEText(f'{ticker} Top 3 strategies sorted by {sortby}')
    msg.attach(text)
    for ImgFileName in ImgFileNameList:
        with open(ImgFileName, 'rb') as f:
            img_data = f.read()
        image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
        msg.attach(image)

    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login('algo_notifications@acsysindia.com', 'esahYah8')
    s.sendmail('algo_notifications@acsysindia.com', ['algo_notifications@acsysindia.com', 'suprabhashsahu@acsysindia.com'], msg.as_string())  #, 'aditya@shankar.biz'
    s.quit()

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))

if __name__ == '__main__':

    ticker = ticker_inp

    if not os.path.exists(f'{ticker}/SelectedStrategies'):
        os.makedirs(f'{ticker}/SelectedStrategies')

    print(f"Processing {ticker}")
    temp_og = get_data(ticker, "investpy")
    dates = valid_dates(pd.date_range(start=str(temp_og.iloc[1]['Date'] + timedelta(days=365))[:10], end="2024-06-15", freq=f'3M'))
    res_test2 = select_all_strategies(24,dates, temp_og, ticker,save=True)
    res_test4 = select_all_strategies(48,dates, temp_og, ticker,save=True)
    res_test8 = select_all_strategies(96,dates, temp_og, ticker,save=True)
    ss_test, res_test = select_strategies_from_corr_filter(res_test2,res_test4,res_test8, dates, temp_og, 1,10, ticker, save=True)
    ss_test, res_test = select_strategies_from_corr_filter(res_test2,res_test4,res_test8, dates, temp_og, 2,10, ticker, save=True)
    ss_test, res_test = select_strategies_from_corr_filter(res_test2,res_test4,res_test8, dates, temp_og, 3,10, ticker, save=True)

    for number_of_optimization_periods in [1]: #1,2,3
        for recalib_months in [3]:  #3,6,12
            for num_strategies in [1]: #1,3,5,7
                for metric in ['rolling_sharpe']: #

                    if not os.path.exists(f'{ticker}/equity_curves'):
                        os.makedirs(f'{ticker}/equity_curves')

                    if not os.path.exists(f'{ticker}/csv_files'):
                        os.makedirs(f'{ticker}/csv_files')

                    if not os.path.exists(f'{ticker}/weights'):
                        os.makedirs(f'{ticker}/weights')

                    if path.exists(f"{ticker}\csv_files\Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.csv"):
                        print("Already processed")
                        continue

                    #try:
                    temp_og = get_data(ticker, "investpy")
                    dates = valid_dates(pd.date_range(start=str(temp_og.iloc[1]['Date'] + timedelta(days=365))[:10],
                                                      end="2024-06-15", freq=f'3M'))

                    temp_og["Date"] = pd.to_datetime(temp_og["Date"])

                    with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_ss.pkl', 'rb') as file:
                        ss_test_imp = pickle.load(file)
                    with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_res.pkl', 'rb') as file:
                        res_test_imp = pickle.load(file)

                    res_test = []
                    ss_test = []
                    datesb = []
                    for date_i in range(len(dates) - (int(24 / 3) + 1)):
                        if (3 * date_i) % recalib_months == 0:
                            datesb.append(dates[date_i + int(24 / 3)])
                            ss_test.append(ss_test_imp[date_i])
                            res_test.append(res_test_imp[date_i])

                    datesb.append(date.today())
                    inputs = []
                    for date_i in range(len(datesb)-1):
                        inputs.append([date_i, datesb, temp_og, ss_test, res_test, num_strategies, metric, recalib_months,dates])
                    try:
                        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
                        results_multi = pool.map(optimize_weights_and_backtest, inputs)
                    finally: # To make sure processes are closed in the end, even if errors happen
                        pool.close()
                        pool.join()

                    weights = [None] * (len(datesb)-1)
                    for date_i in range(len(datesb)-1):
                        weights[results_multi[date_i][0]] = results_multi[date_i][2]

                    with open(f"{ticker}/weights/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.pkl",'wb') as file:
                        pickle.dump(weights, file)

                    results_final = pd.DataFrame()
                    for tt in results_multi:
                        results_final = pd.concat([results_final, tt[1]], axis=0)

                    temp_res = results_final
                    temp_res['Return'] = np.log(temp_res['Close'] / temp_res['Close'].shift(1))
                    temp_res['Market_Return'] = temp_res['Return'].expanding().sum()
                    temp_res['Strategy_Return'] = temp_res['S_Return'].expanding().sum()
                    temp_res['Portfolio Value'] = ((temp_res['Strategy_Return'] + 1) * 10000)
                    temp_res.reset_index(inplace=True)

                    temp_res.to_csv(f"{ticker}/csv_files/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.csv")
                    # except:
                    #     print(f"Could not process for Ticker: {ticker} Number of Optimization Periods: {number_of_optimization_periods}, recalib_months: {recalib_months}, num_strategies: {num_strategies}, metric: {metric}")
    res = []
    for number_of_optimization_periods in [1, 2, 3]:
        for recalib_months in [3, 6, 12]:
            for num_strategies in [1, 3, 5, 7]:
                for metric in ['rolling_sharpe', 'rolling_sortino', 'rolling_cagr', 'maxdrawup_by_maxdrawdown',
                               'outperformance']:
                    try:
                        temp_res = pd.read_csv(
                            f"{ticker}/csv_files/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.csv",
                            parse_dates=True)
                        temp_res['Date'] = pd.to_datetime(temp_res['Date'])
                        plt.plot(temp_res['Date'], temp_res['Market_Return'], color='black', label='Market Returns')
                        plt.plot(temp_res['Date'], temp_res['Strategy_Return'], color='blue', label='Strategy Returns')
                        plt.title('Strategy Backtest')
                        plt.legend(loc=0)
                        plt.tight_layout()
                        plt.savefig(
                            f"{ticker}/equity_curves/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.jpg")
                        plt.clf()
                        res.append({'Ticker': ticker, "Optimization Periods": number_of_optimization_periods,
                                    "Recalibration Months": recalib_months, "Number of Strategies": num_strategies,
                                    "Metric": metric, "Sortino": backtest_sortino(temp_res, 0, 0),
                                    "Sharpe": backtest_sharpe(temp_res, 0, 0),
                                    "Rolling Sortino": backtest_rolling_sortino(temp_res, 0, 0),
                                    "Rolling Sharpe": backtest_rolling_sharpe(temp_res, 0, 0),
                                    "Rollling CAGR": backtest_rolling_cagr(temp_res, 0, 0),
                                    "MaxDrawupByMaxDrawdown": backtest_maxdrawup_by_maxdrawdown(temp_res, 0, 0),
                                    "Outperformance": backtest_outperformance(temp_res, 0, 0)})
                    except:
                        print("Not processed")

    pd.DataFrame(res).to_csv(f"{ticker}/Results_Parametric.csv")

    #Emailer for top3 strategies
    for sortby in ["Outperformance"]:   #"Outperformance", "Sharpe", "MaxDrawupByMaxDrawdown"
        res_sorted = pd.DataFrame(res).sort_values(sortby, ascending=False)
        for i in range(1):   #3
            number_of_optimization_periods = res_sorted.iloc[i]["Optimization Periods"]
            recalib_months = res_sorted.iloc[i]["Recalibration Months"]
            num_strategies = res_sorted.iloc[i]["Number of Strategies"]
            metric = res_sorted.iloc[i]["Metric"]
            temp_res = pd.read_csv(
                f"{ticker}/csv_files/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.csv",
                parse_dates=True)
            temp_res['Date'] = pd.to_datetime(temp_res['Date'])
            plt.plot(temp_res['Date'], temp_res['Market_Return'], color='black', label='Market Returns')
            plt.plot(temp_res['Date'], temp_res['Strategy_Return'], color='blue', label='Strategy Returns')
            plt.title('Strategy Backtest')
            plt.legend(loc=0)
            plt.tight_layout()
            plt.savefig(
                f"{ticker}/SortedBy_{sortby}_{(i+1)}_Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.jpg")
            plt.clf()
        path_mail = f"{ticker}"
        files = os.listdir(path_mail)
        images = []
        for file in files:
            if file.startswith(f"SortedBy_{sortby}") & file.endswith('.jpg'):
                img_path = path_mail + '/' + file
                images.append(img_path)
        SendMail(ticker, sortby, images)

    zipf = zipfile.ZipFile(f'{ticker}.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir(f"{ticker}/", zipf)
    zipf.close()