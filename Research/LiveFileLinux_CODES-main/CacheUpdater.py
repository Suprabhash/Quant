import datetime
import pickle
import os
import pandas as pd
import yfinance as yf
import investpy
import math
import numpy as np
from datetime import timedelta
from datetime import date

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
def add_fisher(temp):
    for f_look in range(50, 400, 20):
        temp[f'Fisher{f_look}'] = fisher(temp, f_look)
    return temp
def get_data_investpy( symbol, country, from_date, to_date ):
    find = investpy.search.search_quotes(text=symbol, products=["stocks", "etfs", "indices", "currencies"])
    for f in find:
        #print( f )
        if f.symbol.lower() == symbol.lower() and f.country.lower() == country.lower():
            break
    if f.symbol.lower() != symbol.lower():
        return None
    ret = f.retrieve_historical_data(from_date=from_date, to_date=to_date)
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

    if ret is None:
        try:
            ret = investpy.currency_crosses.get_currency_cross_historical_data(currency_cross=symbol,
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
        temp_og = get_data_investpy(symbol=ticker, country='india', from_date="01/07/2007",to_date=(date.today()+timedelta(1)).strftime("%d/%m/%Y"))
        temp_og.reset_index(inplace=True)
        temp_og = add_fisher(temp_og)


    return temp_og

def valid_dates(dates_all):
    dates = []
    i = 0
    while True:
        dates.append(dates_all[i])
        if dates_all[i] > pd.to_datetime(date.today()):
            break
        i = i + 1
    return dates

constituent_alpha_params = {'ROST': {'ticker_yfinance': 'ROST',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 6,
                                             'num_strategies': 7,
                                             'metric': 'rolling_sharpe'},
                                    'MNST': {'ticker_yfinance': 'MNST',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 3,
                                             'num_strategies': 5,
                                             'metric': 'rolling_sharpe'},
                                    'CMCSA': {'ticker_yfinance': 'CMCSA',
                                              'number_of_optimization_periods': 3,
                                              'recalib_months': 12,
                                              'num_strategies': 7,
                                              'metric': 'rolling_cagr'},
                                    'KLAC': {'ticker_yfinance': 'KLAC',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 3,
                                             'num_strategies': 5,
                                             'metric': 'rolling_sortino'},
                                    'NXPI': {'ticker_yfinance': 'NXPI',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 3,
                                             'num_strategies': 5,
                                             'metric': 'rolling_sortino'},
                                    'SHPG': {'ticker_yfinance': 'SHPG',
                                             'number_of_optimization_periods': 0,
                                             'recalib_months': 0,
                                             'num_strategies': 0,
                                             'metric': ''},
                                    'XLNX': {'ticker_yfinance': 'XLNX',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 6,
                                             'num_strategies': 3,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'ALGN': {'ticker_yfinance': 'ALGN',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 3,
                                             'num_strategies': 5,
                                             'metric': 'rolling_cagr'},
                                    'MRVL': {'ticker_yfinance': 'MRVL',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 6,
                                             'num_strategies': 5,
                                             'metric': 'rolling_sharpe'},
                                    'ISRG': {'ticker_yfinance': 'ISRG',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 12,
                                             'num_strategies': 1,
                                             'metric': 'rolling_sharpe'},
                                    'MAT': {'ticker_yfinance': 'MAT',
                                            'number_of_optimization_periods': 1,
                                            'recalib_months': 3,
                                            'num_strategies': 7,
                                            'metric': 'maxdrawup_by_maxdrawdown'},
                                    'OKTA': {'ticker_yfinance': 'OKTA',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 3,
                                             'num_strategies': 3,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'AVGO': {'ticker_yfinance': 'AVGO',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 3,
                                             'num_strategies': 3,
                                             'metric': 'rolling_sharpe'},
                                    'DXCM': {'ticker_yfinance': 'DXCM',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 12,
                                             'num_strategies': 5,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'AMD': {'ticker_yfinance': 'AMD',
                                            'number_of_optimization_periods': 3,
                                            'recalib_months': 6,
                                            'num_strategies': 3,
                                            'metric': 'rolling_sortino'},
                                    'DOCU': {'ticker_yfinance': 'DOCU',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 3,
                                             'num_strategies': 7,
                                             'metric': 'rolling_cagr'},
                                    'INTC': {'ticker_yfinance': 'INTC',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 12,
                                             'num_strategies': 3,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'UAL': {'ticker_yfinance': 'UAL',
                                            'number_of_optimization_periods': 1,
                                            'recalib_months': 6,
                                            'num_strategies': 7,
                                            'metric': 'rolling_cagr'},
                                    'KDP': {'ticker_yfinance': 'KDP',
                                            'number_of_optimization_periods': 3,
                                            'recalib_months': 6,
                                            'num_strategies': 5,
                                            'metric': 'maxdrawup_by_maxdrawdown'},
                                    'WBA': {'ticker_yfinance': 'WBA',
                                            'number_of_optimization_periods': 3,
                                            'recalib_months': 3,
                                            'num_strategies': 7,
                                            'metric': 'rolling_cagr'},
                                    'CSCO': {'ticker_yfinance': 'CSCO',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 6,
                                             'num_strategies': 7,
                                             'metric': 'rolling_sortino'},
                                    'SIRI': {'ticker_yfinance': 'SIRI',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 12,
                                             'num_strategies': 1,
                                             'metric': 'outperformance'},
                                    'LRCX': {'ticker_yfinance': 'LRCX',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 3,
                                             'num_strategies': 3,
                                             'metric': 'rolling_sharpe'},
                                    'GILD': {'ticker_yfinance': 'GILD',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 12,
                                             'num_strategies': 7,
                                             'metric': 'outperformance'},
                                    'ADP': {'ticker_yfinance': 'ADP',
                                            'number_of_optimization_periods': 3,
                                            'recalib_months': 6,
                                            'num_strategies': 5,
                                            'metric': 'outperformance'},
                                    'NLOK': {'ticker_yfinance': 'NLOK',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 3,
                                             'num_strategies': 5,
                                             'metric': 'rolling_sharpe'},
                                    'ADSK': {'ticker_yfinance': 'ADSK',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 12,
                                             'num_strategies': 1,
                                             'metric': 'rolling_sortino'},
                                    'AMZN': {'ticker_yfinance': 'AMZN',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 12,
                                             'num_strategies': 3,
                                             'metric': 'outperformance'},
                                    'QRTEA': {'ticker_yfinance': 'QRTEA',
                                              'number_of_optimization_periods': 1,
                                              'recalib_months': 12,
                                              'num_strategies': 7,
                                              'metric': 'outperformance'},
                                    'REGN': {'ticker_yfinance': 'REGN',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 6,
                                             'num_strategies': 7,
                                             'metric': 'outperformance'},
                                    'LBTYA': {'ticker_yfinance': 'LBTYA',
                                              'number_of_optimization_periods': 1,
                                              'recalib_months': 12,
                                              'num_strategies': 5,
                                              'metric': 'outperformance'},
                                    'TMUS': {'ticker_yfinance': 'TMUS',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 6,
                                             'num_strategies': 3,
                                             'metric': 'rolling_sortino'},
                                    'LULU': {'ticker_yfinance': 'LULU',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 3,
                                             'num_strategies': 5,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'SGEN': {'ticker_yfinance': 'SGEN',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 6,
                                             'num_strategies': 7,
                                             'metric': 'rolling_sharpe'},
                                    'MDLZ': {'ticker_yfinance': 'MDLZ',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 12,
                                             'num_strategies': 1,
                                             'metric': 'rolling_sortino'},
                                    'INCY': {'ticker_yfinance': 'INCY',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 12,
                                             'num_strategies': 3,
                                             'metric': 'rolling_sortino'},
                                    'TCOM': {'ticker_yfinance': 'TCOM',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 6,
                                             'num_strategies': 5,
                                             'metric': 'rolling_sharpe'},
                                    'STX': {'ticker_yfinance': 'STX',
                                            'number_of_optimization_periods': 1,
                                            'recalib_months': 12,
                                            'num_strategies': 7,
                                            'metric': 'rolling_sharpe'},
                                    'CDNS': {'ticker_yfinance': 'CDNS',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 3,
                                             'num_strategies': 1,
                                             'metric': 'rolling_sharpe'},
                                    'NTAP': {'ticker_yfinance': 'NTAP',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 3,
                                             'num_strategies': 7,
                                             'metric': 'outperformance'},
                                    'HAS': {'ticker_yfinance': 'HAS',
                                            'number_of_optimization_periods': 3,
                                            'recalib_months': 12,
                                            'num_strategies': 5,
                                            'metric': 'rolling_sharpe'},
                                    'CHTR': {'ticker_yfinance': 'CHTR',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 6,
                                             'num_strategies': 5,
                                             'metric': 'outperformance'},
                                    'ILMN': {'ticker_yfinance': 'ILMN',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 3,
                                             'num_strategies': 3,
                                             'metric': 'rolling_sortino'},
                                    'SBUX': {'ticker_yfinance': 'SBUX',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 12,
                                             'num_strategies': 7,
                                             'metric': 'rolling_sortino'},
                                    'PYPL': {'ticker_yfinance': 'PYPL',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 12,
                                             'num_strategies': 7,
                                             'metric': 'rolling_sharpe'},
                                    'EBAY': {'ticker_yfinance': 'EBAY',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 3,
                                             'num_strategies': 3,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'AMGN': {'ticker_yfinance': 'AMGN',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 3,
                                             'num_strategies': 5,
                                             'metric': 'rolling_cagr'},
                                    'TEAM': {'ticker_yfinance': 'TEAM',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 3,
                                             'num_strategies': 3,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'MCHP': {'ticker_yfinance': 'MCHP',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 6,
                                             'num_strategies': 5,
                                             'metric': 'rolling_sortino'},
                                    'BIDU': {'ticker_yfinance': 'BIDU',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 3,
                                             'num_strategies': 7,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'NCLH': {'ticker_yfinance': 'NCLH',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 3,
                                             'num_strategies': 3,
                                             'metric': 'rolling_sharpe'},
                                    'EA': {'ticker_yfinance': 'EA',
                                           'number_of_optimization_periods': 3,
                                           'recalib_months': 6,
                                           'num_strategies': 1,
                                           'metric': 'rolling_sortino'},
                                    'XEL': {'ticker_yfinance': 'XEL',
                                            'number_of_optimization_periods': 1,
                                            'recalib_months': 3,
                                            'num_strategies': 7,
                                            'metric': 'rolling_sortino'},
                                    'CERN': {'ticker_yfinance': 'CERN',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 12,
                                             'num_strategies': 5,
                                             'metric': 'rolling_cagr'},
                                    'CDW': {'ticker_yfinance': 'CDW',
                                            'number_of_optimization_periods': 2,
                                            'recalib_months': 6,
                                            'num_strategies': 1,
                                            'metric': 'rolling_sortino'},
                                    'AMAT': {'ticker_yfinance': 'AMAT',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 3,
                                             'num_strategies': 5,
                                             'metric': 'outperformance'},
                                    'CPRT': {'ticker_yfinance': 'CPRT',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 12,
                                             'num_strategies': 7,
                                             'metric': 'outperformance'},
                                    'BKNG': {'ticker_yfinance': 'BKNG',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 6,
                                             'num_strategies': 5,
                                             'metric': 'rolling_sharpe'},
                                    'CTSH': {'ticker_yfinance': 'CTSH',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 12,
                                             'num_strategies': 1,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'AEP': {'ticker_yfinance': 'AEP',
                                            'number_of_optimization_periods': 2,
                                            'recalib_months': 12,
                                            'num_strategies': 7,
                                            'metric': 'maxdrawup_by_maxdrawdown'},
                                    'CHKP': {'ticker_yfinance': 'CHKP',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 3,
                                             'num_strategies': 5,
                                             'metric': 'rolling_sortino'},
                                    'PEP': {'ticker_yfinance': 'PEP',
                                            'number_of_optimization_periods': 3,
                                            'recalib_months': 6,
                                            'num_strategies': 5,
                                            'metric': 'rolling_sortino'},
                                    'FB': {'ticker_yfinance': 'FB',
                                           'number_of_optimization_periods': 1,
                                           'recalib_months': 12,
                                           'num_strategies': 3,
                                           'metric': 'rolling_cagr'},
                                    'JD': {'ticker_yfinance': 'JD',
                                           'number_of_optimization_periods': 2,
                                           'recalib_months': 6,
                                           'num_strategies': 7,
                                           'metric': 'rolling_sortino'},
                                    'ANSS': {'ticker_yfinance': 'ANSS',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 6,
                                             'num_strategies': 7,
                                             'metric': 'rolling_sharpe'},
                                    'VTRS': {'ticker_yfinance': 'VTRS',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 12,
                                             'num_strategies': 3,
                                             'metric': 'rolling_sortino'},
                                    'INTU': {'ticker_yfinance': 'INTU',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 3,
                                             'num_strategies': 3,
                                             'metric': 'outperformance'},
                                    'LILA': {'ticker_yfinance': 'LILA',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 6,
                                             'num_strategies': 7,
                                             'metric': 'rolling_cagr'},
                                    'CSGP': {'ticker_yfinance': 'CSGP',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 3,
                                             'num_strategies': 7,
                                             'metric': 'rolling_sharpe'},
                                    'NVDA': {'ticker_yfinance': 'NVDA',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 12,
                                             'num_strategies': 5,
                                             'metric': 'rolling_sharpe'},
                                    'GOOGL': {'ticker_yfinance': 'GOOGL',
                                              'number_of_optimization_periods': 3,
                                              'recalib_months': 6,
                                              'num_strategies': 5,
                                              'metric': 'outperformance'},
                                    'VOD': {'ticker_yfinance': 'VOD',
                                            'number_of_optimization_periods': 3,
                                            'recalib_months': 12,
                                            'num_strategies': 7,
                                            'metric': 'maxdrawup_by_maxdrawdown'},
                                    'NFLX': {'ticker_yfinance': 'NFLX',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 12,
                                             'num_strategies': 5,
                                             'metric': 'rolling_sharpe'},
                                    'JBHT': {'ticker_yfinance': 'JBHT',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 6,
                                             'num_strategies': 1,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'XRAY': {'ticker_yfinance': 'XRAY',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 3,
                                             'num_strategies': 3,
                                             'metric': 'rolling_sharpe'},
                                    'DLTR': {'ticker_yfinance': 'DLTR',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 6,
                                             'num_strategies': 7,
                                             'metric': 'outperformance'},
                                    'VRTX': {'ticker_yfinance': 'VRTX',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 6,
                                             'num_strategies': 5,
                                             'metric': 'rolling_cagr'},
                                    'COST': {'ticker_yfinance': 'COST',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 3,
                                             'num_strategies': 3,
                                             'metric': 'outperformance'},
                                    'IDXX': {'ticker_yfinance': 'IDXX',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 3,
                                             'num_strategies': 3,
                                             'metric': 'rolling_cagr'},
                                    'TTWO': {'ticker_yfinance': 'TTWO',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 12,
                                             'num_strategies': 7,
                                             'metric': 'rolling_sortino'},
                                    'FISV': {'ticker_yfinance': 'FISV',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 3,
                                             'num_strategies': 1,
                                             'metric': 'rolling_sortino'},
                                    'AKAM': {'ticker_yfinance': 'AKAM',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 12,
                                             'num_strategies': 7,
                                             'metric': 'rolling_sharpe'},
                                    'ADBE': {'ticker_yfinance': 'ADBE',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 12,
                                             'num_strategies': 5,
                                             'metric': 'rolling_sharpe'},
                                    'NTES': {'ticker_yfinance': 'NTES',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 12,
                                             'num_strategies': 5,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'BIIB': {'ticker_yfinance': 'BIIB',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 6,
                                             'num_strategies': 5,
                                             'metric': 'rolling_cagr'},
                                    'SWKS': {'ticker_yfinance': 'SWKS',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 12,
                                             'num_strategies': 1,
                                             'metric': 'rolling_sharpe'},
                                    'SNPS': {'ticker_yfinance': 'SNPS',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 12,
                                             'num_strategies': 7,
                                             'metric': 'rolling_cagr'},
                                    'AAL': {'ticker_yfinance': 'AAL',
                                            'number_of_optimization_periods': 1,
                                            'recalib_months': 6,
                                            'num_strategies': 5,
                                            'metric': 'maxdrawup_by_maxdrawdown'},
                                    'EXC': {'ticker_yfinance': 'EXC',
                                            'number_of_optimization_periods': 3,
                                            'recalib_months': 3,
                                            'num_strategies': 1,
                                            'metric': 'rolling_cagr'},
                                    'DISH': {'ticker_yfinance': 'DISH',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 12,
                                             'num_strategies': 5,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'MU': {'ticker_yfinance': 'MU',
                                           'number_of_optimization_periods': 1,
                                           'recalib_months': 12,
                                           'num_strategies': 3,
                                           'metric': 'maxdrawup_by_maxdrawdown'},
                                    'VRSN': {'ticker_yfinance': 'VRSN',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 3,
                                             'num_strategies': 1,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'QCOM': {'ticker_yfinance': 'QCOM',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 12,
                                             'num_strategies': 3,
                                             'metric': 'outperformance'},
                                    'TSCO': {'ticker_yfinance': 'TSCO',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 6,
                                             'num_strategies': 7,
                                             'metric': 'outperformance'},
                                    'MELI': {'ticker_yfinance': 'MELI',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 3,
                                             'num_strategies': 5,
                                             'metric': 'rolling_sortino'},
                                    'HOLX': {'ticker_yfinance': 'HOLX',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 12,
                                             'num_strategies': 3,
                                             'metric': 'rolling_sortino'},
                                    'WYNN': {'ticker_yfinance': 'WYNN',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 6,
                                             'num_strategies': 3,
                                             'metric': 'rolling_sortino'},
                                    'EXPE': {'ticker_yfinance': 'EXPE',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 12,
                                             'num_strategies': 5,
                                             'metric': 'rolling_cagr'},
                                    'BMRN': {'ticker_yfinance': 'BMRN',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 3,
                                             'num_strategies': 3,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'FAST': {'ticker_yfinance': 'FAST',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 3,
                                             'num_strategies': 5,
                                             'metric': 'outperformance'},
                                    'ASML': {'ticker_yfinance': 'ASML',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 12,
                                             'num_strategies': 1,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'TSLA': {'ticker_yfinance': 'TSLA',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 12,
                                             'num_strategies': 7,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'KHC': {'ticker_yfinance': 'KHC',
                                            'number_of_optimization_periods': 3,
                                            'recalib_months': 12,
                                            'num_strategies': 3,
                                            'metric': 'rolling_cagr'},
                                    'MSFT': {'ticker_yfinance': 'MSFT',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 3,
                                             'num_strategies': 5,
                                             'metric': 'rolling_cagr'},
                                    'ORLY': {'ticker_yfinance': 'ORLY',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 3,
                                             'num_strategies': 5,
                                             'metric': 'outperformance'},
                                    'PAYX': {'ticker_yfinance': 'PAYX',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 12,
                                             'num_strategies': 7,
                                             'metric': 'rolling_sortino'},
                                    'CTXS': {'ticker_yfinance': 'CTXS',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 3,
                                             'num_strategies': 5,
                                             'metric': 'rolling_cagr'},
                                    'PCAR': {'ticker_yfinance': 'PCAR',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 3,
                                             'num_strategies': 7,
                                             'metric': 'outperformance'},
                                    'ULTA': {'ticker_yfinance': 'ULTA',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 12,
                                             'num_strategies': 7,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'CSX': {'ticker_yfinance': 'CSX',
                                            'number_of_optimization_periods': 1,
                                            'recalib_months': 6,
                                            'num_strategies': 5,
                                            'metric': 'rolling_sharpe'},
                                    'DISCA': {'ticker_yfinance': 'DISCA',
                                              'number_of_optimization_periods': 2,
                                              'recalib_months': 12,
                                              'num_strategies': 1,
                                              'metric': 'rolling_sharpe'},
                                    'WDC': {'ticker_yfinance': 'WDC',
                                            'number_of_optimization_periods': 3,
                                            'recalib_months': 3,
                                            'num_strategies': 7,
                                            'metric': 'rolling_cagr'},
                                    'ADI': {'ticker_yfinance': 'ADI',
                                            'number_of_optimization_periods': 3,
                                            'recalib_months': 6,
                                            'num_strategies': 1,
                                            'metric': 'rolling_sharpe'},
                                    'HSIC': {'ticker_yfinance': 'HSIC',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 12,
                                             'num_strategies': 7,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'TXN': {'ticker_yfinance': 'TXN',
                                            'number_of_optimization_periods': 3,
                                            'recalib_months': 3,
                                            'num_strategies': 7,
                                            'metric': 'rolling_sharpe'},
                                    'ATVI': {'ticker_yfinance': 'ATVI',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 12,
                                             'num_strategies': 7,
                                             'metric': 'rolling_cagr'},
                                    'AAPL': {'ticker_yfinance': 'AAPL',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 12,
                                             'num_strategies': 3,
                                             'metric': 'rolling_cagr'},
                                    'CTAS': {'ticker_yfinance': 'CTAS',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 3,
                                             'num_strategies': 3,
                                             'metric': 'outperformance'},
                                    'SPLK': {'ticker_yfinance': 'SPLK',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 6,
                                             'num_strategies': 5,
                                             'metric': 'rolling_cagr'},
                                    'HON': {'ticker_yfinance': 'HON',
                                            'number_of_optimization_periods': 3,
                                            'recalib_months': 3,
                                            'num_strategies': 7,
                                            'metric': 'maxdrawup_by_maxdrawdown'},
                                    'VRSK': {'ticker_yfinance': 'VRSK',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 12,
                                             'num_strategies': 5,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'PDD': {'ticker_yfinance': 'PDD',
                                            'number_of_optimization_periods': 2,
                                            'recalib_months': 12,
                                            'num_strategies': 5,
                                            'metric': 'rolling_sharpe'},
                                    'TRIP': {'ticker_yfinance': 'TRIP',
                                             'number_of_optimization_periods': 3,
                                             'recalib_months': 12,
                                             'num_strategies': 7,
                                             'metric': 'outperformance'},
                                    'MTCH': {'ticker_yfinance': 'MTCH',
                                             'number_of_optimization_periods': 2,
                                             'recalib_months': 12,
                                             'num_strategies': 5,
                                             'metric': 'maxdrawup_by_maxdrawdown'},
                                    'WDAY': {'ticker_yfinance': 'WDAY',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 6,
                                             'num_strategies': 7,
                                             'metric': 'rolling_sharpe'},
                                    'SBAC': {'ticker_yfinance': 'SBAC',
                                             'number_of_optimization_periods': 1,
                                             'recalib_months': 12,
                                             'num_strategies': 5,
                                             'metric': 'outperformance'},
                                    'TAMdv': {"ticker_yfinance": "TATAMTRDVR.NS", "number_of_optimization_periods": 1,
                                              "recalib_months": 12, "num_strategies": 5, "metric": 'rolling_sharpe'},
                                    'SBI': {"ticker_yfinance": "SBIN.NS", "number_of_optimization_periods": 3,
                                            "recalib_months": 3,
                                            "num_strategies": 7, "metric": 'rolling_sortino'},
                                    'NEST': {"ticker_yfinance": "NESTLEIND.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 6, "num_strategies": 5, "metric": 'rolling_sortino'},
                                    'INFY': {"ticker_yfinance": "INFY.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 12,
                                             "num_strategies": 5, "metric": 'outperformance'},
                                    'TCS': {"ticker_yfinance": "TCS.NS", "number_of_optimization_periods": 3,
                                            "recalib_months": 3,
                                            "num_strategies": 7, "metric": 'outperformance'},
                                    'COAL': {"ticker_yfinance": "COALINDIA.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 12, "num_strategies": 7, "metric": 'rolling_sortino'},
                                    'HCLT': {"ticker_yfinance": "HCLTECH.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 6,
                                             "num_strategies": 5, "metric": 'rolling_cagr'},
                                    'NTPC': {"ticker_yfinance": "NTPC.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 12,
                                             "num_strategies": 7, "metric": 'rolling_sortino'},
                                    'ICBK': {"ticker_yfinance": "ICICIBANK.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 6, "num_strategies": 7, "metric": 'rolling_sortino'},
                                    'LART': {"ticker_yfinance": "LT.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 12,
                                             "num_strategies": 7, "metric": 'rolling_sharpe'},
                                    'HDBK': {"ticker_yfinance": "HDFCBANK.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 12, "num_strategies": 5, "metric": 'rolling_cagr'},
                                    'TAMO': {"ticker_yfinance": "TATAMOTORS.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 12, "num_strategies": 5, "metric": 'outperformance'},
                                    'TISC': {"ticker_yfinance": "TATASTEEL.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 3, "num_strategies": 1, "metric": 'rolling_sortino'},
                                    'BAJA': {"ticker_yfinance": "BAJAJ-AUTO.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 3, "num_strategies": 7, "metric": 'outperformance'},
                                    'ASPN': {"ticker_yfinance": "ASIANPAINT.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 12, "num_strategies": 5, "metric": 'rolling_sortino'},
                                    'REDY': {"ticker_yfinance": "DRREDDY.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 12,
                                             "num_strategies": 7, "metric": 'rolling_cagr'},
                                    'TEML': {"ticker_yfinance": "TECHM.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 3,
                                             "num_strategies": 7, "metric": 'outperformance'},
                                    'CIPL': {"ticker_yfinance": "CIPLA.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 12,
                                             "num_strategies": 5, "metric": 'outperformance'},
                                    'ULTC': {"ticker_yfinance": "ULTRACEMCO.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 3, "num_strategies": 3, "metric": 'rolling_sharpe'},
                                    'BJFS': {"ticker_yfinance": "BAJAJFINSV.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 6, "num_strategies": 3, "metric": 'rolling_sortino'},
                                    'HDFC': {"ticker_yfinance": "HDFC.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 6,
                                             "num_strategies": 7, "metric": 'rolling_sharpe'},
                                    'SUN': {"ticker_yfinance": "SUNPHARMA.NS", "number_of_optimization_periods": 3,
                                            "recalib_months": 12, "num_strategies": 3, "metric": 'outperformance'},
                                    'ITC': {"ticker_yfinance": "ITC.NS", "number_of_optimization_periods": 2,
                                            "recalib_months": 3,
                                            "num_strategies": 5, "metric": 'rolling_sortino'},
                                    'WIPR': {"ticker_yfinance": "WIPRO.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 3,
                                             "num_strategies": 3, "metric": 'rolling_sharpe'},
                                    'GAIL': {"ticker_yfinance": "GAIL.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 12,
                                             "num_strategies": 1, "metric": 'rolling_sortino'},
                                    'VDAN': {"ticker_yfinance": "VEDL.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 12,
                                             "num_strategies": 1, "metric": 'maxdrawup_by_maxdrawdown'},
                                    'PGRD': {"ticker_yfinance": "POWERGRID.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 12, "num_strategies": 3, "metric": 'rolling_sortino'},
                                    'HROM': {"ticker_yfinance": "HEROMOTOCO.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 12, "num_strategies": 5, "metric": 'rolling_sortino'},
                                    'AXBK': {"ticker_yfinance": "AXISBANK.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 12, "num_strategies": 7, "metric": 'outperformance'},
                                    'YESB': {"ticker_yfinance": "YESBANK.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 12,
                                             "num_strategies": 5, "metric": 'maxdrawup_by_maxdrawdown'},
                                    'ONGC': {"ticker_yfinance": "ONGC.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 3,
                                             "num_strategies": 5, "metric": 'maxdrawup_by_maxdrawdown'},
                                    'HLL': {"ticker_yfinance": "HINDUNILVR.NS", "number_of_optimization_periods": 2,
                                            "recalib_months": 12,
                                            "num_strategies": 1, "metric": 'rolling_sharpe'},
                                    'APSE': {"ticker_yfinance": "ADANIPORTS.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 3,
                                             "num_strategies": 5, "metric": 'outperformance'},
                                    'BRTI': {"ticker_yfinance": "BHARTIARTL.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 12,
                                             "num_strategies": 5, "metric": 'maxdrawup_by_maxdrawdown'},
                                    'VODA': {"ticker_yfinance": "IDEA.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 12,
                                             "num_strategies": 5, "metric": 'rolling_sortino'},
                                    'BFRG': {"ticker_yfinance": "BHARATFORG.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 3,
                                             "num_strategies": 7, "metric": 'rolling_sortino'},
                                    'CUMM': {"ticker_yfinance": "CUMMINSIND.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 3,
                                             "num_strategies": 1, "metric": 'outperformance'},
                                    'CAST': {"ticker_yfinance": "CASTROLIND.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 12,
                                             "num_strategies": 3, "metric": 'rolling_sortino'},
                                    'ASOK': {"ticker_yfinance": "ASHOKLEY.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 12,
                                             "num_strategies": 3, "metric": 'rolling_sharpe'},
                                    'AUFI': {"ticker_yfinance": "AUBANK.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 12,
                                             "num_strategies": 7, "metric": 'rolling_cagr'},
                                    'SRTR': {"ticker_yfinance": "SRTRANSFIN.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 6,
                                             "num_strategies": 5, "metric": 'rolling_cagr'},
                                    'MAXI': {"ticker_yfinance": "MFSL.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 12,
                                             "num_strategies": 3, "metric": 'maxdrawup_by_maxdrawdown'},
                                    'BATA': {"ticker_yfinance": "BATAINDIA.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 12,
                                             "num_strategies": 5, "metric": 'maxdrawup_by_maxdrawdown'},
                                    'MINT': {"ticker_yfinance": "MINDTREE.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 6,
                                             "num_strategies": 7, "metric": 'maxdrawup_by_maxdrawdown'},
                                    'COFO': {"ticker_yfinance": "COFORGE.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 3,
                                             "num_strategies": 7, "metric": 'rolling_cagr'},
                                    'TVSM': {"ticker_yfinance": "TVSMOTOR.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 12,
                                             "num_strategies": 5, "metric": 'rolling_sharpe'},
                                    'PAGE': {"ticker_yfinance": "PAGEIND.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 3,
                                             "num_strategies": 3, "metric": 'maxdrawup_by_maxdrawdown'},
                                    'CCRI': {"ticker_yfinance": "CONCOR.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 6,
                                             "num_strategies": 5, "metric": 'rolling_cagr'},
                                    'ESCO': {"ticker_yfinance": "ESCORTS.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 3,
                                             "num_strategies": 7, "metric": 'rolling_cagr'},
                                    'SRFL': {"ticker_yfinance": "SRF.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 6,
                                             "num_strategies": 5, "metric": 'maxdrawup_by_maxdrawdown'},
                                    'CNBK': {"ticker_yfinance": "CANBK.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 6,
                                             "num_strategies": 7, "metric": 'maxdrawup_by_maxdrawdown'},
                                    'TTPW': {"ticker_yfinance": "TATAPOWER.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 12,
                                             "num_strategies": 5, "metric": 'rolling_sharpe'},
                                    'ZEE': {"ticker_yfinance": "ZEEL.NS", "number_of_optimization_periods": 2,
                                            "recalib_months": 12,
                                            "num_strategies": 3, "metric": 'maxdrawup_by_maxdrawdown'},
                                    'MNFL': {"ticker_yfinance": "MANAPPURAM.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 12,
                                             "num_strategies": 3, "metric": 'maxdrawup_by_maxdrawdown'},
                                    'FED': {"ticker_yfinance": "FEDERALBNK.NS", "number_of_optimization_periods": 2,
                                            "recalib_months": 3,
                                            "num_strategies": 7, "metric": 'rolling_sharpe'},
                                    'GLEN': {"ticker_yfinance": "GLENMARK.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 12,
                                             "num_strategies": 7, "metric": 'rolling_cagr'},
                                    'CHLA': {"ticker_yfinance": "CHOLAFIN.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 3,
                                             "num_strategies": 3, "metric": 'maxdrawup_by_maxdrawdown'},
                                    'AMAR': {"ticker_yfinance": "AMARAJABAT.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 12,
                                             "num_strategies": 5, "metric": 'outperformance'},
                                    'APLO': {"ticker_yfinance": "APOLLOTYRE.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 6,
                                             "num_strategies": 3, "metric": 'maxdrawup_by_maxdrawdown'},
                                    'BAJE': {"ticker_yfinance": "BEL.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 6,
                                             "num_strategies": 1, "metric": 'rolling_sortino'},
                                    'SAIL': {"ticker_yfinance": "SAIL.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 3,
                                             "num_strategies": 1, "metric": 'rolling_cagr'},
                                    'MMFS': {"ticker_yfinance": "M&MFIN.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 12,
                                             "num_strategies": 7, "metric": 'rolling_cagr'},
                                    'BLKI': {"ticker_yfinance": "BALKRISIND.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 6,
                                             "num_strategies": 5, "metric": 'outperformance'},
                                    'PWFC': {"ticker_yfinance": "PFC.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 6,
                                             "num_strategies": 7, "metric": 'outperformance'},
                                    'TOPO': {"ticker_yfinance": "TORNTPOWER.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 12,
                                             "num_strategies": 1, "metric": 'outperformance'},
                                    'BOB': {"ticker_yfinance": "BANKBARODA.NS", "number_of_optimization_periods": 2,
                                            "recalib_months": 3,
                                            "num_strategies": 5, "metric": 'rolling_sortino'},
                                    'GODR': {"ticker_yfinance": "GODREJPROP.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 12,
                                             "num_strategies": 3, "metric": 'rolling_cagr'},
                                    'LTFH': {"ticker_yfinance": "L&TFH.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 12,
                                             "num_strategies": 3, "metric": 'rolling_sortino'},
                                    'INBF': {"ticker_yfinance": "IBULHSGFIN.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 3,
                                             "num_strategies": 1, "metric": 'rolling_cagr'},
                                    'BOI': {"ticker_yfinance": "BANKINDIA.NS", "number_of_optimization_periods": 3,
                                            "recalib_months": 3,
                                            "num_strategies": 7, "metric": 'maxdrawup_by_maxdrawdown'},
                                    'JNSP': {"ticker_yfinance": "JINDALSTEL.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 6,
                                             "num_strategies": 7, "metric": 'rolling_sortino'},
                                    'IDFB': {"ticker_yfinance": "IDFCFIRSTB.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 3,
                                             "num_strategies": 3, "metric": 'rolling_sharpe'},
                                    'SUTV': {"ticker_yfinance": "SUNTV.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 12,
                                             "num_strategies": 1, "metric": 'rolling_cagr'},
                                    'VOLT': {"ticker_yfinance": "VOLTAS.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 3,
                                             "num_strategies": 1, "metric": 'outperformance'},
                                    'MGAS': {"ticker_yfinance": "MGL.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 3,
                                             "num_strategies": 3, "metric": 'rolling_sortino'},
                                    'RECM': {"ticker_yfinance": "RECLTD.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 3,
                                             "num_strategies": 5, "metric": 'rolling_sortino'},
                                    'GMRI': {"ticker_yfinance": "GMRINFRA.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 6,
                                             "num_strategies": 7, "metric": 'outperformance'},
                                    'BHEL': {"ticker_yfinance": "BHEL.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 12,
                                             "num_strategies": 1, "metric": 'rolling_sortino'},
                                    'LICH': {"ticker_yfinance": "LICHSGFIN.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 6,
                                             "num_strategies": 7, "metric": 'rolling_sharpe'},
                                    'EXID': {"ticker_yfinance": "EXIDEIND.NS", "number_of_optimization_periods": 1,
                                             "recalib_months": 12,
                                             "num_strategies": 1, "metric": 'rolling_sharpe'},
                                    'TRCE': {"ticker_yfinance": "RAMCOCEM.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 6,
                                             "num_strategies": 5, "metric": 'rolling_sharpe'},
                                    'INBK': {"ticker_yfinance": "INDUSINDBK.NS", "number_of_optimization_periods": 3,
                                             "recalib_months": 3,
                                             "num_strategies": 3, "metric": 'rolling_sharpe'},
                                    'RELI': {"ticker_yfinance": "RELIANCE.NS", "number_of_optimization_periods": 2,
                                             "recalib_months": 3,
                                             "num_strategies": 1, "metric": 'rolling_sortino'},
                                    }

def recalibrate_indices(ticker, number_of_optimization_periods, recalibrate_what):
    #update indices
    print(f"{datetime.datetime.now()}" + " - recalibrating " + ticker + " " + recalibrate_what)
    if ticker == "^NSEI":
        number_of_optimization_periods = 1
        recalib_months = 3
        num_strategies = 5
        metric = 'outperformance'
    if ticker == "GOLDBEES.NS":
        number_of_optimization_periods = 2
        recalib_months = 6
        num_strategies = 1
        metric = 'outperformance'
    if ticker == "^NSEMDCP50":
        number_of_optimization_periods = 2
        recalib_months = 12
        num_strategies = 7
        metric = 'maxdrawup_by_maxdrawdown'
    if ticker == "^IXIC":
        number_of_optimization_periods = 2
        recalib_months = 3
        num_strategies = 5
        metric = 'maxdrawup_by_maxdrawdown'
    if ticker == "TLT":
        number_of_optimization_periods = 2
        recalib_months = 6
        num_strategies = 5
        metric = 'rolling_sharpe'

    cwd = os.getcwd()
    print(
        f"This is the path in CacheUpdater.py recalibration = {cwd}/{ticker}/SelectedStrategies/{ticker}_OptPeriods_2_Selected_Strategies_res.pkl")

    if number_of_optimization_periods == 1:
        with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'rb') as file:
            res_test2 = pickle.load(file)
        with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_OptPeriods_1_Selected_Strategies_res.pkl', 'rb') as file:
            res_test = pickle.load(file)
        with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_OptPeriods_1_Selected_Strategies_ss.pkl', 'rb') as file:
            ss_test = pickle.load(file)
        res_test4 = []
        res_test8 = []
    if number_of_optimization_periods == 2:
        with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'rb') as file:
            res_test2 = pickle.load(file)
        with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_TrainYrs_4_All_Strategies.pkl', 'rb') as file:
            res_test4 = pickle.load(file)
        with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_OptPeriods_2_Selected_Strategies_res.pkl', 'rb') as file:
            res_test = pickle.load(file)
        with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_OptPeriods_2_Selected_Strategies_ss.pkl', 'rb') as file:
            ss_test = pickle.load(file)

        res_test8 = []
    if number_of_optimization_periods == 3:
        with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'rb') as file:
            res_test2 = pickle.load(file)
        with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_TrainYrs_4_All_Strategies.pkl', 'rb') as file:
            res_test4 = pickle.load(file)
        with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_TrainYrs_8_All_Strategies.pkl', 'rb') as file:
            res_test8 = pickle.load(file)
        with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_OptPeriods_3_Selected_Strategies_res.pkl', 'rb') as file:
            res_test = pickle.load(file)
        with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_OptPeriods_3_Selected_Strategies_ss.pkl', 'rb') as file:
            ss_test = pickle.load(file)

    temp_og = get_data(ticker, "yfinance")
    dates = valid_dates(pd.date_range(start=str(temp_og.iloc[1]['Date'] + timedelta(days=365))[:10],
                                      end="2024-06-15", freq=f'3M'))
    datesb = []
    for date_i in range(len(dates) - (int(24 / 3) + 1)):
        if (3 * date_i) % recalib_months == 0:
            datesb.append(dates[date_i + int(24 / 3)])
    datesb.append(date.today())
    weights_supposed_to_be = [None] * (len(datesb)-1)

    with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/weights/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.pkl', 'rb') as file:
        weights = pickle.load(file)

    if number_of_optimization_periods == 1:
        with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'rb') as file:
            res_test2_imp = pickle.load(file)
        with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_OptPeriods_1_Selected_Strategies_res.pkl', 'rb') as file:
            res_test_imp = pickle.load(file)
        with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_OptPeriods_1_Selected_Strategies_ss.pkl', 'rb') as file:
            ss_test_imp = pickle.load(file)
        res_test4_imp = []
        res_test8_imp = []
    if number_of_optimization_periods == 2:
        with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'rb') as file:
            res_test2_imp = pickle.load(file)
        with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_TrainYrs_4_All_Strategies.pkl', 'rb') as file:
            res_test4_imp = pickle.load(file)
        with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_OptPeriods_2_Selected_Strategies_res.pkl', 'rb') as file:
            res_test_imp = pickle.load(file)
        with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_OptPeriods_2_Selected_Strategies_ss.pkl', 'rb') as file:
            ss_test_imp = pickle.load(file)
        res_test8_imp = []
    if number_of_optimization_periods == 3:
        with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'rb') as file:
            res_test2_imp = pickle.load(file)
        with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_TrainYrs_4_All_Strategies.pkl', 'rb') as file:
            res_test4_imp = pickle.load(file)
        with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_TrainYrs_8_All_Strategies.pkl', 'rb') as file:
            res_test8_imp = pickle.load(file)
        with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_OptPeriods_3_Selected_Strategies_res.pkl', 'rb') as file:
            res_test_imp = pickle.load(file)
        with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_OptPeriods_3_Selected_Strategies_ss.pkl', 'rb') as file:
            ss_test_imp = pickle.load(file)

    with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/weights/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.pkl', 'rb') as file:
        weights_imp = pickle.load(file)


    print("*"*100)
    print(ticker)
    print(f"len2: {len(res_test2)}")
    print(f"len4: {len(res_test4)}")
    print(f"len8: {len(res_test8)}")
    print(f"len_res_test: {len(res_test)}")
    print(f"len_ss_test: {len(ss_test)}")
    print(f"Weights: {len(weights)}")
    print(f"weights_supposed_to_be: {len(weights_supposed_to_be)}")
    print(f"len2_imp: {len(res_test2_imp)}")
    print(f"len4_imp: {len(res_test4_imp)}")
    print(f"len8_imp: {len(res_test8_imp)}")
    print(f"len_res_test_imp: {len(res_test_imp)}")
    print(f"len_ss_test_imp: {len(ss_test_imp)}")
    print(f"Weights_imp: {len(weights_imp)}")
    print("*"*100)

    if recalibrate_what=="strategies":
        if number_of_optimization_periods == 1:
            with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'wb') as file:
                pickle.dump((res_test2+res_test2_imp), file)
            with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_OptPeriods_1_Selected_Strategies_res.pkl', 'wb') as file:
                pickle.dump((res_test+res_test_imp), file)
            with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_OptPeriods_1_Selected_Strategies_ss.pkl', 'wb') as file:
                pickle.dump((ss_test+ss_test_imp), file)
            res_test4 = []
            res_test8 = []
        if number_of_optimization_periods == 2:
            with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'wb') as file:
                pickle.dump((res_test2+res_test2_imp), file)
            with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_TrainYrs_4_All_Strategies.pkl', 'wb') as file:
                pickle.dump((res_test4+res_test4_imp), file)
            with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_OptPeriods_2_Selected_Strategies_res.pkl', 'wb') as file:
                pickle.dump((res_test+res_test_imp), file)
            with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_OptPeriods_2_Selected_Strategies_ss.pkl', 'wb') as file:
                pickle.dump((ss_test+ss_test_imp),file)
            res_test8 = []
        if number_of_optimization_periods == 3:
            with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'wb') as file:
                pickle.dump((res_test2+res_test2_imp), file)
            with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_TrainYrs_4_All_Strategies.pkl', 'wb') as file:
                pickle.dump((res_test4+res_test4_imp), file)
            with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_TrainYrs_8_All_Strategies.pkl', 'wb') as file:
                pickle.dump((res_test8+res_test8_imp), file)
            with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_OptPeriods_3_Selected_Strategies_res.pkl', 'wb') as file:
                pickle.dump((res_test+res_test_imp), file)
            with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/SelectedStrategies/{ticker}_OptPeriods_3_Selected_Strategies_ss.pkl', 'wb') as file:
                pickle.dump((ss_test+ss_test_imp),file)

    print(len(weights))
    print(len(weights_supposed_to_be))

    if recalibrate_what=="weights":
        with open(f'/home/azurelinux/Desktop/LiveFiles/{ticker}/weights/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.pkl', 'wb') as file:
            pickle.dump((weights+[weights_imp[0][1]]), file)

def recalibrate_constituent_ticker(ticker, number_of_optimization_periods, recalibrate_what):
    print(str(datetime.datetime.now()) + " - recalibrating " + ticker + " " + recalibrate_what)
    ticker_list = [ticker]
    for ticker in list(constituent_alpha_params.keys()):
        if ticker not in ticker_list:
            continue

        number_of_optimization_periods = constituent_alpha_params[ticker]["number_of_optimization_periods"]
        recalib_months = constituent_alpha_params[ticker]["recalib_months"]
        num_strategies = constituent_alpha_params[ticker]["num_strategies"]
        metric = constituent_alpha_params[ticker]["metric"]
        if number_of_optimization_periods == 1:
            with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'rb') as file:
                res_test2 = pickle.load(file)
            with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_1_Selected_Strategies_res.pkl', 'rb') as file:
                res_test = pickle.load(file)
            with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_1_Selected_Strategies_ss.pkl', 'rb') as file:
                ss_test = pickle.load(file)
            res_test4 = []
            res_test8 = []
        if number_of_optimization_periods == 2:
            with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'rb') as file:
                res_test2 = pickle.load(file)
            with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_4_All_Strategies.pkl', 'rb') as file:
                res_test4 = pickle.load(file)
            with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_2_Selected_Strategies_res.pkl', 'rb') as file:
                res_test = pickle.load(file)
            with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_2_Selected_Strategies_ss.pkl', 'rb') as file:
                ss_test = pickle.load(file)
            res_test8 = []
        if number_of_optimization_periods == 3:
            with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'rb') as file:
                res_test2 = pickle.load(file)
            with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_4_All_Strategies.pkl', 'rb') as file:
                res_test4 = pickle.load(file)
            with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_8_All_Strategies.pkl', 'rb') as file:
                res_test8 = pickle.load(file)
            with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_3_Selected_Strategies_res.pkl', 'rb') as file:
                res_test = pickle.load(file)
            with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_3_Selected_Strategies_ss.pkl', 'rb') as file:
                ss_test = pickle.load(file)

        temp_og = get_data(ticker, "investpy")

        dates = valid_dates(pd.date_range(start=str(temp_og.iloc[1]['Date'] + timedelta(days=365))[:10],
                                          end="2024-06-15", freq=f'3M'))
        datesb = []
        for date_i in range(len(dates) - (int(24 / 3) + 1)):
            if (3 * date_i) % recalib_months == 0:
                datesb.append(dates[date_i + int(24 / 3)])
        datesb.append(date.today())
        weights_supposed_to_be = [None] * (len(datesb) - 1)

        with open(
                f'{ticker}/weights/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.pkl',
                'rb') as file:
            weights = pickle.load(file)

        if not os.path.exists(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}'):
            continue

        if number_of_optimization_periods == 1:
            with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'rb') as file:
                res_test2_imp = pickle.load(file)
            with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_OptPeriods_1_Selected_Strategies_res.pkl',
                      'rb') as file:
                res_test_imp = pickle.load(file)
            with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_OptPeriods_1_Selected_Strategies_ss.pkl',
                      'rb') as file:
                ss_test_imp = pickle.load(file)
            res_test4_imp = []
            res_test8_imp = []
        if number_of_optimization_periods == 2:
            with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'rb') as file:
                res_test2_imp = pickle.load(file)
            with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_TrainYrs_4_All_Strategies.pkl', 'rb') as file:
                res_test4_imp = pickle.load(file)
            with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_OptPeriods_2_Selected_Strategies_res.pkl',
                      'rb') as file:
                res_test_imp = pickle.load(file)
            with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_OptPeriods_2_Selected_Strategies_ss.pkl',
                      'rb') as file:
                ss_test_imp = pickle.load(file)
            res_test8_imp = []
        if number_of_optimization_periods == 3:
            with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'rb') as file:
                res_test2_imp = pickle.load(file)
            with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_TrainYrs_4_All_Strategies.pkl', 'rb') as file:
                res_test4_imp = pickle.load(file)
            with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_TrainYrs_8_All_Strategies.pkl', 'rb') as file:
                res_test8_imp = pickle.load(file)
            with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_OptPeriods_3_Selected_Strategies_res.pkl',
                      'rb') as file:
                res_test_imp = pickle.load(file)
            with open(f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/SelectedStrategies/{ticker}_OptPeriods_3_Selected_Strategies_ss.pkl',
                      'rb') as file:
                ss_test_imp = pickle.load(file)
        try:
            with open(
                    f'/nas/Algo/Live_emailer_updates/zip_files_for_recalibration/{ticker}/weights/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.pkl',
                    'rb') as file:
                weights_imp = pickle.load(file)
        except:
            weights_imp = []
        print("*" * 100)
        print(ticker)
        print(f"len2: {len(res_test2)}")
        print(f"len4: {len(res_test4)}")
        print(f"len8: {len(res_test8)}")
        print(f"len_res_test: {len(res_test)}")
        print(f"len_ss_test: {len(ss_test)}")
        print(f"Weights: {len(weights)}")
        print(f"weights_supposed_to_be: {len(weights_supposed_to_be)}")
        print(f"len2_imp: {len(res_test2_imp)}")
        print(f"len4_imp: {len(res_test4_imp)}")
        print(f"len8_imp: {len(res_test8_imp)}")
        print(f"len_res_test_imp: {len(res_test_imp)}")
        print(f"len_ss_test_imp: {len(ss_test_imp)}")
        print(f"Weights_imp: {len(weights_imp)}")
        print("*" * 100)

        if recalibrate_what=="strategies":
            if number_of_optimization_periods == 1:
                with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'wb') as file:
                    pickle.dump((res_test2+res_test2_imp), file)
                with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_1_Selected_Strategies_res.pkl', 'wb') as file:
                    pickle.dump((res_test+res_test_imp), file)
                with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_1_Selected_Strategies_ss.pkl', 'wb') as file:
                    pickle.dump((ss_test+ss_test_imp), file)
                res_test4 = []
                res_test8 = []
            if number_of_optimization_periods == 2:
                with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'wb') as file:
                    pickle.dump((res_test2+res_test2_imp), file)
                with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_4_All_Strategies.pkl', 'wb') as file:
                    pickle.dump((res_test4+res_test4_imp), file)
                with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_2_Selected_Strategies_res.pkl', 'wb') as file:
                    pickle.dump((res_test+res_test_imp), file)
                with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_2_Selected_Strategies_ss.pkl', 'wb') as file:
                    pickle.dump((ss_test+ss_test_imp),file)
                res_test8 = []
            if number_of_optimization_periods == 3:
                with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_2_All_Strategies.pkl', 'wb') as file:
                    pickle.dump((res_test2+res_test2_imp), file)
                with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_4_All_Strategies.pkl', 'wb') as file:
                    pickle.dump((res_test4+res_test4_imp), file)
                with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_8_All_Strategies.pkl', 'wb') as file:
                    pickle.dump((res_test8+res_test8_imp), file)
                with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_3_Selected_Strategies_res.pkl', 'wb') as file:
                    pickle.dump((res_test+res_test_imp), file)
                with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_3_Selected_Strategies_ss.pkl', 'wb') as file:
                    pickle.dump((ss_test+ss_test_imp),file)

        if recalibrate_what == "weights":
            with open(f'{ticker}/weights/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.pkl', 'wb') as file:
                pickle.dump((weights+[weights_imp[0][1]]), file)