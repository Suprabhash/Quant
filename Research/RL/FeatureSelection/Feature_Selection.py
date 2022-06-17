import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import pickle
from tqdm import tqdm
from functools import partial
from Utils.add_features import add_fisher, add_inverse_fisher, add_constance_brown
from Utils.add_features import max_over_lookback, min_over_lookback, sma, x, shift
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from Optimisers.Optimiser import Optimiser
from Data.data_retrieval import get_data
import talib
from Utils.utils import calc_NMI
import itertools
import multiprocessing
from datetime import datetime

#Fs
def add_BBANDS_Mid(input):
    temp = input[0].copy()
    lookback = input[1]
    _, middleband, _ = talib.BBANDS(temp["Close"], timeperiod=lookback, nbdevup=2, nbdevdn=2, matype=0)
    if f'BBMiddleband{lookback}' not in temp.columns:
        temp[f'BBMiddleband{lookback}'] = middleband
    return temp

def add_BBANDS_Upper(input):
    temp = input[0].copy()
    lookback = input[1]
    upperband, _, _ = talib.BBANDS(temp["Close"], timeperiod=lookback, nbdevup=2, nbdevdn=2, matype=0)
    if f'BBUpperband{lookback}' not in temp.columns:
        temp[f'BBUpperband{lookback}'] = upperband
    return temp

def add_BBANDS_Lower(input):
    temp = input[0].copy()
    lookback = input[1]
    _, _, lowerband = talib.BBANDS(temp["Close"], timeperiod=lookback, nbdevup=2, nbdevdn=2, matype=0)
    if f'BBLowerrband{lookback}' not in temp.columns:
        temp[f'BBLowerrband{lookback}'] = lowerband
    return temp

def add_DEMA(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'DEMA{lookback}' not in temp.columns:
        temp[f'DEMA{lookback}'] = talib.DEMA(temp["Close"], timeperiod=lookback)
    return temp

def add_EMA(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'EMA{lookback}' not in temp.columns:
        temp[f'EMA{lookback}'] = talib.EMA(temp["Close"], timeperiod=lookback)
    return temp

def add_HT_TRENDLINE(input):
    temp = input[0].copy()
    if f'HT_TRENDLINE' not in temp.columns:
        temp[f'HT_TRENDLINE'] = talib.HT_TRENDLINE(temp["Close"])
    return temp

def add_KAMA(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'KAMA{lookback}' not in temp.columns:
        temp[f'KAMA{lookback}'] = talib.KAMA(temp["Close"], timeperiod=lookback)
    return temp

def add_MA(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'MA{lookback}' not in temp.columns:
        temp[f'MA{lookback}'] = talib.MA(temp["Close"], timeperiod=lookback)
    return temp

def add_MAMA(input):
    temp = input[0].copy()
    mama, _ = talib.MAMA(temp["Close"], fastlimit=0, slowlimit=0)
    if f'MAMA' not in temp.columns:
        temp[f'MAMA'] = mama
    return temp

def add_FAMA(input):
    temp = input[0].copy()
    _, fama = talib.MAMA(temp["Close"], fastlimit=0, slowlimit=0)
    if f'FAMA' not in temp.columns:
        temp[f'FAMA'] = fama
    return temp

def add_MIDPOINT(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'MIDPOINT{lookback}' not in temp.columns:
        temp[f'MIDPOINT{lookback}'] = talib.MIDPOINT(temp["Close"], timeperiod=lookback)
    return temp

def add_MIDPRICE(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'MIDPRICE{lookback}' not in temp.columns:
        temp[f'MIDPRICE{lookback}'] = talib.MIDPRICE(temp["High"], temp["Low"], timeperiod=lookback)
    return temp

def add_SAR(input):
    temp = input[0].copy()
    if f'SAR' not in temp.columns:
        temp[f'SAR'] = talib.SAR(temp["High"], temp["Low"], acceleration=0, maximum=0)
    return temp

def add_SAREXT(input):
    temp = input[0].copy()
    if f'SAREXT' not in temp.columns:
        temp[f'SAREXT'] = talib.SAREXT(temp["High"], temp["Low"], startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
    return temp

def add_SMA(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'SMA{lookback}' not in temp.columns:
        temp[f'SMA{lookback}'] = talib.SMA(temp["Close"], timeperiod=lookback)
    return temp

def add_T3(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'T3{lookback}' not in temp.columns:
        temp[f'T3{lookback}'] = talib.T3(temp["Close"], timeperiod=lookback, vfactor=0)
    return temp

def add_TEMA(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'TEMA{lookback}' not in temp.columns:
        temp[f'TEMA{lookback}'] = talib.TEMA(temp["Close"], timeperiod=lookback)
    return temp

def add_TRIMA(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'TRIMA{lookback}' not in temp.columns:
        temp[f'TRIMA{lookback}'] = talib.TRIMA(temp["Close"], timeperiod=lookback)
    return temp

def add_WMA(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'WMA{lookback}' not in temp.columns:
        temp[f'WMA{lookback}'] = talib.WMA(temp["Close"], timeperiod=lookback)
    return temp

def add_AccumulationDistribution(input):
    temp = input[0].copy()
    if f'AccumulationDistribution' not in temp.columns:
        temp[f'AccumulationDistribution'] = talib.AD(temp["High"], temp["Low"], temp["Close"], temp["Volume"])
    return temp

def add_OnBalanceVolume(input):
    temp = input[0].copy()
    if f'OnBalanceVolume' not in temp.columns:
        temp[f'OnBalanceVolume'] = talib.AD(temp["Close"], temp["Volume"])
    return temp

def add_AverageTrueRange(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'ATR{lookback}' not in temp.columns:
        temp[f'ATR{lookback}'] = talib.ATR(temp["High"], temp["Low"], temp["Close"], timeperiod=lookback)
    return temp

def add_NormalizedAverageTrueRange(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'NATR{lookback}' not in temp.columns:
        temp[f'NATR{lookback}'] = talib.NATR(temp["High"], temp["Low"], temp["Close"], timeperiod=lookback)
    return temp

def add_TrueRange(input):
    temp = input[0].copy()
    if f'TR' not in temp.columns:
        temp[f'TR'] = talib.TRANGE(temp["High"], temp["Low"], temp["Close"])
    return temp


def add_AvgPrice(input):
    temp = input[0].copy()
    if f'AvgPrice' not in temp.columns:
        temp[f'AvgPrice'] = talib.AVGPRICE(temp["Open"], temp["High"], temp["Low"], temp["Close"])
    return temp

def add_MedPrice(input):
    temp = input[0].copy()
    if f'MedPrice' not in temp.columns:
        temp[f'MedPrice'] = talib.MEDPRICE(temp["High"], temp["Low"])
    return temp

def add_TypicalPrice(input):
    temp = input[0].copy()
    if f'TypicalPrice' not in temp.columns:
        temp[f'TypicalPrice'] = talib.TYPPRICE(temp["High"], temp["Low"], temp["Close"])
    return temp

def add_WeightedClosePrice(input):
    temp = input[0].copy()
    if f'WeightedClosePrice' not in temp.columns:
        temp[f'WeightedClosePrice'] = talib.WCLPRICE(temp["High"], temp["Low"], temp["Close"])
    return temp

def add_Beta(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'BETA{lookback}' not in temp.columns:
        temp[f'BETA{lookback}'] = talib.BETA(temp["High"], temp["Low"], timeperiod=lookback)
    return temp

def add_AverageDirectionalMovementIndex(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'ADX{lookback}' not in temp.columns:
        temp[f'ADX{lookback}'] = talib.ADX(temp["High"], temp["Low"], temp["Close"], timeperiod=lookback)
    return temp

def add_AverageDirectionalMovementIndexRating(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'ADXR{lookback}' not in temp.columns:
        temp[f'ADXR{lookback}'] = talib.ADXR(temp["High"], temp["Low"], temp["Close"], timeperiod=lookback)
    return temp

def add_AROON(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'AROON{lookback}' not in temp.columns:
        temp[f'AROON{lookback}'] = talib.AROON(temp["High"], temp["Low"], timeperiod=lookback)
    return temp

def add_AROONOSC(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'AROONOSC{lookback}' not in temp.columns:
        temp[f'AROONOSC{lookback}'] = talib.AROONOSC(temp["High"], temp["Low"], timeperiod=lookback)
    return temp

def add_BalanceofPower(input):
    temp = input[0].copy()
    if f'BOP' not in temp.columns:
        temp[f'BOP'] = talib.BOP(temp["Open"], temp["High"], temp["Low"], temp["Close"])
    return temp

def add_CommodityChannelIndex(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'CCI{lookback}' not in temp.columns:
        temp[f'CCI{lookback}'] = talib.CCI(temp["High"], temp["Low"], temp["Close"], timeperiod=lookback)
    return temp

def add_ChandeMomentumOscillator(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'CMO{lookback}' not in temp.columns:
        temp[f'CMO{lookback}'] = talib.CMO(temp["Close"], timeperiod=lookback)
    return temp

def add_DirectionalMovementIndex(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'DX' not in temp.columns:
        temp[f'DX'] = talib.DX(temp["High"], temp["Low"], temp["Close"], timeperiod=lookback)
    return temp

def add_MoneyFlowIndex(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'MFI' not in temp.columns:
        temp[f'MFI'] = talib.MFI(temp["High"], temp["Low"], temp["Close"], temp["Volume"], timeperiod=lookback)
    return temp

def add_MinusDirectionalIndicator(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'MINUS_DI' not in temp.columns:
        temp[f'MINUS_DI'] = talib.MINUS_DI(temp["High"], temp["Low"], temp["Close"], timeperiod=lookback)
    return temp

def add_MinusDirectionalMovement(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'MINUS_DM' not in temp.columns:
        temp[f'MINUS_DM'] = talib.MINUS_DM(temp["High"], temp["Low"], timeperiod=lookback)
    return temp

def add_Momentum(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'MOM' not in temp.columns:
        temp[f'MOM'] = talib.MOM(temp["Close"], timeperiod=lookback)
    return temp

def add_PlusDirectionalIndicator(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'PLUS_DI' not in temp.columns:
        temp[f'PLUS_DI'] = talib.PLUS_DI(temp["High"], temp["Low"], temp["Close"], timeperiod=lookback)
    return temp


def add_PlusDirectionalMovement(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'PLUS_DM' not in temp.columns:
        temp[f'PLUS_DM'] = talib.PLUS_DM(temp["High"], temp["Low"], timeperiod=lookback)
    return temp

def add_RSI(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'RSI' not in temp.columns:
        temp[f'RSI'] = talib.RSI(temp["Close"], timeperiod=lookback)
    return temp


#Gs

def LinearRegression(df, col_name,lookback):
    df[f"LinearRegression{lookback}_{col_name}"] = talib.LINEARREG(df[col_name], timeperiod=lookback)
    return df

def LinearRegressionAngle(df, col_name,lookback):
    df[f"LinearRegressionAngle{lookback}_{col_name}"] = talib.LINEARREG_ANGLE(df[col_name], timeperiod=lookback)
    return df

def LinearRegressionIntercept(df, col_name,lookback):
    df[f"LinearRegressionIntercept{lookback}_{col_name}"] = talib.LINEARREG_INTERCEPT(df[col_name], timeperiod=lookback)
    return df

def LinearRegressionSlope(df, col_name,lookback):
    df[f"LinearRegressionSlope{lookback}_{col_name}"] = talib.LINEARREG_SLOPE(df[col_name], timeperiod=lookback)
    return df

def Stdev(df, col_name,lookback):
    df[f"Stdev{lookback}_{col_name}"] = talib.STDDEV(df[col_name], timeperiod=lookback)
    return df

def TimeSeriesForecast(df, col_name,lookback):
    df[f"TimeSeriesForecast{lookback}_{col_name}"] = talib.TSF(df[col_name], timeperiod=lookback)
    return df

def Variance(df, col_name,lookback):
    df[f"Variance{lookback}_{col_name}"] = talib.VAR(df[col_name], timeperiod=lookback)
    return df

def ACOS(df, col_name,lookback):
    df[f"ACOS_{col_name}"] = talib.ACOS(df[col_name])
    return df

def ASIN(df, col_name,lookback):
    df[f"ASIN_{col_name}"] = talib.ASIN(df[col_name])
    return df

def ATAN(df, col_name,lookback):
    df[f"ATAN_{col_name}"] = talib.ATAN(df[col_name])
    return df

def CEIL(df, col_name,lookback):
    df[f"CEIL_{col_name}"] = talib.CEIL(df[col_name])
    return df

def COS(df, col_name,lookback):
    df[f"COS_{col_name}"] = talib.COS(df[col_name])
    return df

def COSH(df, col_name,lookback):
    df[f"COSH_{col_name}"] = talib.COSH(df[col_name])
    return df

def EXP(df, col_name,lookback):
    df[f"EXP_{col_name}"] = talib.EXP(df[col_name])
    return df

def FLOOR(df, col_name,lookback):
    df[f"FLOOR_{col_name}"] = talib.FLOOR(df[col_name])
    return df

def LN(df, col_name,lookback):
    df[f"LN_{col_name}"] = talib.LN(df[col_name])
    return df

def LOG10(df, col_name,lookback):
    df[f"LOG10_{col_name}"] = talib.LOG10(df[col_name])
    return df

def SIN(df, col_name,lookback):
    df[f"SIN_{col_name}"] = talib.SIN(df[col_name])
    return df

def SINH(df, col_name,lookback):
    df[f"SINH_{col_name}"] = talib.SINH(df[col_name])
    return df

def SQRT(df, col_name,lookback):
    df[f"SQRT_{col_name}"] = talib.SQRT(df[col_name])
    return df

def TAN(df, col_name,lookback):
    df[f"TAN_{col_name}"] = talib.TAN(df[col_name])
    return df

def TANH(df, col_name,lookback):
    df[f"TANH_{col_name}"] = talib.TANH(df[col_name])
    return df

def MAXINDEX(df, col_name,lookback):
    df[f"MAXINDEX_{col_name}"] = talib.MAXINDEX(df[col_name], timeperiod=lookback)
    return df

def MININDEX(df, col_name,lookback):
    df[f"MININDEX{col_name}"] = talib.MININDEX(df[col_name], timeperiod=lookback)
    return df

###

def return_nmi(input):
    try:
        with open(f'Data.pkl','rb') as file:
            data = pickle.load(file)
        [f, flb, g, glb, frtype, lf] = input
        X = f([data, flb]).iloc[flb - 1:].reset_index(drop=True)
        col_name = X.columns[-1]
        X = g(X, col_name, glb).iloc[glb - 1:].reset_index(drop=True)
        Y = X[f"{frtype}{lf}"]
        X = X[X.columns[-1]]
        nmi = calc_NMI(X, Y)
        return {"F": f.__name__, "F_Lookback": flb, "G": g.__name__, "G_Lookback": glb, "FReturn_Type": frtype, "FReturn_lookforward": lf, "NMI": nmi}
    except Exception as e:
        with open('readme.txt', 'a') as f:
            f.write(f"Error: {e} in: {[el if not(callable(el)) else el.__name__ for el in input ]}\n")

if __name__ == "__main__":

    start_time = datetime.now()
    data = get_data('NIFc1', 'D')

    #Add Forward Returns
    freturn_types = ["FReturn", "MaxFReturn", "MinFReturn"]
    lookforwards = [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 180, 260]
    for i in range(1, max(lookforwards)+1):
        data[f"FReturn{i}"] = data["Close"].shift(-i) / data["Close"] - 1
    data.dropna(inplace=True)
    for lookforward in lookforwards:
        data[f"MaxFReturn{lookforward}"] = data[[f"FReturn{i}" for i in range(1, lookforward+1)]].max(axis=1)
        data[f"MinFReturn{lookforward}"] = data[[f"FReturn{i}" for i in range(1, lookforward+1)]].min(axis=1)
    data = data.drop(columns=[f"FReturn{i}" for i in range(1, max(lookforwards)+1) if i not in lookforwards])

    with open(f'Data.pkl','wb') as file:
        pickle.dump(data, file)

    #Feature space
    lookbacks = [3,5,7,10,15,21,30,45,50,63,75,90,100,126,150,175,200,225,252]
    f_functions = [add_fisher, add_inverse_fisher, add_constance_brown, add_BBANDS_Mid, add_BBANDS_Lower, add_BBANDS_Upper, add_DEMA, add_EMA, add_HT_TRENDLINE, add_KAMA, add_MA, add_MAMA, add_FAMA, add_MIDPOINT, add_MIDPRICE, add_SAR, add_SAREXT,
                   add_SMA, add_T3, add_TEMA, add_TRIMA, add_WMA, add_AccumulationDistribution, add_OnBalanceVolume, add_AverageTrueRange, add_NormalizedAverageTrueRange, add_TrueRange,
                   add_AvgPrice, add_MedPrice, add_TypicalPrice, add_WeightedClosePrice, add_Beta, add_AverageDirectionalMovementIndex, add_AverageDirectionalMovementIndexRating, add_AROON,
                   add_AROONOSC, add_BalanceofPower, add_CommodityChannelIndex, add_ChandeMomentumOscillator, add_DirectionalMovementIndex, add_MoneyFlowIndex, add_MinusDirectionalIndicator,
                   add_MinusDirectionalMovement, add_Momentum, add_PlusDirectionalIndicator, add_PlusDirectionalMovement, add_RSI]

    g_functions = [x, max_over_lookback, min_over_lookback, sma, shift, LinearRegression, LinearRegressionAngle, LinearRegressionSlope, LinearRegressionIntercept, Stdev, TimeSeriesForecast, Variance,
                   ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH, MAXINDEX, MININDEX]

    #Creating combinations for NMI calculation
    params = [f_functions, lookbacks, g_functions, lookbacks, freturn_types, lookforwards]
    inputs = list(itertools.product(*params))
    print(len(inputs))

    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results = pool.map(return_nmi, inputs)
    finally:  # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    results = list(filter(None, results))
    results = pd.DataFrame(results)
    results.sort_values(by="NMI", ascending=False, inplace=True)
    results.reset_index(drop=True, inplace=True)
    results.to_csv("FeatureSelectionResults.csv")
    with open(f'FeatureSelectionResults.pkl','wb') as file:
        pickle.dump(results, file)
    print(f"Time taken: {datetime.now()-start_time}")










