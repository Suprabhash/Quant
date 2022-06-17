import ssl
import datetime
import pandas as pd
import pickle
import os
import math
import numpy as np
from datetime import date
import multiprocessing
from scipy import stats
from sklearn.cluster import KMeans
import itertools
from sklearn.preprocessing import StandardScaler
import psutil

RANDOM_STATE = 835


def RSI(data_df,period):
    series = data_df['Close']

    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period - 1]] = np.mean(u[:period])  # first value is sum of avg gains
    u = u.drop(u.index[:(period - 1)])
    d[d.index[period - 1]] = np.mean(d[:period])  # first value is sum of avg losses
    d = d.drop(d.index[:(period - 1)])
    rs = pd.DataFrame.ewm(u, com=period - 1, adjust=False).mean() / \
         pd.DataFrame.ewm(d, com=period - 1, adjust=False).mean()
    rsi = (100 - 100 / (1 + rs))
    data_df = pd.concat([data_df, rsi], axis=1)
    rsi_df = (data_df.iloc[:, -1:])
    rsi_df.columns = ['RSI']
    return rsi_df.fillna(0).iloc[:,0]

def constance_brown(data_df):
  r=RSI(data_df,14)
  rsi_mom_length=9
  ma_length=3
  rsi_ma_length=3
  fastLength=13
  slowLength=33
  rsidelta=[0]*len(data_df)

  for i in range(len(data_df)):
    if i<rsi_mom_length:
        rsidelta[i] = np.nan
    else:
        rsidelta[i]=r[i]-r[i-rsi_mom_length]

  rsisigma=RSI(data_df,rsi_ma_length).rolling(window=ma_length).mean()
  rsidelta = [0 if math.isnan(x) else x for x in rsidelta]
  s=[0]*len(data_df)
  for i in range(len(rsidelta)):
    s[i]=rsidelta[i]+rsisigma[i]
  #s = [0 if math.isnan(x) else x for x in s]

  data_df["CB"] = s
  data_df["FMACB"] = pd.DataFrame(s).rolling(window=fastLength).mean()
  data_df["SMACB"] = pd.DataFrame(s).rolling(window=slowLength).mean()
  return data_df

def add_pivots(temp_og,col_name):

    temp_og[f"{col_name}_TypeCurrentPivot"] = np.where((temp_og[f"{col_name}"] > temp_og[f"{col_name}"].shift(1)) & (temp_og[f"{col_name}"] > temp_og[f"{col_name}"].shift(-1)), 1, np.nan)
    temp_og[f"{col_name}_TypeCurrentPivot"] = np.where((temp_og[f"{col_name}"] < temp_og[f"{col_name}"].shift(1)) & (temp_og[f"{col_name}"] < temp_og[f"{col_name}"].shift(-1)), -1, temp_og[f"{col_name}_TypeCurrentPivot"])
    temp_og[f"{col_name}_PivotValue"] = np.where((temp_og[f"{col_name}"] > temp_og[f"{col_name}"].shift(1)) & (temp_og[f"{col_name}"] > temp_og[f"{col_name}"].shift(-1)), temp_og[f"{col_name}"], np.nan)
    temp_og[f"{col_name}_PivotValue"] = np.where((temp_og[f"{col_name}"] < temp_og[f"{col_name}"].shift(1)) & (temp_og[f"{col_name}"] < temp_og[f"{col_name}"].shift(-1)), temp_og[f"{col_name}"], temp_og[f"{col_name}_PivotValue"])
    temp_og[f"{col_name}_TypePreviousPivot"] = temp_og[f"{col_name}_TypeCurrentPivot"].fillna(method='ffill').shift(1).fillna(0)
    temp_og[f"{col_name}_PreviousPivotValue"] = temp_og[f"{col_name}_PivotValue"].fillna(method='ffill').shift(1).fillna(0)
    temp_og[f"{col_name}_TypePreviousPivot"] = np.where(np.isnan(temp_og[f"{col_name}_TypeCurrentPivot"]), np.nan, temp_og[f"{col_name}_TypePreviousPivot"])
    temp_og[f"{col_name}_PreviousPivotValue"] = np.where(np.isnan(temp_og[f"{col_name}_PivotValue"]), np.nan, temp_og[f"{col_name}_PreviousPivotValue"])
    temp_og[f"{col_name}_PreviousHighPivotValueCB"] = pd.DataFrame(np.where(((temp_og[f"{col_name}_TypePreviousPivot"]==-1)),temp_og[f"{col_name}_PivotValue"],np.nan)).fillna(method='ffill').shift(1)
    temp_og[f"{col_name}_PreviousLowPivotValueCB"] = pd.DataFrame(np.where(((temp_og[f"{col_name}_TypePreviousPivot"]==1)),temp_og[f"{col_name}_PivotValue"],np.nan)).fillna(method='ffill').shift(1)
    temp_og[f"{col_name}_DaysSincePreviousHighPivot"] = np.nan
    temp_og[f"{col_name}_DaysSincePreviousLowPivot"] = np.nan
    temp_og[f"{col_name}_IsHighPivot"] = np.where(temp_og[f"{col_name}_TypeCurrentPivot"]==1,1,0)
    temp_og[f"{col_name}_IsLowPivot"] = np.where(temp_og[f"{col_name}_TypeCurrentPivot"]==-1,1,0)
    for i in range(len(temp_og)):
        if not(np.isnan(temp_og.iloc[i][f"{col_name}_TypeCurrentPivot"])):
            try:
                if len(temp_og[f"Date"][temp_og[f"{col_name}_PivotValue"]==temp_og.iloc[i][f"{col_name}_PreviousPivotValue"]]) == 1:
                    temp_og.loc[i, f"{col_name}_DaysSincePreviousHighPivot"] = (temp_og.iloc[i][f"Date"] - temp_og[f"Date"][temp_og[f"{col_name}_PivotValue"] == temp_og.iloc[i][f"{col_name}_PreviousHighPivotValueCB"]].iloc[0]).days ##
                    temp_og.loc[i, f"{col_name}_DaysSincePreviousLowPivot"] = (temp_og.iloc[i][f"Date"] - temp_og[f"Date"][temp_og[f"{col_name}_PivotValue"] == temp_og.iloc[i][f"{col_name}_PreviousLowPivotValueCB"]].iloc[0]).days  ##
            except:
                continue
        else:
            try:
                temp_og.loc[i, f"{col_name}_DaysSincePreviousHighPivot"] = temp_og.loc[i-1, f"{col_name}_DaysSincePreviousHighPivot"] + 1
                temp_og.loc[i, f"{col_name}_DaysSincePreviousLowPivot"] = temp_og.loc[i-1, f"{col_name}_DaysSincePreviousLowPivot"] + 1
            except:
                continue

    #temp_og[f"{col_name}_TypeCurrentPivot"] = temp_og[f"{col_name}_TypeCurrentPivot"].shift(1)
    #temp_og[f"{col_name}_PivotValue"] = temp_og[f"{col_name}_PivotValue"].shift(1)
    #temp_og[f"{col_name}_TypePreviousPivot"] = temp_og[f"{col_name}_TypePreviousPivot"].shift(1)
    #temp_og[f"{col_name}_PreviousPivotValue"] = temp_og[f"{col_name}_PreviousPivotValue"].shift(1)
    temp_og[f"{col_name}_PreviousHighPivotValueCB"] = temp_og[f"{col_name}_PreviousHighPivotValueCB"].shift(1)
    temp_og[f"{col_name}_PreviousLowPivotValueCB"] = temp_og[f"{col_name}_PreviousLowPivotValueCB"].shift(1)
    temp_og[f"{col_name}_DaysSincePreviousHighPivot"] = temp_og[f"{col_name}_DaysSincePreviousHighPivot"].shift(1)
    temp_og[f"{col_name}_DaysSincePreviousLowPivot"] = temp_og[f"{col_name}_DaysSincePreviousLowPivot"].shift(1)
    temp_og[f"{col_name}_IsHighPivot"] = temp_og[f"{col_name}_IsHighPivot"].shift(1)
    temp_og[f"{col_name}_IsLowPivot"] = temp_og[f"{col_name}_IsLowPivot"].shift(1)
    # window = 10
    # th_high = -1
    # th_low = 1
    # num_stdev = 1
    #
    # temp_og[f"{col_name}RMean"] = temp_og[f"{col_name}"].rolling(window=window).mean()
    # temp_og[f"{col_name}RStDev"] = temp_og[f"{col_name}"].rolling(window=window).std()

    # temp_og[f"{col_name}PreviousPivotVal"] = np.where(abs(temp_og[f"{col_name}If{col_name}HighPivot"]) > 0,temp_og[f"{col_name}If{col_name}HighPivot"],temp_og[f"{col_name}If{col_name}LowPivot"])
    # temp_og[f"{col_name}PreviousPivotVal"] = temp_og[f"{col_name}PreviousPivotVal"].fillna(method='ffill').shift(1).fillna(0)

    # temp_og[f"{col_name}If{col_name}HighPivot"] = np.where((abs(temp_og[f"{col_name}If{col_name}HighPivot"])/abs(temp_og[f"{col_name}PreviousPivotVal"])-1>th_high)&(temp_og[f"{col_name}If{col_name}HighPivot"]>(temp_og[f"{col_name}RMean"]+num_stdev*temp_og[f"{col_name}RStDev"])), temp_og[f"{col_name}If{col_name}HighPivot"], np.nan)
    # temp_og[f"{col_name}If{col_name}LowPivot"] = np.where((abs(temp_og[f"{col_name}If{col_name}LowPivot"]) / abs(temp_og[f"{col_name}PreviousPivotVal"]) - 1 < th_low)&(temp_og[f"{col_name}If{col_name}LowPivot"]<(temp_og[f"{col_name}RMean"]-num_stdev*temp_og[f"{col_name}RStDev"])), temp_og[f"{col_name}If{col_name}LowPivot"], np.nan)
    # temp_og[f"Is{col_name}HighPivot"] = np.where((abs(temp_og[f"{col_name}If{col_name}HighPivot"]) / abs(temp_og[f"{col_name}PreviousPivotVal"]) - 1 > th_high)&(temp_og[f"{col_name}If{col_name}HighPivot"]>(temp_og[f"{col_name}RMean"]+num_stdev*temp_og[f"{col_name}RStDev"])),1, np.nan)
    # temp_og[f"Is{col_name}LowPivot"] = np.where((abs(temp_og[f"{col_name}If{col_name}LowPivot"]) / abs(temp_og[f"{col_name}PreviousPivotVal"]) - 1 < th_low)&(temp_og[f"{col_name}If{col_name}LowPivot"]<(temp_og[f"{col_name}RMean"]-num_stdev*temp_og[f"{col_name}RStDev"])),1, np.nan)

    return temp_og

def add_maxmin(temp_og, col_name,periods):
    for period in periods:
        temp_og[f"Max_{col_name}_{period}"] = temp_og[col_name].rolling(window = period).max()
        temp_og[f"Min_{col_name}_{period}"] = temp_og[col_name].rolling(window=period).min()

    return temp_og

def add_derivatives(temp_og, col_name, periods):
    for period in periods:
        temp_og[f"ROC_{col_name}_{period}"] = temp_og[col_name].pct_change(periods=period)
    if (col_name == "FMACB") | (col_name == "SMACB"):
        temp_og[f"Convexity_{col_name}"] = temp_og[col_name].pct_change().pct_change()
    return temp_og

def add_forward_return(temp_og, period):
    temp_og["FReturn"] = temp_og["Close"].pct_change(-period)  #incorrect. deprecated.
    temp_og["BinaryOutcome"]  = temp_og["FReturn"].apply(np.sign)
    return temp_og

def get_data_BTC(add_cb_features=True):
    filepath = "https://www.cryptodatadownload.com/cdd/Bitstamp_BTCUSD_d.csv"
    ssl._create_default_https_context = ssl._create_unverified_context
    temp_og = pd.read_csv(filepath, parse_dates=True, skiprows=1)
    temp_og.drop(columns=["date", "symbol", "Volume BTC"], inplace=True)
    temp_og["unix"] = temp_og["unix"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000) if len(
        str(int(x))) == 13 else datetime.datetime.fromtimestamp(x))
    temp_og.rename(columns={"unix": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close",
                            "Volume USD": "Volume"}, inplace=True)
    temp_og["Date"] = pd.to_datetime(temp_og["Date"])
    temp_og.sort_values("Date", ascending=True, inplace=True)
    temp_og.reset_index(drop=True, inplace=True)

    if os.path.isdir('BTC_D.pkl'):
        with open(f'BTC_D.pkl', 'rb') as file:
            temp_og_imp = pickle.load(file)
        temp_og = pd.concat([temp_og_imp, temp_og], axis=0)
        temp_og.drop_duplicates(keep="first", inplace=True)
        temp_og.reset_index(drop=True, inplace=True)
        with open(f'BTC_D.pkl', 'wb') as file:
            pickle.dump(pd.DataFrame(temp_og), file)
    else:
        with open(f'BTC_D.pkl', 'wb') as file:
            pickle.dump(pd.DataFrame(temp_og), file)

    filepath = "https://www.cryptodatadownload.com/cdd/Bitstamp_BTCUSD_1h.csv"
    ssl._create_default_https_context = ssl._create_unverified_context
    temp_og1 = pd.read_csv(filepath, parse_dates=True, skiprows=1)
    temp_og1.drop(columns=["date", "symbol", "Volume BTC"], inplace=True)
    temp_og1["unix"] = temp_og1["unix"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000) if len(
        str(int(x))) == 13 else datetime.datetime.fromtimestamp(x))
    temp_og1.rename(columns={"unix": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close",
                             "Volume USD": "Volume"}, inplace=True)
    temp_og1["Date"] = pd.to_datetime(temp_og1["Date"])
    temp_og1.Date = temp_og1.Date.dt.tz_localize('Asia/Kolkata')
    temp_og1.sort_values("Date", ascending=True, inplace=True)
    temp_og1.reset_index(drop=True, inplace=True)

    if os.path.isdir('BTC_H.pkl'):
        with open(f'BTC_H.pkl', 'rb') as file:
            temp_og1_imp = pickle.load(file)
        temp_og1 = pd.concat([temp_og1_imp, temp_og1], axis=0)
        temp_og1.drop_duplicates(keep="first", inplace=True)
        temp_og1.reset_index(drop=True, inplace=True)
        with open(f'BTC_H.pkl', 'wb') as file:
            pickle.dump(pd.DataFrame(temp_og1), file)
    else:
        with open(f'BTC_H.pkl', 'wb') as file:
            pickle.dump(pd.DataFrame(temp_og1), file)

    if add_cb_features==True:
        temp_og = constance_brown(temp_og)
        temp_og = add_pivots(temp_og, "CB")
        temp_og = add_pivots(temp_og, "FMACB")
        temp_og = add_pivots(temp_og, "SMACB")
        temp_og = add_maxmin(temp_og, "CB", [10, 20, 30, 60])
        temp_og = add_maxmin(temp_og, "FMACB", [10, 20, 30, 60])
        temp_og = add_maxmin(temp_og, "SMACB", [10, 20, 30, 60])
        temp_og = add_derivatives(temp_og, "CB", [10, 20, 30, 60])
        temp_og = add_derivatives(temp_og, "FMACB", [10, 20, 30, 60])
        temp_og = add_derivatives(temp_og, "SMACB", [10, 20, 30, 60])
        temp_og = add_forward_return(temp_og, 1)

    # temp_og1 = constance_brown(temp_og1)
    # temp_og1 = add_pivots(temp_og1, "CB")
    # temp_og1 = add_pivots(temp_og1, "FMACB")
    # temp_og1 = add_pivots(temp_og1, "SMACB")
    # temp_og1 = add_maxmin(temp_og1, "CB", [10, 20, 30, 60])
    # temp_og1 = add_maxmin(temp_og1, "FMACB", [10, 20, 30, 60])
    # temp_og1 = add_maxmin(temp_og1, "SMACB", [10, 20, 30, 60])
    # temp_og1 = add_derivatives(temp_og1, "CB", [10, 20, 30, 60])
    # temp_og1 = add_derivatives(temp_og1, "FMACB", [10, 20, 30, 60])
    # temp_og1 = add_derivatives(temp_og1, "SMACB", [10, 20, 30, 60])
    # temp_og1 = add_forward_return(temp_og1, 1)

    return temp_og, temp_og1

def select_all_strategies(train_monthsf, datesf, temp_ogf, ticker, save=True):
    inputs =[]
    for date_i in range(len(datesf)-(int(train_monthsf/3)+1)):
        inputs.append([date_i, datesf, temp_ogf, train_monthsf])
    try:
        pool = multiprocessing.Pool(processes=5, maxtasksperchild=1)
        parent = psutil.Process()
        parent.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        for child in parent.children():
            child.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        results = pool.map(get_strategies, inputs)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    res_test = [{'stats': None, 'results_upside': None, 'results_downside': None}] * (len(datesf)-(int(24/3)+1))
    for i in range(len(results)):
        res_test[results[i][0]+int((train_monthsf-24)/3)] = ({'stats': results[i][1], 'results_upside': results[i][2], 'results_downside': results[i][3]})

    if save==True:
        with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_{int(train_monthsf/12)}_All_Strategies.pkl', 'wb') as file:
            pickle.dump(res_test, file)

    return res_test

def prepare_volume_features_from_cache(temp_og):
    with open(f'BTC_VolumeLevels.pkl', 'rb') as file:
        temp = pickle.load(file)
    vol_temp = pd.DataFrame()
    vol_temp["Date"] = temp['Date']
    temp = pd.concat([temp.set_index("Date"), temp_og.set_index("Date")], axis=1, join="inner").reset_index()
    for n in [2, 5, 10, 21, 42, 63, 126, 252, 504]:
        vol_temp[f"dev_from_poc_{n}"] = np.nan
        for i in range(len(vol_temp)):
            try:
                vol_temp.loc[i, f"dev_from_poc_{n}"] = temp.iloc[i]["Close"]/temp.iloc[i][f"PriceLevels_{n}"]['poc']-1
            except:
                continue
    vol_temp = pd.concat([vol_temp.set_index("Date"), temp_og.set_index("Date")], axis=1, join="inner").reset_index()
    return vol_temp

def cluster_grid_search(df_cluster):
    results = []
    all_features = list(df_cluster.columns)
    all_features.remove('FReturn')
    all_features.remove('FReturn30')
    for num_features in [7,8,9]:
        selected_features = list(itertools.combinations(all_features, num_features))
        for i in range(len(selected_features)):
            for num_clusters in [3,4,5]:
                df_cluster_input = df_cluster[list(selected_features[i])]
                X = df_cluster_input.values
                k_means = KMeans(n_clusters=num_clusters, random_state = RANDOM_STATE)
                model = k_means.fit(X)
                labels = k_means.labels_
                df_cluster_input = pd.concat([df_cluster_input.reset_index(), df_cluster["FReturn"].reset_index()["FReturn"],pd.DataFrame(labels).rename({0: 'labels'}, axis=1)], axis=1).set_index("Date")
                sharpes = []
                for label in range(num_clusters):
                    if len(df_cluster_input[df_cluster_input["labels"] == label]) > 0:
                        sharpes.append(df_cluster_input[df_cluster_input["labels"] == label]["FReturn"].mean(axis=0)/df_cluster_input[df_cluster_input["labels"] == label]["FReturn"].std(axis=0))
                    else:
                        sharpes.append(0)
                results.append({"Num Features": num_features, "num_clusters": num_clusters,"selected_features": list(selected_features[i]), "sharpes": sharpes})
    res = pd.DataFrame(results)
    return res


def get_strategies(inp):

    date_i = inp[0]
    dates = inp[1]
    temp_og = inp[2]
    train_months = inp[3]

    for i in range(1, 31):
        temp_og[f"FReturn{i}"] = temp_og["Close"].pct_change(i)

    temp = temp_og.loc[(temp_og["Date"] > str(dates[date_i])) & (temp_og["Date"] < str(
        dates[date_i + int(train_months / 3)]))].reset_index().drop(['index'], axis=1)

    data = temp.copy().drop(columns=["FReturn","Volume","CB_TypeCurrentPivot","CB_PivotValue","CB_TypePreviousPivot","SMACB_TypeCurrentPivot","SMACB_PivotValue","SMACB_TypePreviousPivot","FMACB_TypeCurrentPivot","FMACB_PivotValue","FMACB_TypePreviousPivot","CB_PreviousPivotValue","SMACB_PreviousPivotValue","FMACB_PreviousPivotValue"])
    data.set_index("Date", inplace=True)

    scaled_df_train = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns, index=data.index)

    df_cluster_upside = prepare_clustering_dataframe(scaled_df_train, data, how="UpsideMax")
    df_cluster_downside = prepare_clustering_dataframe(scaled_df_train, data, how="DownsideMin")

    stats = data.describe()

    results_upside = cluster_grid_search(df_cluster_upside)
    results_downside = cluster_grid_search(df_cluster_downside)
    return (date_i, stats, results_upside, results_downside)

def valid_dates(dates_all):
    dates = []
    i = 0
    while True:
        dates.append(dates_all[i])
        #if dates_all[i] > pd.to_datetime(date.today()):
        if dates_all[i] > pd.to_datetime(date.today().replace(month=9, day=17)):
            break
        i = i + 1
    return dates

def prepare_clustering_dataframe(df_train, df_train_unscaled, how):
    df_cluster = df_train.copy()

    # if how == "UpsideMaxAll":
    #     for i in range(1,31):
    #         df_train_unscaled[f"FReturn{i}"] = df_train_unscaled["Close"].pct_change(i)
    #     df_train_unscaled[f"FReturn"] = df_train_unscaled[[f"FReturn{i}" for i in range(1,31)]].max( axis=1)
    #     df_train_unscaled = df_train_unscaled.drop(columns=[f"FReturn{i}" for i in range(1,30)])
    #     df_train_unscaled.dropna(inplace=True)
    #     df_cluster = pd.concat([df_cluster, df_train_unscaled[["FReturn","FReturn30"]]], axis=1)

    if how == "UpsideMax":
        # for i in range(1,31):
        #     df_train_unscaled[f"FReturn{i}"] = df_train_unscaled["Close"].pct_change(i)
        df_train_unscaled[f"FReturn"] = df_train_unscaled[[f"FReturn{i}" for i in range(1,31)]].max( axis=1)
        df_train_unscaled = df_train_unscaled.drop(columns=[f"FReturn{i}" for i in range(1,30)])
        df_train_unscaled.dropna(inplace=True)
        df_train_unscaled = rolling_percentile(df_train_unscaled, 300)
        df_cluster = pd.concat([df_cluster, df_train_unscaled[["FReturn", "FReturn30", "Percentile"]]], axis=1)
        df_cluster = df_cluster[df_cluster.Percentile>85]
        df_cluster = df_cluster.drop(columns=["Percentile"])

    # if how == "Upside":
    #     df_train_unscaled[f"FReturn"] = df_train_unscaled["Close"].pct_change(30)
    #     df_train_unscaled.dropna(inplace=True)
    #     df_train_unscaled = rolling_percentile(df_train_unscaled, 300)
    #     df_cluster = pd.concat([df_cluster, df_train_unscaled[["FReturn", "Percentile"]]], axis=1)
    #     df_cluster = df_cluster[df_cluster.Percentile > 85]
    #     df_cluster = df_cluster.drop(columns=["Percentile"])
    #
    # if how == "DownsideMinAll":
    #     for i in range(1,31):
    #         df_train_unscaled[f"FReturn{i}"] = df_train_unscaled["Close"].pct_change(i)
    #     df_train_unscaled[f"FReturn"] = df_train_unscaled[[f"FReturn{i}" for i in range(1,31)]].min( axis=1)
    #     df_train_unscaled = df_train_unscaled.drop(columns=[f"FReturn{i}" for i in range(1,30)])
    #     df_train_unscaled.dropna(inplace=True)
    #     df_cluster = pd.concat([df_cluster, df_train_unscaled[["FReturn","FReturn30"]]], axis=1)

    if how == "DownsideMin":
        # for i in range(1,31):
        #     df_train_unscaled[f"FReturn{i}"] = df_train_unscaled["Close"].pct_change(i)
        df_train_unscaled[f"FReturn"] = df_train_unscaled[[f"FReturn{i}" for i in range(1,31)]].min( axis=1)
        df_train_unscaled = df_train_unscaled.drop(columns=[f"FReturn{i}" for i in range(1,30)])
        df_train_unscaled.dropna(inplace=True)
        df_train_unscaled = rolling_percentile(df_train_unscaled, 300)
        df_cluster = pd.concat([df_cluster, df_train_unscaled[["FReturn", "FReturn30", "Percentile"]]], axis=1)
        df_cluster = df_cluster[df_cluster.Percentile < 15]
        df_cluster = df_cluster.drop(columns=["Percentile"])

    # if how == "Downside":
    #     df_train_unscaled[f"FReturn"] = df_train_unscaled["Close"].pct_change(30)
    #     df_train_unscaled.dropna(inplace=True)
    #     df_train_unscaled = rolling_percentile(df_train_unscaled, 300)
    #     df_cluster = pd.concat([df_cluster, df_train_unscaled[["FReturn", "Percentile"]]], axis=1)
    #     df_cluster = df_cluster[df_cluster.Percentile < 15]
    #     df_cluster = df_cluster.drop(columns=["Percentile"])

    df_cluster = df_cluster.drop(columns=["BinaryOutcome", "Open", "High", "Low", "Close", 'Max_CB_20', 'Min_CB_20',
           'Max_CB_30', 'Min_CB_30', 'Max_CB_60', 'Min_CB_60', 'Max_FMACB_20', 'Min_FMACB_20', 'Max_FMACB_30',
           'Min_FMACB_30', 'Max_FMACB_60', 'Min_FMACB_60', 'Max_SMACB_20', 'Min_SMACB_20', 'Max_SMACB_30',
           'Min_SMACB_30', 'Max_SMACB_60', 'Min_SMACB_60',
           'ROC_CB_20', 'ROC_CB_30', 'ROC_CB_60', 'ROC_FMACB_20',
           'ROC_FMACB_30', 'ROC_FMACB_60', 'Convexity_FMACB',
           'ROC_SMACB_20', 'ROC_SMACB_30', 'ROC_SMACB_60'])

    return df_cluster

def rolling_percentile(df, lookback):
    df["Percentile"] = np.nan
    for i in range(len(df)):
        if i>=lookback-1:
            df.loc[df.index[i], "Percentile"] = stats.percentileofscore(df.iloc[i - lookback + 1:i]["FReturn"], df.iloc[i]["FReturn"])
        else:
            continue
    return df
