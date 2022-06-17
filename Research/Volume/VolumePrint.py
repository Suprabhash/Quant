Skip
to
content
Search or jump
to…
Pull
requests
Issues
Marketplace
Explore


@AcsysAlgo


AcsysAlgo
/
DataRetrieval
Private
Code
Issues
Pull
requests
Actions
Projects
Security
Insights
Settings
DataRetrieval / data / VolumeLevelsCreation / VolumeLevelsCreation.py /


@AcsysAlgo


AcsysAlgo
clustering
of
orange
days, important
methods
separated
Latest
commit
fa8b6ef
15
days
ago
History
1
contributor
5117
lines(4342
sloc)  284
KB

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 15)
# plt.rcParams['axes.grid'] = False
import seaborn as sns

sns.set_style("whitegrid", {'axes.grid': False})
from tqdm import tqdm
from datetime import timedelta
import pandas as pd
import ssl
import datetime
import pickle
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from pathos.multiprocessing import ProcessingPool
from config import *
import importlib.util
from azure.data.tables import TableServiceClient, UpdateMode
from azure.core.credentials import AzureNamedKeyCredential
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from Optimisers.Optimiser import *
import statistics
from functools import partial
from numba import jit
import scipy.stats as stat

# from sklearn.mixture import GaussianMixture
# from sklearn.cluster import KMeans
# from sklearn import svm

spec = importlib.util.spec_from_file_location("account_name_and_key",
                                              "Z:\\Algo\\keys_and_passwords\\Azure\\account_name_and_key.py")
azure_keys = importlib.util.module_from_spec(spec)
spec.loader.exec_module(azure_keys)
_STORAGE_ACCOUNT_NAME = azure_keys._STORAGE_ACCOUNT_NAME
_STORAGE_ACCOUNT_KEY = azure_keys._STORAGE_ACCOUNT_KEY
_TABLE_SERVIVCE_ENDPOINT = azure_keys._TABLE_SERVIVCE_ENDPOINT

spec = importlib.util.spec_from_file_location("email_and_password",
                                              "Z:\\Algo\\keys_and_passwords\\Gmail\\email_and_password.py")
email_and_password = importlib.util.module_from_spec(spec)
spec.loader.exec_module(email_and_password)
sender_email = email_and_password.sender_email
sender_password = email_and_password.sender_password
receiver_email1 = email_and_password.receiver_email1
receiver_email2 = email_and_password.receiver_email2
receiver_email3 = email_and_password.receiver_email3

spec = importlib.util.spec_from_file_location("current_nifty_tickers",
                                              "Z:\\Algo\\data_retrieval\\Tickers\\current_nifty_tickers.py")
current_nifty_tickers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(current_nifty_tickers)

current_nifty_tickers = current_nifty_tickers.current_nifty_tickers


def get_data(ticker, frequency):
    """
    :param ticker: Ticker as on Reuters. Investpy and yfinance tickers can be passed using the lookup dict in tickers.py
    :param frequency: Frequency of the data required. Currently supports daily and hourly. Pass "D" or "H"
    :return:  Returns the OHLCV dataframe indexed by datetime
    """

    credential = AzureNamedKeyCredential(_STORAGE_ACCOUNT_NAME, _STORAGE_ACCOUNT_KEY)
    table_service = TableServiceClient(endpoint="https://acsysbatchstroageacc.table.core.windows.net/",
                                       credential=credential)
    if frequency == 'H':
        table_client = table_service.get_table_client(table_name="HourlyData")
    if frequency == 'D':
        table_client = table_service.get_table_client(table_name="DailyData")

    tasks = table_client.query_entities(query_filter=f"PartitionKey eq '{ticker}'")
    list_dict = []
    for i in tasks:
        list_dict.append(i)

    ticker_dataframe = pd.DataFrame(list_dict)
    ticker_dataframe.drop(columns=["PartitionKey", "RowKey"], inplace=True)
    ticker_dataframe.drop(columns="API", inplace=True)
    ticker_dataframe[["Open", "High", "Low", "Close", "Volume"]] = ticker_dataframe[
        ["Open", "High", "Low", "Close", "Volume"]].astype(float)
    if 'Date' in ticker_dataframe.columns:
        ticker_dataframe.rename(columns={'Date': 'Datetime'}, inplace=True)
    ticker_dataframe["Datetime"] = pd.to_datetime(ticker_dataframe["Datetime"])
    return ticker_dataframe


def get_data_ETH_minute(path):
    filepaths = ["https://www.cryptodatadownload.com/cdd/Bitstamp_ETHUSD_2017_minute.csv", ]
    # "https://www.cryptodatadownload.com/cdd/Bitstamp_ETHUSD_2018_minute.csv",
    # "https://www.cryptodatadownload.com/cdd/Bitstamp_ETHUSD_2019_minute.csv",
    # "https://www.cryptodatadownload.com/cdd/Bitstamp_ETHUSD_2020_minute.csv",
    # "https://www.cryptodatadownload.com/cdd/Bitstamp_ETHUSD_2021_minute.csv"]

    data = pd.DataFrame()
    for filepath in filepaths:
        ssl._create_default_https_context = ssl._create_unverified_context
        temp_og = pd.read_csv(filepath, parse_dates=True, skiprows=1)
        temp_og.drop(columns=["date", "symbol", "Volume USD"], inplace=True)
        temp_og["unix"] = temp_og["unix"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000) if len(
            str(int(x))) == 13 else datetime.datetime.fromtimestamp(x))
        temp_og.rename(columns={"unix": "Datetime", "open": "Open", "high": "High", "low": "Low", "close": "Close",
                                "Volume ETH": "Volume"}, inplace=True)
        temp_og["Datetime"] = pd.to_datetime(temp_og["Datetime"])
        temp_og["Datetime"] = temp_og["Datetime"] - timedelta(hours=4, minutes=30)
        temp_og.sort_values("Datetime", ascending=True, inplace=True)
        temp_og = temp_og[temp_og["Datetime"] > "2017-08-18"]
        temp_og.reset_index(drop=True, inplace=True)
        data = pd.concat([data, temp_og])

    if os.path.isdir(path):
        with open(path, 'rb') as file:
            temp_og_imp = pickle.load(file)
        temp_og = pd.concat([temp_og_imp, data], axis=0)
        temp_og.drop_duplicates(keep="first", inplace=True)
        temp_og.reset_index(drop=True, inplace=True)
        with open(path, 'wb') as file:
            pickle.dump(pd.DataFrame(temp_og), file)
    else:
        with open(path, 'wb') as file:
            pickle.dump(pd.DataFrame(data), file)

    return data


def resample_data(df, minutes):
    return df.set_index("Datetime").groupby(pd.Grouper(freq=f'{minutes}Min')).agg({"Open": "first",
                                                                                   "Close": "last",
                                                                                   "Low": "min",
                                                                                   "High": "max",
                                                                                   "Volume": "sum"}).reset_index()


def return_volume_features_minute_hourly_helper(args):
    [total_length, df_hour, df_min, start_index, end_index] = args
    # [total_length, df_hour, df_min, start_index,end_index] = args[100]
    start = datetime.datetime.now().strftime("%Y_%m_%d_%HH_%MM_%SS")
    batch_list = []
    res = {}
    x = range(start_index, end_index + 1, 1)
    for i in x:
        try:
            status = f"{i}/{total_length}-started" + "\n"
            print(status)
            res = {}
            res["Datetime"] = df_hour.loc[i]["Datetime"]
            try:
                for n in lookback_periods:  # number of hours for lookback of 10 days,30 days,3 months,6months,1year
                    if i >= n - 1:
                        volumes, high_prices, low_prices = return_dataframe_minute(df_min, df_hour, i, n)
                        res[f"CalcHow_{n}"] = "Minute"
                        price_levels = calc_distribution(high_prices, low_prices, volumes)
                        res[f"PriceLevels_{n}"] = price_levels
                    else:
                        res[f"PriceLevels_{n}"] = {}
                        res[f"CalcHow_{n}"] = "DataNotAvailable"
                    status = f"[{start}]-{i}/{total_length}-finished" + "\n"
                batch_list.append(res)
            except Exception as e:
                res[f"PriceLevels_{n}"] = {}
                res[f"CalcHow_{n}"] = "DataNotAvailable"
                status = f"[{start}]-{i}/{total_length}-{e}" + "\n"
                batch_list.append(res)
        except Exception as e:
            print(e)
        f = open("VolumeLevelsLog.txt", "a")
        f.write(status)
        f.close()

    return batch_list


def return_volume_features_minute_hourly(df_hour, df_min):
    inputs = []
    max_lookback_hours = max(lookback_periods)
    max_lookback_minutes = max_lookback_hours * 60
    total_length = len(df_hour)
    i = 0

    while i <= total_length:
        start_index = i
        end_index = i + max_lookback_hours * 3
        print(start_index, end_index)
        if start_index > max_lookback_hours:
            df_hour_for_lookback = df_hour.iloc[start_index - max_lookback_hours:end_index + 1]
            df_min_for_lookback = df_min.iloc[((start_index - max_lookback_hours) * 60):(60 * end_index) + 1]
        else:
            df_hour_for_lookback = df_hour.iloc[0:end_index + 1]
            df_min_for_lookback = df_min.iloc[0:(60 * end_index) + 1]
        inputs.append([total_length, df_hour_for_lookback, df_min_for_lookback, start_index, end_index])
        i = end_index + 1

    pool = ProcessingPool()
    res = pool.map(return_volume_features_minute_hourly_helper, inputs)
    pool.clear()

    # for i in range(len(inputs)):
    #     res = return_volume_features_minute_hourly_helper(inputs)

    list_for_df = []
    for i in res:
        for j in i:
            list_for_df.append(j)

    return pd.DataFrame(list_for_df)


def return_dataframe_minute(temp_og1, temp_og, i, n):
    volumes = list(temp_og1[(temp_og1["Datetime"] > pd.to_datetime(
        datetime.datetime(temp_og["Datetime"].loc[i - n].year, temp_og["Datetime"].loc[i - n].month,
                          temp_og["Datetime"].loc[i - n].day, temp_og["Datetime"].loc[i - n].hour,
                          temp_og["Datetime"].loc[i - n].minute))) &
                            (temp_og1["Datetime"] <= pd.to_datetime(
                                datetime.datetime(temp_og["Datetime"].loc[i].year, temp_og["Datetime"].loc[i].month,
                                                  temp_og["Datetime"].loc[i].day, temp_og["Datetime"].loc[i].hour,
                                                  temp_og["Datetime"].loc[i].minute)))]["Volume"])
    high_prices = list(temp_og1[(temp_og1["Datetime"] > pd.to_datetime(
        datetime.datetime(temp_og["Datetime"].loc[i - n].year, temp_og["Datetime"].loc[i - n].month,
                          temp_og["Datetime"].loc[i - n].day, temp_og["Datetime"].loc[i - n].hour,
                          temp_og["Datetime"].loc[i - n].minute))) &
                                (temp_og1["Datetime"] <= pd.to_datetime(
                                    datetime.datetime(temp_og["Datetime"].loc[i].year,
                                                      temp_og["Datetime"].loc[i].month,
                                                      temp_og["Datetime"].loc[i].day, temp_og["Datetime"].loc[i].hour,
                                                      temp_og["Datetime"].loc[i].minute)))]["High"])
    low_prices = list(temp_og1[(temp_og1["Datetime"] > pd.to_datetime(
        datetime.datetime(temp_og["Datetime"].loc[i - n].year, temp_og["Datetime"].loc[i - n].month,
                          temp_og["Datetime"].loc[i - n].day, temp_og["Datetime"].loc[i - n].hour,
                          temp_og["Datetime"].loc[i - n].minute))) &
                               (temp_og1["Datetime"] <= pd.to_datetime(
                                   datetime.datetime(temp_og["Datetime"].loc[i].year,
                                                     temp_og["Datetime"].loc[i].month,
                                                     temp_og["Datetime"].loc[i].day, temp_og["Datetime"].loc[i].hour,
                                                     temp_og["Datetime"].loc[i].minute)))]["Low"])

    return volumes, high_prices, low_prices


# Deprecated or maybe useful for crypto hourly/minute data
def calc_distribution_old(highs, lows, volumes, plot_hist=False):
    x = []
    y = []
    for i in range(len(volumes)):
        prices = np.round(np.linspace(lows[i], highs[i], num=10), 2)
        for j in range(10):
            x.append(prices[j])
            y.append(volumes[i] / 10)
    prices = np.linspace(min(x), max(x), num=25)
    p = [0] * (len(prices) - 1)
    v = [0] * (len(prices) - 1)
    for j in range(len(prices) - 1):
        p[j] = (prices[j] + prices[j + 1]) / 2
    for i in range(len(x)):
        for j in range(len(prices) - 1):
            if (x[i] > prices[j] - 0.001) & (x[i] <= prices[j + 1]):
                v[j] = v[j] + (y[i])

    poc = p[v.index(max(v))]
    profile_high = max(highs)
    profile_low = min(lows)
    target_volume = 0.7 * sum(v)
    vol = max(v)
    bars_in_value_area = [v.index(max(v))]
    while vol < target_volume:
        # print("*"*100)
        # print(f"Target vol: {target_volume}")
        # print(f"Vol before: {vol}")
        # print(f"bars_in_value_area before: {bars_in_value_area}")
        if max(bars_in_value_area) > 21:
            vol_above = 0
        else:
            vol_above = v[max(bars_in_value_area) + 1] + v[max(bars_in_value_area) + 2]
        if min(bars_in_value_area) < 2:
            vol_below = 0
        else:
            vol_below = v[min(bars_in_value_area) - 1] + v[min(bars_in_value_area) - 2]
        if vol_above > vol_below:
            if max(bars_in_value_area) < 22:
                vol = vol + vol_above
                bars_in_value_area.extend([max(bars_in_value_area) + 1, max(bars_in_value_area) + 2])
            else:
                vol = vol + vol_below
                bars_in_value_area.extend([min(bars_in_value_area) - 1, min(bars_in_value_area) - 2])
        else:
            if min(bars_in_value_area) > 1:
                vol = vol + vol_below
                bars_in_value_area.extend([min(bars_in_value_area) - 1, min(bars_in_value_area) - 2])
            else:
                vol = vol + vol_above
                bars_in_value_area.extend([max(bars_in_value_area) + 1, max(bars_in_value_area) + 2])
        bars_in_value_area.sort()
        # print(f"bars_in_value_area after: {bars_in_value_area}")
        # print(f"Vol after: {vol}")
        if bars_in_value_area[-1] > 30:
            raise NameError("Hi there")
    vah = p[max(bars_in_value_area)]
    val = p[min(bars_in_value_area)]
    if plot_hist:
        plt.bar(p, v)
        plt.plot(p, v)
        plt.axvline(x=poc, color="orange")
        plt.axvline(x=profile_high, color="darkgreen")
        plt.axvline(x=profile_low, color="maroon")
        plt.axvline(x=vah, color="lime")
        plt.axvline(x=val, color="red")
        plt.legend(["Point of Control", "Profile High", "Profile Low", "Value Area High", "Value Area Low"])
    return {"poc": poc, "profile_high": profile_high, "profile_low": profile_low, "vah": vah, "val": val}


def calc_distribution(daily_chunks, start_index, end_index, plot_hist=False):
    results = {}
    for i in range(start_index, end_index):
        results[daily_chunks[i]["Datetime"]] = {}

    for n in lookback_periods_in_days:
        highs = []
        lows = []
        volumes = []
        for i in range(start_index, end_index):
            try:
                # status = f"{i}/{total_length}-started" + "\n"
                # print(status)
                if i - n < 0:
                    results[daily_chunks[i]["Datetime"]][f"PriceLevels_{n}"] = {}
                    results[daily_chunks[i]["Datetime"]][f"CalcHow_{n}"] = f"NoDataAvailable"
                elif i > start_index:
                    highs = highs[number_of_intraday_low_high_chunks:]
                    highs.extend(daily_chunks[i]["highs"])
                    lows = lows[number_of_intraday_low_high_chunks:]
                    lows.extend(daily_chunks[i]["lows"])
                    volumes = volumes[number_of_intraday_low_high_chunks:]
                    volumes.extend(daily_chunks[i]["volumes"])
                elif i == start_index:
                    for j in range(i - n, i + 1):
                        highs.extend(daily_chunks[j]["highs"])
                        lows.extend(daily_chunks[j]["lows"])
                        volumes.extend(daily_chunks[j]["volumes"])

                results[daily_chunks[i]["Datetime"]][f"PriceLevels_{n}"] = value_area_calculation(lows, highs, volumes,
                                                                                                  plot_hist)
                results[daily_chunks[i]["Datetime"]][
                    f"CalcHow_{n}"] = f"Daily data distributed uniformly over {number_of_intraday_low_high_chunks} chunks"
            except Exception as e:
                results[daily_chunks[i]["Datetime"]][f"PriceLevels_{n}"] = {}
                results[daily_chunks[i]["Datetime"]][f"CalcHow_{n}"] = f"NoDataAvailable"
                print(e)

            # value_area_results =
            # results["Datetime"] = daily_chunks[i]["Datetime"]

    return results


def calculate_confluence_points(volume_features_path):
    with open(f'{volume_features_path}', 'rb') as file:
        volume_df = pickle.load(file)

    price_level_columns = [i for i in volume_df.columns if i.startswith("PriceLevels")]

    for i in price_level_columns:
        for j in volume_df.index:
            if (bool(volume_df[i][j]) == False) or pd.isnull(volume_df[i][j]):
                if j > 0:
                    volume_df[i][j] = volume_df[i][j - 1]
                else:
                    volume_df[i][j] = volume_df[i][0]


def value_area_calculation(lows, highs, volumes, plot_hist):
    high_over_lookbacks = max(highs)
    low_over_lookbacks = min(lows)
    price_levels_over_lookbacks = list(
        np.linspace(low_over_lookbacks, high_over_lookbacks, low_high_chunks_over_lookbacks))
    volumes_over_price_levels = [0] * low_high_chunks_over_lookbacks

    for i in range(len(lows)):
        for j in range(len(price_levels_over_lookbacks) - 1):
            if (price_levels_over_lookbacks[j] > highs[i]) and (price_levels_over_lookbacks[j + 1] > highs[i]):
                continue
            elif (price_levels_over_lookbacks[j] < lows[i]) and (price_levels_over_lookbacks[j + 1] < lows[i]):
                continue
            elif (price_levels_over_lookbacks[j] <= lows[i]) and (price_levels_over_lookbacks[j + 1] >= highs[i]):
                volumes_over_price_levels[j] += volumes[i]
            elif (price_levels_over_lookbacks[j] > lows[i]):
                if (price_levels_over_lookbacks[j + 1] < highs[i]):
                    volumes_over_price_levels[j] += ((price_levels_over_lookbacks[j + 1] - price_levels_over_lookbacks[
                        j]) / (highs[i] - lows[i])) * volumes[i]
                elif (price_levels_over_lookbacks[j + 1] > highs[i]):
                    volumes_over_price_levels[j] += ((highs[i] - price_levels_over_lookbacks[j]) / (
                            highs[i] - lows[i])) * volumes[i]

    max_volume_index = volumes_over_price_levels.index(max(volumes_over_price_levels))
    if max_volume_index >= len(price_levels_over_lookbacks) - 1:
        poc = price_levels_over_lookbacks[max_volume_index]
    else:
        poc = (price_levels_over_lookbacks[max_volume_index] + price_levels_over_lookbacks[max_volume_index + 1]) / 2

    value_area_threshold_volume = value_area_threshold * sum(volumes_over_price_levels)
    value_area_bars_indices = [max_volume_index]
    sum_of_value_area_bars_volume = volumes_over_price_levels[max_volume_index]

    i = j = max_volume_index
    while sum_of_value_area_bars_volume < value_area_threshold_volume:
        if i - 2 >= 0 and j + 2 < len(volumes_over_price_levels):
            if volumes_over_price_levels[i - 1] + volumes_over_price_levels[i - 2] > volumes_over_price_levels[j + 1] + \
                    volumes_over_price_levels[j + 2]:
                sum_of_value_area_bars_volume += volumes_over_price_levels[i - 1] + volumes_over_price_levels[i - 2]
                value_area_bars_indices.extend([i - 1, i - 2])
                i = i - 2
            else:
                sum_of_value_area_bars_volume += volumes_over_price_levels[j + 1] + volumes_over_price_levels[j + 2]
                value_area_bars_indices.extend([j + 1, j + 2])
                j = j + 2
        elif i - 2 < 0:
            sum_of_value_area_bars_volume += volumes_over_price_levels[j + 1] + volumes_over_price_levels[j + 2]
            value_area_bars_indices.extend([j + 1, j + 2])
            j = j + 2
        elif j + 2 >= len(volumes_over_price_levels):
            sum_of_value_area_bars_volume += volumes_over_price_levels[i - 1] + volumes_over_price_levels[i - 2]
            value_area_bars_indices.extend([i - 1, i - 2])
            i = i - 2

    prices_of_value_area = [price_levels_over_lookbacks[i] for i in value_area_bars_indices]
    profile_high = max(highs)
    profile_low = min(lows)
    vah = max(prices_of_value_area)
    val = min(prices_of_value_area)

    index_of_least_volume_in_value_area = max_volume_index
    for i in value_area_bars_indices:
        if volumes_over_price_levels[i] < volumes_over_price_levels[index_of_least_volume_in_value_area]:
            index_of_least_volume_in_value_area = i

    # if index_of_least_volume_in_value_area < len(price_levels_over_lookbacks):
    point_of_least_volume_in_va = (price_levels_over_lookbacks[index_of_least_volume_in_value_area - 1] +
                                   price_levels_over_lookbacks[index_of_least_volume_in_value_area]) / 2
    # else:
    #     point_of_least_volume_in_va = (price_levels_over_lookbacks[index_of_least_volume_in_value_area] +
    #                                            price_levels_over_lookbacks[index_of_least_volume_in_value_area - 1]) / 2
    price_levels_plot_labels = [str(round(i, 2)) for i in price_levels_over_lookbacks]

    if plot_hist:
        value_area_bars_indices.sort()
        volumes_of_value_area = [volumes_over_price_levels[i] for i in value_area_bars_indices]
        prices_of_value_area = [price_levels_over_lookbacks[i] for i in value_area_bars_indices]
        prices_of_value_area_labels = [str(round(i, 2)) for i in prices_of_value_area]
        df_daily = get_data(".NSEI", "D")
        nifty_x = df_daily["Datetime"].to_list()
        nifty_y = [round(i, 2) for i in df_daily["Close"].to_list()]
        plt.subplot(121)
        plt.barh(price_levels_plot_labels, volumes_over_price_levels)
        plt.subplot(122)
        plt.plot(nifty_x, nifty_y)
        plt.subplot(121)
        plt.barh(prices_of_value_area_labels, volumes_of_value_area)
        plt.show()

    return {"poc": round(poc, 2), "profile_high": round(profile_high, 2), "profile_low": round(profile_low, 2),
            "vah": round(vah, 2), "val": round(val, 2),
            "point_of_least_volume_in_va": round(point_of_least_volume_in_va, 2)}


def value_area_calculation_for_any_histogram(values, plot=False, plot_path=None, plot_label=None,
                                             cache_path="test_hist_value_area.pkl"):
    step = returns_histogram_bucket_size
    start = np.floor(min(values) / step) * step
    stop = max(values) + step
    bin_edges = np.arange(start, stop, step=step)
    histogram_profile = np.histogram(values, bins=bin_edges)
    frequency_values = list(histogram_profile[0])
    max_frequency_index = frequency_values.index(max(frequency_values))
    if max_frequency_index >= len(bin_edges) - 1:
        poc = bin_edges[max_frequency_index]
    else:
        poc = (bin_edges[max_frequency_index] + bin_edges[max_frequency_index + 1]) / 2

    value_area_threshold_frequency = value_area_threshold * len(values)
    value_area_bars_indices = [max_frequency_index]
    sum_of_value_area_bars_volume = frequency_values[max_frequency_index]

    i = j = max_frequency_index
    while sum_of_value_area_bars_volume < value_area_threshold_frequency:
        if i - 2 >= 0 and j + 2 < len(frequency_values):
            if frequency_values[i - 1] + frequency_values[i - 2] > frequency_values[j + 1] + \
                    frequency_values[j + 2]:
                sum_of_value_area_bars_volume += frequency_values[i - 1] + frequency_values[i - 2]
                value_area_bars_indices.extend([i - 1, i - 2])
                i = i - 2
            else:
                sum_of_value_area_bars_volume += frequency_values[j + 1] + frequency_values[j + 2]
                value_area_bars_indices.extend([j + 1, j + 2])
                j = j + 2
        elif i - 2 < 0:
            sum_of_value_area_bars_volume += frequency_values[j + 1] + frequency_values[j + 2]
            value_area_bars_indices.extend([j + 1, j + 2])
            j = j + 2
        elif j + 2 >= len(frequency_values):
            sum_of_value_area_bars_volume += frequency_values[i - 1] + frequency_values[i - 2]
            value_area_bars_indices.extend([i - 1, i - 2])
            i = i - 2

    bins_of_value_area = [bin_edges[i] for i in value_area_bars_indices]
    vah = max(bins_of_value_area)
    val = min(bins_of_value_area)

    if plot:
        plt.hist(values, bins=bin_edges, label=plot_label)
        start = np.floor(min(bins_of_value_area) / step) * step
        stop = max(bins_of_value_area) + step
        bin_edges_value_area = np.arange(start, stop, step=step)
        plt.hist(values, bins=bin_edges_value_area, color="orange", label=f"value_area")
        plt.xticks(bin_edges, rotation=90)
        plt.tight_layout()
        plt.xlabel("returns")
        plt.legend()
        plt.savefig(plot_path)
        plt.close()

    return {"poc": poc, "vah": vah, "val": val}


def return_resampled_features_daily(df_daily):
    daily_chunks = {}
    for i in df_daily.index:
        volumes = []
        high_prices = []
        low_prices = []
        try:
            low = df_daily.loc[i]["Low"]
            high = df_daily.loc[i]["High"]
            chunk_size = (high - low) / number_of_intraday_low_high_chunks
            volume = df_daily.loc[i]["Volume"] / number_of_intraday_low_high_chunks
            for k in range(number_of_intraday_low_high_chunks):
                low_prices.append(low)
                high_prices.append(low + chunk_size)
                low = low + chunk_size
                volumes.append(volume)
            daily_chunks[i] = {"Datetime": df_daily.loc[i]["Datetime"], "lows": low_prices, "highs": high_prices,
                               "volumes": volumes}
        except Exception as e:
            print(e)
    return daily_chunks


def return_volume_features_daily_helper(args):
    [total_length, df_daily, start_index, end_index] = args
    # [total_length, df_daily, start_index, end_index] = args[-2]
    start = datetime.datetime.now().strftime("%Y-%m-%d-%HH:%MM:%SS")

    if end_index > max(df_daily.index):
        end_index = max(df_daily.index) + 1

    daily_chunks = return_resampled_features_daily(df_daily)

    results = {}
    for i in range(start_index, end_index + 1):
        try:
            results[daily_chunks[i]["Datetime"]] = {}
        except Exception as e:
            print(e)

    for n in lookback_periods_in_days:
        highs = []
        lows = []
        volumes = []
        for i in range(start_index, end_index + 1):
            try:
                status = f"{i}/{total_length}-started" + "\n"
                print(status)
                if i - n < 0:
                    results[daily_chunks[i]["Datetime"]][f"PriceLevels_{n}"] = {}
                    results[daily_chunks[i]["Datetime"]][f"CalcHow_{n}"] = f"NoDataAvailable"
                elif i > start_index:
                    highs = highs[number_of_intraday_low_high_chunks:]
                    highs.extend(daily_chunks[i]["highs"])
                    lows = lows[number_of_intraday_low_high_chunks:]
                    lows.extend(daily_chunks[i]["lows"])
                    volumes = volumes[number_of_intraday_low_high_chunks:]
                    volumes.extend(daily_chunks[i]["volumes"])
                elif i == start_index:
                    for j in range(i - n, i + 1):
                        highs.extend(daily_chunks[j]["highs"])
                        lows.extend(daily_chunks[j]["lows"])
                        volumes.extend(daily_chunks[j]["volumes"])

                results[daily_chunks[i]["Datetime"]][f"PriceLevels_{n}"] = value_area_calculation(lows, highs, volumes,
                                                                                                  plot_hist=False)
                results[daily_chunks[i]["Datetime"]][
                    f"CalcHow_{n}"] = f"Daily data distributed uniformly over {number_of_intraday_low_high_chunks} chunks"
                status = f"[{start}]-{i}/{total_length}-done" + "\n"
            except Exception as e:
                try:
                    results[daily_chunks[i]["Datetime"]][f"PriceLevels_{n}"] = {}
                    results[daily_chunks[i]["Datetime"]][f"CalcHow_{n}"] = f"NoDataAvailable"
                    status = f"{i}/{total_length}-failed" + "\n"
                    print(status)
                    status = f"[{start}]-{i}/{total_length}-{e}" + "\n"
                except Exception as err:
                    print(err)
                # print(e)
            f = open("VolumeLevelsLog.txt", "a")
            f.write(status)
            f.close()

    return results


def return_volume_features_daily(df_daily, ticker=".NSEI"):
    inputs = []
    max_lookback_days = max(lookback_periods_in_days)
    # max_lookback_minutes = max_lookback_hours * 60
    total_length = len(df_daily)
    i = 0
    chunk_size_for_multiprocessing = round(len(df_daily) / multprocessing_chunk_divider) + 1

    while i <= total_length:
        start_index = i
        end_index = i + chunk_size_for_multiprocessing
        # end_index = i + 100
        print(start_index, end_index)
        if start_index > max_lookback_days:
            df_daily_for_lookback = df_daily.iloc[start_index - max_lookback_days:end_index + 1]
        else:
            df_daily_for_lookback = df_daily.iloc[0:end_index + 1]
        inputs.append([total_length, df_daily_for_lookback, start_index, end_index])
        i = end_index + 1

    pool = ProcessingPool(nodes=multprocessing_chunk_divider)
    res = pool.map(return_volume_features_daily_helper, inputs)
    pool.clear()

    # for i in range(0,1):
    #     res = return_volume_features_daily_helper(inputs)

    dict_for_df = {}
    for i in res:
        dict_for_df.update(i)

    volume_features_daily = pd.DataFrame.from_dict(dict_for_df, orient='index')

    for i in volume_features_daily.index:
        for j in volume_features_daily.columns:
            if j.startswith("PriceLevels"):
                if bool(volume_features_daily.loc[i][j]) == False:
                    if list(volume_features_daily.columns).index(j) != 0:
                        volume_features_daily.at[i, j] = volume_features_daily.loc[i][
                            list(volume_features_daily.columns)[list(volume_features_daily.columns).index(j) - 2]]

    return volume_features_daily  # pd.DataFrame.from_dict(res, orient='index')


def return_resampled_features_of_hourly_candles(df_daily):
    daily_chunks = {}
    for i in df_daily.index:
        volumes = []
        high_prices = []
        low_prices = []
        try:
            low = df_daily.loc[i]["Low"]
            high = df_daily.loc[i]["High"]
            chunk_size = (high - low) / number_of_intraday_low_high_chunks
            volume = df_daily.loc[i]["Volume"] / number_of_intraday_low_high_chunks
            for k in range(number_of_intraday_low_high_chunks):
                low_prices.append(low)
                high_prices.append(low + chunk_size)
                low = low + chunk_size
                volumes.append(volume)
            daily_chunks[df_daily.loc[i]["Datetime"]] = {"lows": low_prices, "highs": high_prices, "volumes": volumes}
        except Exception as e:
            print(e)
    return daily_chunks


def return_volume_features_daily_based_on_hourly_candles_single_lookback_which_is_number_of_hourly_candles_for_each_day_helper(
        args):
    [total_length, df_hourly, dates_list] = args
    # [total_length, df_hourly, dates_list] = args[-2]
    start = datetime.datetime.now().strftime("%Y-%m-%d-%HH:%MM:%SS")

    daily_chunks = return_resampled_features_of_hourly_candles(df_hourly)

    results = {}
    for i in dates_list:
        try:
            results[i] = {}
        except Exception as e:
            print(e)

    for i in dates_list:
        try:
            highs = []
            lows = []
            volumes = []
            intraday_candle_times = [k for k in df_hourly["Datetime"] if k.strftime("%Y-%m-%d") == i]
            for j in intraday_candle_times:
                highs.extend(daily_chunks[j]["highs"])
                lows.extend(daily_chunks[j]["lows"])
                volumes.extend(daily_chunks[j]["volumes"])

            value_area_details = value_area_calculation(lows, highs, volumes, plot_hist=False)
            results[i]["poc"] = value_area_details["poc"]
            results[i]["vah"] = value_area_details["vah"]
            results[i]["val"] = value_area_details["val"]
            results[i]["point_of_least_volume_in_va"] = value_area_details["point_of_least_volume_in_va"]

            results[i][
                f"CalcHow"] = f"Hourly candles distributed uniformly over {number_of_intraday_low_high_chunks} chunks"
            status = f"[{start}]-{i}/{total_length}-done" + "\n"
        except Exception as e:
            try:
                results[i] = {}
                results[i][f"CalcHow"] = f"NoDataAvailable"
                status = f"{i}/{total_length}-failed" + "\n"
                print(status)
                status = f"[{start}]-{i}/{total_length}-{e}" + "\n"
            except Exception as err:
                print(err)
                # print(e)
                f = open("VolumeLevelsLog.txt", "a")
                f.write(status)
                f.close()

    return results


def return_volume_features_daily_based_on_hourly_candles_single_lookback_which_is_number_of_hourly_candles_for_each_day(
        df_hourly, ticker=".NSEI"):
    inputs = []
    dates_list = list(set([i.strftime("%Y-%m-%d") for i in df_hourly["Datetime"]]))
    dates_list.sort()
    total_length = len(dates_list)
    chunk_size_for_multiprocessing = round(len(df_hourly) / multprocessing_chunk_divider) + 1

    for i in range(0, total_length, chunk_size_for_multiprocessing):
        j = i + chunk_size_for_multiprocessing
        if j > total_length:
            j = total_length - 1
        print(i, j)
        df_hourly_for_lookback = df_hourly[
            (df_hourly["Datetime"] >= dates_list[i]) & (df_hourly["Datetime"] <= dates_list[j])]
        inputs.append([total_length, df_hourly_for_lookback, dates_list[i:j]])

    pool = ProcessingPool(nodes=multprocessing_chunk_divider)
    res = pool.map(
        return_volume_features_daily_based_on_hourly_candles_single_lookback_which_is_number_of_hourly_candles_for_each_day_helper,
        inputs)
    pool.clear()

    # for i in range(0,1):
    #     res = return_volume_features_daily_based_on_hourly_candles_single_lookback_which_is_number_of_hourly_candles_for_each_day_helper(inputs)

    dict_for_df = {}
    for i in res:
        dict_for_df.update(i)

    return pd.DataFrame.from_dict(dict_for_df, orient='index')  # pd.DataFrame.from_dict(res, orient='index')


def plot_volume_level_points_for_n_days(df_daily, cache_path):
    df_daily = df_daily.iloc[-number_of_days_for_volume_levels_plot - 1:]

    with open(f'{cache_path}', 'rb') as file:
        volume_levels_df = pickle.load(file)
    days_to_plot = volume_levels_df.iloc[-number_of_days_for_volume_levels_plot:]

    ax = plt.subplot(1, 1, 1)
    ax2 = plt.subplot(1, 1, 1)
    color_number = 0
    for i in lookback_periods_in_days:
        for j in days_to_plot.index:
            ax.scatter(j, days_to_plot[f"PriceLevels_{i}"][j]["poc"],
                       color=colors_of_n_day_volume_level_points[color_number])
            ax.scatter(j, days_to_plot[f"PriceLevels_{i}"][j]["vah"],
                       color=colors_of_n_day_volume_level_points[color_number])
            ax.scatter(j, days_to_plot[f"PriceLevels_{i}"][j]["val"],
                       color=colors_of_n_day_volume_level_points[color_number])
        color_number += 1
        # ax.label(label=f"{i} days lookback")
    # ax.legend()
    ax.plot()
    ax2.plot(df_daily["Datetime"], df_daily["Close"])
    plt.show()
    return


def matching_logic_for_confluence_areas_for_each_day(all_points):
    all_points.sort()
    list_of_confluence_lists = []
    i = 0
    j = 1
    while i < len(all_points) and j < len(all_points):
        confluence_list = [all_points[i]]
        while j < len(all_points) and all_points[j] <= all_points[i] * (1 + confluence_area_threshold):
            confluence_list.append(all_points[j])
            j = j + 1
        list_of_confluence_lists.append(confluence_list)
        i = j
        j = j + 1

    confluence_points = {sum(i) / len(i): len(i) for i in list_of_confluence_lists if len(i) > 1}
    return confluence_points


def matching_logic_for_confluence_areas_for_each_day_return_with_lookback(all_points):
    all_points = sorted(all_points, key=lambda x: x[1])
    list_of_confluence_lists = []
    i = 0
    j = 1
    while i < len(all_points) and j < len(all_points):
        confluence_list = [all_points[i]]
        while j < len(all_points) and all_points[j][1] <= all_points[i][1] * (1 + confluence_area_threshold):
            confluence_list.append(all_points[j])
            j = j + 1
        list_of_confluence_lists.append(confluence_list)
        i = j
        j = j + 1

    confluence_points = {}
    for i in list_of_confluence_lists:
        if len(i) > 1:
            sum = 0
            lookbacks_of_confluence = []
            for j in i:
                sum += j[1]
                lookbacks_of_confluence.append(int(j[0]))
            confluence_points[float(sum) / float(len(i))] = lookbacks_of_confluence
    return confluence_points


def matching_logic_for_confluence_areas_for_each_day_return_with_constituent_and_their_lookback(all_points):
    all_points = sorted(all_points, key=lambda x: x[1])
    list_of_confluence_lists = []
    i = 0
    j = 1
    while i < len(all_points) and j < len(all_points):
        confluence_list = [all_points[i]]
        while j < len(all_points) and all_points[j][1] <= all_points[i][1] * (1 + confluence_area_threshold):
            confluence_list.append(all_points[j])
            j = j + 1
        list_of_confluence_lists.append(confluence_list)
        i = j
        j = j + 1

    confluence_points = {}
    for i in list_of_confluence_lists:
        if len(i) > 1:
            sum = 0
            constituent_and_its_lookback = []
            for j in i:
                sum += j[1]
                constituent_and_its_lookback.append([int(j[0]), j[1], j[2]])
            confluence_points[float(sum) / float(len(i))] = constituent_and_its_lookback
    return confluence_points


def matching_logic_for_confluence_areas_for_each_day_return_with_constituent_and_their_lookback_with_weighted_confluence_strength(
        all_points):
    all_points = sorted(all_points, key=lambda x: x[1])
    list_of_confluence_lists = []
    i = 0
    j = 1
    while i < len(all_points) and j < len(all_points):
        confluence_list = [all_points[i]]
        while j < len(all_points) and all_points[j][1] <= all_points[i][1] * (1 + confluence_area_threshold):
            confluence_list.append(all_points[j])
            j = j + 1
        list_of_confluence_lists.append(confluence_list)
        i = j
        j = j + 1

    confluence_points = {}
    for i in list_of_confluence_lists:
        if len(i) > 1:
            sum = 0
            constituent_and_its_lookback = []
            confluence_strength = 0
            for j in i:
                sum += j[1]
                confluence_strength += lookback_weights_for_confluence_area_calculation[int(j[0])]
                constituent_and_its_lookback.append([int(j[0]), j[1], j[2]])
            confluence_points[float(sum) / float(len(i))] = {"constituents_and_lookback": constituent_and_its_lookback,
                                                             "confluence_strength": confluence_strength}

    return confluence_points


def matching_logic_for_confluence_areas_for_each_day_based_on_weights_for_each_lookback(all_points, weights):
    all_points = sorted(all_points, key=lambda x: x[1])
    list_of_confluence_lists = []
    i = 0
    j = 1
    while i < len(all_points) and j < len(all_points):
        confluence_list = [all_points[i]]
        while j < len(all_points) and all_points[j][1] <= all_points[i][1] * (1 + confluence_area_threshold):
            confluence_list.append(all_points[j])
            j = j + 1
        list_of_confluence_lists.append(confluence_list)
        i = j
        j = j + 1

    confluence_points = {}
    for i in list_of_confluence_lists:
        if len(i) > 1:
            weighted_sum = 0
            sum_of_weights = 0
            for j in i:
                weighted_sum += weights[lookback_periods_in_days.index(j[0])] * j[1]
                sum_of_weights += weights[lookback_periods_in_days.index(j[0])]
            confluence_points[float(weighted_sum) / float(sum_of_weights)] = sum_of_weights

    return confluence_points


def prior(params):
    return 1


# def alpha(*params):
#     # params = params[0]
#     with open(f'.NSEI_forward_returns.pkl', 'rb') as file:
#         forward_returns = pickle.load(file)
#     forward_returns = forward_returns[(forward_returns.index >= "2012-01-01") & (forward_returns.index <= "2022-02-01")]
#     with open(f'.NSEI_low_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'rb') as file:
#         low_pivots = pickle.load(file)
#     X = []
#     dates = []
#     for i in forward_returns.index:
#         if i in low_pivots.keys():
#             lookbacks_of_pivot = low_pivots[i][1]
#             confluence_strength = 0
#             for j in lookbacks_of_pivot:
#                 confluence_strength += params[lookback_periods_in_days.index(j)]
#             # if confluence_strength >= confluence_strength_threshold_mcmc:
#             X.append(confluence_strength)
#             # else:
#             #     X.append(0)
#             dates.append(i)
#         else:
#             X.append(0)
#
#     return np.asarray(X)

def alpha_low_pivots(*params):
    # params = params[0]
    with open(f'.NSEI_low_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'rb') as file:
        low_pivots = pickle.load(file)
    X = []
    dates = []
    for i in low_pivots.keys():
        if i >= datetime.datetime.strptime('2012-01-01 00:00:00',
                                           "%Y-%m-%d %H:%M:%S") and i <= datetime.datetime.strptime(
                '2022-02-01 00:00:00', "%Y-%m-%d %H:%M:%S"):
            lookbacks_of_pivot = low_pivots[i][1]
            confluence_strength = 0
            for j in lookbacks_of_pivot:
                confluence_strength += params[lookback_periods_in_days.index(j)]
            # if confluence_strength >= confluence_strength_threshold_mcmc:
            X.append(confluence_strength)
            # else:
            #     X.append(0)
            dates.append(i)

    return np.asarray(X)


def alpha_high_pivots(*params):
    with open(f'.NSEI_high_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'rb') as file:
        high_pivots = pickle.load(file)
    X = []
    dates = []
    for i in high_pivots.keys():
        if i >= datetime.datetime.strptime('2012-01-01 00:00:00',
                                           "%Y-%m-%d %H:%M:%S") and i <= datetime.datetime.strptime(
                '2022-02-01 00:00:00', "%Y-%m-%d %H:%M:%S"):
            lookbacks_of_pivot = high_pivots[i][1]
            confluence_strength = 0
            for j in lookbacks_of_pivot:
                confluence_strength += params[lookback_periods_in_days.index(j)]
            # if confluence_strength >= confluence_strength_threshold_mcmc:
            X.append(confluence_strength)
            # else:
            #     X.append(0)
            dates.append(i)

    return np.asarray(X)


# def alpha(*params):
#     # params = params[0]
#     with open(f'.NSEI_low_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'rb') as file:
#         low_pivots = pickle.load(file)
#     X = []
#     dates = []
#     for i in low_pivots.keys():
#         if i >= datetime.datetime.strptime('2012-01-01 00:00:00', "%Y-%m-%d %H:%M:%S") and i<= datetime.datetime.strptime('2022-02-01 00:00:00', "%Y-%m-%d %H:%M:%S"):
#             lookbacks_of_pivot = low_pivots[i][1]
#             confluence_strength = 0
#             for j in lookbacks_of_pivot:
#                 confluence_strength += params[lookback_periods_in_days.index(j)]
#             if confluence_strength >= confluence_strength_threshold_mcmc:
#                 X.append(confluence_strength)
#             else:
#                 X.append(0)
#             dates.append(i)
#
#     return np.asarray(X)

def alpha_low_and_high_pivots(*params):
    with open(f'.NSEI_high_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'rb') as file:
        high_pivots = pickle.load(file)
    with open(f'.NSEI_low_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'rb') as file:
        low_pivots = pickle.load(file)
    X = []
    dates = []
    for i in high_pivots.keys():
        if i >= datetime.datetime.strptime('2012-01-01 00:00:00',
                                           "%Y-%m-%d %H:%M:%S") and i <= datetime.datetime.strptime(
                '2022-02-01 00:00:00', "%Y-%m-%d %H:%M:%S"):
            lookbacks_of_pivot = high_pivots[i][1]
            confluence_strength = 0
            for j in lookbacks_of_pivot:
                confluence_strength += params[lookback_periods_in_days.index(j)]
            # if confluence_strength >= confluence_strength_threshold_mcmc:
            X.append(confluence_strength)
            # else:
            #     X.append(0)
            dates.append(i)

    for i in low_pivots.keys():
        if i >= datetime.datetime.strptime('2012-01-01 00:00:00',
                                           "%Y-%m-%d %H:%M:%S") and i <= datetime.datetime.strptime(
                '2022-02-01 00:00:00', "%Y-%m-%d %H:%M:%S"):
            lookbacks_of_pivot = low_pivots[i][1]
            confluence_strength = 0
            for j in lookbacks_of_pivot:
                confluence_strength += params[lookback_periods_in_days.index(j)]
            # if confluence_strength >= confluence_strength_threshold_mcmc:
            X.append(confluence_strength)
            # else:
            #     X.append(0)
            dates.append(i)

    return np.asarray(X)


@jit(nopython=True)
def alpha_all_points_with_exponential_function_output_numba_helper(data, params, lookbacks):
    confluence_strength_list = []
    for i in data:
        confluence_strength = 0
        for j in i:
            if math.isnan(j[0]) == False:
                confluence_strength += params[lookbacks.index(int(j[0]))] * j[2]
        confluence_strength_list.append(confluence_strength)

    return confluence_strength_list


def alpha_all_points_with_exponential_function_output(*params):
    # start = datetime.datetime.now()
    with open(
            f'Closest_Confluence_Point_.NSEI_along_with_its_constituents_and_their_lookbacks_and_exponential_function_value_in_form_of_numpy_array.pkl',
            'rb') as file:
        all_points_numpy_array = pickle.load(file)

    # X = []
    # dates = []
    # for i in all_points.index:
    # constituents_of_closest_confluence = all_points.loc[i]["Constituents_of_Closest_Confluence"]
    # confluence_strength = 0
    # for j in constituents_of_closest_confluence:
    #     confluence_strength += params[lookback_periods_in_days.index(j[0])]*j[2]
    # print("calling numba")
    # print(constituents_of_closest_confluence,params,lookback_periods_in_days)
    # confluence_strength = alpha_all_points_with_exponential_function_output_numba_helper(np.asarray(constituents_of_closest_confluence),list(params),lookback_periods_in_days)
    # X.append(confluence_strength)
    # dates.append(i)
    X = alpha_all_points_with_exponential_function_output_numba_helper(all_points_numpy_array, list(params),
                                                                       lookback_periods_in_days)

    # end = datetime.datetime.now()
    # print((end-start).microseconds)
    return np.asarray(X)


def alpha_all_points_with_exponential_function_output_with_30_day_end_to_end_percentile_gt_85(*params):
    with open(
            f'Closest_Confluence_Point_.NSEI_its_constituents_lookbacks_and_exponential_function_value_in_form_of_numpy_array_for_30_D_end_to_end_returns_percentile_above_85.pkl',
            'rb') as file:
        all_points_numpy_array = pickle.load(file)

    X = alpha_all_points_with_exponential_function_output_numba_helper(all_points_numpy_array, list(params),
                                                                       lookback_periods_in_days)

    return np.asarray(X)


def alpha_all_points_with_exponential_function_output_with_30_day_end_to_end_all_days_percentile_gt_85(*params):
    with open(
            f'Closest_Confluence_Point_.NSEI_constituents_lb_and_exponential_function_value_in_form_of_numpy_array_for_30_D_end_to_end_returns_all_days_percentile_above_85.pkl',
            'rb') as file:
        all_points_numpy_array = pickle.load(file)

    X = alpha_all_points_with_exponential_function_output_numba_helper(all_points_numpy_array, list(params),
                                                                       lookback_periods_in_days)

    return np.asarray(X)


def mcmc_multiprocessing_helper(args):
    guess = args[0]
    target_values = args[1]
    guess_length = len(guess)
    res = pd.DataFrame(columns=[str(i) for i in lookback_periods_in_days] + ["NMIS"])
    opt = Optimiser(method="MCMC")
    opt.define_alpha_function(alpha_low_pivots)
    opt.define_prior(prior)
    opt.define_guess(guess=guess)
    opt.define_iterations(100)
    opt.define_optim_function(None)
    opt.define_target(np.asarray(target_values))
    opt.define_lower_and_upper_limit(0, 1)
    mc, rs = opt.optimise()
    res_iter = []
    for i in range(100):
        d = {}
        for j in range(guess_length):
            key = str(lookback_periods_in_days[j])
            val = mc.analyse_results(rs, top_n=100)[0][i][j]
            d[key] = val
            d.update({'NMIS': mc.analyse_results(rs, top_n=100)[1][i]})
        res_iter.append(d)
    res_iter = pd.DataFrame(res_iter)
    res = pd.concat([res, res_iter], axis=0)
    return res


def find_optimal_weights_for_lookbacks_for_confluence_strength_using_mcmc_with_multiprocessing(ticker):
    with open(f'{ticker}_forward_returns.pkl', 'rb') as file:
        forward_returns = pickle.load(file)

    with open(f'{ticker}_low_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'rb') as file:
        low_pivots = pickle.load(file)

    forward_returns = forward_returns[(forward_returns.index >= "2012-01-01") & (forward_returns.index <= "2022-02-01")]
    target = {}
    target_values = []

    # for i in forward_returns.index:
    #     try:
    #         target[i] = forward_returns["end_to_end_returns"][i]["30_day"]
    #         target_values.append(forward_returns["end_to_end_returns"][i]["30_day"])
    #     except Exception as e:
    #         continue

    for i in forward_returns.index:
        if i in low_pivots.keys():
            try:
                target[i] = forward_returns["end_to_end_returns"][i]["30_day"]
                target_values.append(forward_returns["end_to_end_returns"][i]["30_day"])
            except Exception as e:
                continue

    with open(f'{ticker}_30_day_forward_returns_for_mcmc_alpha.pkl', 'wb') as file:
        pickle.dump(target, file)
    guess_length = len(lookback_periods_in_days)
    guess_list = [np.random.dirichlet(np.ones(guess_length), size=1).tolist()[0] for i in range(10)]
    inputs = []
    for i in guess_list:
        inputs.append([i, target_values])

    pool = ProcessingPool(nodes=6)
    results = pool.map(mcmc_multiprocessing_helper, inputs)
    pool.clear()

    res = pd.DataFrame(columns=[str(i) for i in lookback_periods_in_days] + ["NMIS"])

    res = res.sort_values("NMIS", axis=0, ascending=False).reset_index(drop=True)

    chosen_weights = res  # .iloc[0]
    with open(f'{ticker}_mcmc_weights_for_low_pivots_for_30_day_forward_returns_vs_confluence_strength_nmi.pkl',
              'wb') as file:
        pickle.dump(chosen_weights, file)

    return


def find_optimal_weights_for_lookbacks_for_confluence_strength_using_mcmc(ticker):
    with open(f'{ticker}_forward_returns.pkl', 'rb') as file:
        forward_returns = pickle.load(file)

    with open(f'{ticker}_low_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'rb') as file:
        low_pivots = pickle.load(file)

    with open(f'{ticker}_high_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'rb') as file:
        high_pivots = pickle.load(file)

    forward_returns = forward_returns[(forward_returns.index >= "2012-01-01") & (forward_returns.index <= "2022-02-01")]
    target = {}
    target_values = []

    # for i in forward_returns.index:
    #     try:
    #         target[i] = forward_returns["end_to_end_returns"][i]["30_day"]
    #         target_values.append(forward_returns["end_to_end_returns"][i]["30_day"])
    #     except Exception as e:
    #         continue

    for i in forward_returns.index:
        if i in low_pivots.keys():
            try:
                target[i] = forward_returns["end_to_end_returns"][i]["30_day"]
                target_values.append(forward_returns["end_to_end_returns"][i]["30_day"])
            except Exception as e:
                continue

    with open(f'{ticker}_30_day_forward_returns_for_mcmc_alpha.pkl', 'wb') as file:
        pickle.dump(target, file)
    guess_length = len(lookback_periods_in_days)
    guess_list = [np.random.dirichlet(np.ones(guess_length), size=1).tolist()[0] for i in
                  range(mcmc_number_of_starting_points)]
    res = pd.DataFrame(columns=[str(i) for i in lookback_periods_in_days] + ["NMIS"])
    for guess in tqdm(guess_list):
        opt = Optimiser(method="MCMC")
        opt.define_alpha_function(alpha_low_pivots)
        opt.define_prior(prior)
        opt.define_guess(guess=guess)
        opt.define_iterations(mcmc_iterations)
        opt.define_optim_function(None)
        opt.define_target(np.asarray(target_values))
        opt.define_lower_and_upper_limit(0, 1)
        mc, rs = opt.optimise()
        res_iter = []
        for i in range(mcmc_iterations):
            d = {}
            for j in range(guess_length):
                key = str(lookback_periods_in_days[j])
                val = mc.analyse_results(rs, top_n=mcmc_iterations)[0][i][j]
                d[key] = val
                d.update({'NMIS': mc.analyse_results(rs, top_n=mcmc_iterations)[1][i]})
            res_iter.append(d)
            with open(f'mcmc_cache\\mcmc_starting_point_{guess_list.index(guess)}_iter{i}_alpha.pkl', 'wb') as file:
                pickle.dump(res_iter, file)
        res_iter = pd.DataFrame(res_iter)
        res = pd.concat([res, res_iter], axis=0)

    res = res.sort_values("NMIS", axis=0, ascending=False).reset_index(drop=True)

    chosen_weights = res  # .iloc[0]
    with open(f'{ticker}_mcmc_weights_for_low_pivots_for_30_day_forward_returns_vs_confluence_strength_nmi.pkl',
              'wb') as file:
        pickle.dump(chosen_weights, file)

    return


def find_optimal_weights_for_lookbacks_for_confluence_strength_using_mcmc_for_exponential_function_output(ticker,
                                                                                                          number_of_days_for_forward_returns_calculation,
                                                                                                          forward_returns_type_for_nmi):
    with open(f'{ticker}_forward_returns.pkl', 'rb') as file:
        forward_returns = pickle.load(file)

    with open(
            f'Closest_Confluence_Point_{ticker}_along_with_its_constituents_and_their_lookbacks_and_exponential_function_value.pkl',
            'rb') as file:
        all_points = pickle.load(file)

    forward_returns = forward_returns[(forward_returns.index >= "2012-01-01") & (forward_returns.index <= "2022-02-01")]
    target = {}
    target_values = []

    for i in forward_returns.index:
        if i in all_points.index:
            try:
                target[i] = forward_returns[f"{forward_returns_type_for_nmi}"][i][
                    f"{number_of_days_for_forward_returns_calculation}_day"]
                target_values.append(forward_returns[f"{forward_returns_type_for_nmi}"][i][
                                         f"{number_of_days_for_forward_returns_calculation}_day"])
            except Exception as e:
                continue

    guess_length = len(lookback_periods_in_days)
    guess_list = [np.random.dirichlet(np.ones(guess_length), size=1).tolist()[0] for i in
                  range(mcmc_number_of_starting_points)]
    res = pd.DataFrame(columns=[str(i) for i in lookback_periods_in_days] + ["NMIS"])
    for guess in tqdm(guess_list):
        opt = Optimiser(method="MCMC")
        opt.define_alpha_function(alpha_all_points_with_exponential_function_output)
        opt.define_prior(prior)
        opt.define_guess(guess=guess)
        opt.define_iterations(mcmc_iterations)
        opt.define_optim_function(None)
        opt.define_target(np.asarray(target_values))
        opt.define_lower_and_upper_limit(0, 1)
        mc, rs = opt.optimise()
        res_iter = []
        for i in range(mcmc_iterations):
            d = {}
            for j in range(guess_length):
                key = str(lookback_periods_in_days[j])
                val = mc.analyse_results(rs, top_n=mcmc_iterations)[0][i][j]
                d[key] = val
                d.update({'NMIS': mc.analyse_results(rs, top_n=mcmc_iterations)[1][i]})
            res_iter.append(d)
            with open(
                    f'mcmc_cache_with_exponential_function\\{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type_for_nmi}\\mcmc_starting_point_{guess_list.index(guess)}_iter{i}.pkl',
                    'wb') as file:
                pickle.dump(res_iter, file)
        res_iter = pd.DataFrame(res_iter)
        res = pd.concat([res, res_iter], axis=0)

    res = res.sort_values("NMIS", axis=0, ascending=False).reset_index(drop=True)

    chosen_weights = res  # .iloc[0]
    with open(
            f'mcmc_cache_with_exponential_function\\{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type_for_nmi}\\{ticker}_mcmc_weights_master.pkl',
            'wb') as file:
        pickle.dump(chosen_weights, file)

    return


def find_optimal_weights_for_lookbacks_for_confluence_strength_using_mcmc_for_exponential_function_output_percentile_gt_85(
        ticker, number_of_days_for_forward_returns_calculation, forward_returns_type_for_nmi):
    with open(
            f'target_values_{ticker}_{number_of_days_for_forward_returns_calculation}_D_{forward_returns_type_for_nmi}_percentile_above_85.pkl',
            'rb') as file:
        target_values = pickle.load(file)

    guess_length = len(lookback_periods_in_days)
    guess_list = [np.random.dirichlet(np.ones(guess_length), size=1).tolist()[0] for i in
                  range(mcmc_number_of_starting_points)]
    res = pd.DataFrame(columns=[str(i) for i in lookback_periods_in_days] + ["NMIS"])
    for guess in tqdm(guess_list):
        opt = Optimiser(method="MCMC")
        opt.define_alpha_function(
            alpha_all_points_with_exponential_function_output_with_30_day_end_to_end_percentile_gt_85)
        opt.define_prior(prior)
        opt.define_guess(guess=guess)
        opt.define_iterations(mcmc_iterations)
        opt.define_optim_function(None)
        opt.define_target(np.asarray(target_values))
        opt.define_lower_and_upper_limit(0, 1)
        mc, rs = opt.optimise()
        res_iter = []
        for i in range(mcmc_iterations):
            d = {}
            for j in range(guess_length):
                key = str(lookback_periods_in_days[j])
                val = mc.analyse_results(rs, top_n=mcmc_iterations)[0][i][j]
                d[key] = val
                d.update({'NMIS': mc.analyse_results(rs, top_n=mcmc_iterations)[1][i]})
            res_iter.append(d)
            with open(
                    f'mcmc_cache_percentile_gt_85\\{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type_for_nmi}\\mcmc_starting_point_{guess_list.index(guess)}_iter{i}.pkl',
                    'wb') as file:
                pickle.dump(res_iter, file)
        res_iter = pd.DataFrame(res_iter)
        res = pd.concat([res, res_iter], axis=0)

    res = res.sort_values("NMIS", axis=0, ascending=False).reset_index(drop=True)

    chosen_weights = res  # .iloc[0]
    with open(
            f'mcmc_cache_percentile_gt_85\\{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type_for_nmi}\\{ticker}_mcmc_weights_master.pkl',
            'wb') as file:
        pickle.dump(chosen_weights, file)

    return


def find_optimal_weights_for_lookbacks_for_confluence_strength_using_mcmc_for_exponential_function_output_all_days_percentile_gt_85(
        ticker, number_of_days_for_forward_returns_calculation, forward_returns_type_for_nmi):
    with open(
            f'target_values_{ticker}_{number_of_days_for_forward_returns_calculation}_D_{forward_returns_type_for_nmi}_all_days_percentile_above_85.pkl',
            'rb') as file:
        target_values = pickle.load(file)

    guess_length = len(lookback_periods_in_days)
    guess_list = [np.random.dirichlet(np.ones(guess_length), size=1).tolist()[0] for i in
                  range(mcmc_number_of_starting_points)]
    res = pd.DataFrame(columns=[str(i) for i in lookback_periods_in_days] + ["NMIS"])
    for guess in tqdm(guess_list):
        opt = Optimiser(method="MCMC")
        opt.define_alpha_function(
            alpha_all_points_with_exponential_function_output_with_30_day_end_to_end_all_days_percentile_gt_85)
        opt.define_prior(prior)
        opt.define_guess(guess=guess)
        opt.define_iterations(mcmc_iterations)
        opt.define_optim_function(None)
        opt.define_target(np.asarray(target_values))
        opt.define_lower_and_upper_limit(0, 1)
        mc, rs = opt.optimise()
        res_iter = []
        for i in range(mcmc_iterations):
            d = {}
            for j in range(guess_length):
                key = str(lookback_periods_in_days[j])
                val = mc.analyse_results(rs, top_n=mcmc_iterations)[0][i][j]
                d[key] = val
                d.update({'NMIS': mc.analyse_results(rs, top_n=mcmc_iterations)[1][i]})
            res_iter.append(d)
            with open(
                    f'mcmc_cache_all_days_percentile_gt_85\\{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type_for_nmi}\\mcmc_starting_point_{guess_list.index(guess)}_iter{i}.pkl',
                    'wb') as file:
                pickle.dump(res_iter, file)
        res_iter = pd.DataFrame(res_iter)
        res = pd.concat([res, res_iter], axis=0)

    res = res.sort_values("NMIS", axis=0, ascending=False).reset_index(drop=True)

    chosen_weights = res  # .iloc[0]
    with open(
            f'mcmc_cache_all_days_percentile_gt_85\\{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type_for_nmi}\\{ticker}_mcmc_weights_master.pkl',
            'wb') as file:
        pickle.dump(chosen_weights, file)

    return


def find_optimal_weights_for_lookbacks_for_confluence_strength_for_low_pivots_using_mcmc_wrapper_for_threadripper(
        ticker, number_of_days_for_forward_returns_calculation, forward_returns_type_for_nmi):
    with open(f'{ticker}_forward_returns.pkl', 'rb') as file:
        forward_returns = pickle.load(file)

    with open(f'{ticker}_low_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'rb') as file:
        low_pivots = pickle.load(file)

    with open(f'{ticker}_high_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'rb') as file:
        high_pivots = pickle.load(file)

    forward_returns = forward_returns[(forward_returns.index >= "2012-01-01") & (forward_returns.index <= "2022-02-01")]
    target = {}
    target_values = []

    for i in forward_returns.index:
        if i in low_pivots.keys():
            try:
                target[i] = forward_returns[f"{forward_returns_type_for_nmi}"][i][
                    f"{number_of_days_for_forward_returns_calculation}_day"]
                target_values.append(forward_returns[f"{forward_returns_type_for_nmi}"][i][
                                         f"{number_of_days_for_forward_returns_calculation}_day"])
            except Exception as e:
                continue

    guess_length = len(lookback_periods_in_days)
    guess_list = [np.random.dirichlet(np.ones(guess_length), size=1).tolist()[0] for i in
                  range(mcmc_number_of_starting_points)]
    res = pd.DataFrame(columns=[str(i) for i in lookback_periods_in_days] + ["NMIS"])
    for guess in tqdm(guess_list):
        opt = Optimiser(method="MCMC")
        opt.define_alpha_function(alpha_low_pivots)
        opt.define_prior(prior)
        opt.define_guess(guess=guess)
        opt.define_iterations(mcmc_iterations)
        opt.define_optim_function(None)
        opt.define_target(np.asarray(target_values))
        opt.define_lower_and_upper_limit(0, 1)
        mc, rs = opt.optimise()
        res_iter = []
        for i in range(mcmc_iterations):
            d = {}
            for j in range(guess_length):
                key = str(lookback_periods_in_days[j])
                val = mc.analyse_results(rs, top_n=mcmc_iterations)[0][i][j]
                d[key] = val
                d.update({'NMIS': mc.analyse_results(rs, top_n=mcmc_iterations)[1][i]})
            res_iter.append(d)
            with open(
                    f'mcmc_cache\\{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type_for_nmi}\\mcmc_starting_point_{guess_list.index(guess)}_iter{i}.pkl',
                    'wb') as file:
                pickle.dump(res_iter, file)
        res_iter = pd.DataFrame(res_iter)
        res = pd.concat([res, res_iter], axis=0)

    res = res.sort_values("NMIS", axis=0, ascending=False).reset_index(drop=True)

    chosen_weights = res  # .iloc[0]
    with open(
            f'mcmc_cache\\{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type_for_nmi}\\{ticker}_mcmc_weights_master.pkl',
            'wb') as file:
        pickle.dump(chosen_weights, file)

    return


def find_optimal_weights_for_lookbacks_for_confluence_strength_for_high_pivots_using_mcmc_wrapper_for_threadripper(
        ticker, number_of_days_for_forward_returns_calculation, forward_returns_type_for_nmi):
    with open(f'{ticker}_forward_returns.pkl', 'rb') as file:
        forward_returns = pickle.load(file)

    with open(f'{ticker}_low_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'rb') as file:
        low_pivots = pickle.load(file)

    with open(f'{ticker}_high_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'rb') as file:
        high_pivots = pickle.load(file)

    forward_returns = forward_returns[(forward_returns.index >= "2012-01-01") & (forward_returns.index <= "2022-02-01")]
    target = {}
    target_values = []

    for i in forward_returns.index:
        if i in high_pivots.keys():
            try:
                target[i] = forward_returns[f"{forward_returns_type_for_nmi}"][i][
                    f"{number_of_days_for_forward_returns_calculation}_day"]
                target_values.append(forward_returns[f"{forward_returns_type_for_nmi}"][i][
                                         f"{number_of_days_for_forward_returns_calculation}_day"])
            except Exception as e:
                continue

    guess_length = len(lookback_periods_in_days)
    guess_list = [np.random.dirichlet(np.ones(guess_length), size=1).tolist()[0] for i in
                  range(mcmc_number_of_starting_points)]
    res = pd.DataFrame(columns=[str(i) for i in lookback_periods_in_days] + ["NMIS"])
    for guess in tqdm(guess_list):
        opt = Optimiser(method="MCMC")
        opt.define_alpha_function(alpha_high_pivots)
        opt.define_prior(prior)
        opt.define_guess(guess=guess)
        opt.define_iterations(mcmc_iterations)
        opt.define_optim_function(None)
        opt.define_target(np.asarray(target_values))
        opt.define_lower_and_upper_limit(0, 1)
        mc, rs = opt.optimise()
        res_iter = []
        for i in range(mcmc_iterations):
            d = {}
            for j in range(guess_length):
                key = str(lookback_periods_in_days[j])
                val = mc.analyse_results(rs, top_n=mcmc_iterations)[0][i][j]
                d[key] = val
                d.update({'NMIS': mc.analyse_results(rs, top_n=mcmc_iterations)[1][i]})
            res_iter.append(d)
            with open(
                    f'mcmc_cache\\high_pivots\\{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type_for_nmi}\\mcmc_starting_point_{guess_list.index(guess)}_iter{i}.pkl',
                    'wb') as file:
                pickle.dump(res_iter, file)
        res_iter = pd.DataFrame(res_iter)
        res = pd.concat([res, res_iter], axis=0)

    res = res.sort_values("NMIS", axis=0, ascending=False).reset_index(drop=True)

    chosen_weights = res  # .iloc[0]
    with open(
            f'mcmc_cache\\high_pivots\\{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type_for_nmi}\\{ticker}_{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type_for_nmi}_mcmc_weights_master.pkl',
            'wb') as file:
        pickle.dump(chosen_weights, file)

    return


def find_optimal_weights_for_lookbacks_for_confluence_strength_for_low_and_high_pivots_using_mcmc_wrapper_for_threadripper(
        ticker, number_of_days_for_forward_returns_calculation, forward_returns_type_for_nmi):
    with open(f'{ticker}_forward_returns.pkl', 'rb') as file:
        forward_returns = pickle.load(file)

    with open(f'{ticker}_low_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'rb') as file:
        low_pivots = pickle.load(file)

    with open(f'{ticker}_high_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'rb') as file:
        high_pivots = pickle.load(file)

    forward_returns = forward_returns[(forward_returns.index >= "2012-01-01") & (forward_returns.index <= "2022-02-01")]
    target = {}
    target_values = []

    for i in forward_returns.index:
        if i in high_pivots.keys() or i in low_pivots.keys():
            try:
                target[i] = forward_returns[f"{forward_returns_type_for_nmi}"][i][
                    f"{number_of_days_for_forward_returns_calculation}_day"]
                target_values.append(forward_returns[f"{forward_returns_type_for_nmi}"][i][
                                         f"{number_of_days_for_forward_returns_calculation}_day"])
            except Exception as e:
                continue

    guess_length = len(lookback_periods_in_days)
    guess_list = [np.random.dirichlet(np.ones(guess_length), size=1).tolist()[0] for i in
                  range(mcmc_number_of_starting_points)]
    res = pd.DataFrame(columns=[str(i) for i in lookback_periods_in_days] + ["NMIS"])
    for guess in tqdm(guess_list):
        opt = Optimiser(method="MCMC")
        opt.define_alpha_function(alpha_low_and_high_pivots)
        opt.define_prior(prior)
        opt.define_guess(guess=guess)
        opt.define_iterations(mcmc_iterations)
        opt.define_optim_function(None)
        opt.define_target(np.asarray(target_values))
        opt.define_lower_and_upper_limit(0, 1)
        mc, rs = opt.optimise()
        res_iter = []
        for i in range(mcmc_iterations):
            d = {}
            for j in range(guess_length):
                key = str(lookback_periods_in_days[j])
                val = mc.analyse_results(rs, top_n=mcmc_iterations)[0][i][j]
                d[key] = val
                d.update({'NMIS': mc.analyse_results(rs, top_n=mcmc_iterations)[1][i]})
            res_iter.append(d)
            with open(
                    f'mcmc_cache\\low_and_high_pivots\\{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type_for_nmi}\\mcmc_starting_point_{guess_list.index(guess)}_iter{i}.pkl',
                    'wb') as file:
                pickle.dump(res_iter, file)
        res_iter = pd.DataFrame(res_iter)
        res = pd.concat([res, res_iter], axis=0)

    res = res.sort_values("NMIS", axis=0, ascending=False).reset_index(drop=True)

    chosen_weights = res  # .iloc[0]
    with open(
            f'mcmc_cache\\low_and_high_pivots\\{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type_for_nmi}\\{ticker}_{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type_for_nmi}_mcmc_weights_master.pkl',
            'wb') as file:
        pickle.dump(chosen_weights, file)

    return


def plot_confluence_areas(df_daily, ticker, cache_path, plot_path="test.jpg"):
    with open(f'{cache_path}', 'rb') as file:
        volume_levels_df = pickle.load(file)
    volume_levels_df = volume_levels_df.iloc[3160:3243]
    df_daily = df_daily[(df_daily["Datetime"] <= max(volume_levels_df.index).strftime("%Y-%m-%d")) & (
                df_daily["Datetime"] >= min(volume_levels_df.index).strftime("%Y-%m-%d"))]

    price_level_columns = [i for i in volume_levels_df.columns if i.startswith("PriceLevels")]

    confluence_points = {}
    for i in volume_levels_df.index:
        all_points = []
        for j in price_level_columns:
            all_points.append(volume_levels_df[j][i]["poc"])
            # all_points.append(volume_levels_df[j][i]["val"])
            # all_points.append(volume_levels_df[j][i]["vah"])
            # all_points.append(volume_levels_df[j][i]["point_of_least_volume_in_va"])
        confluence_points[i] = matching_logic_for_confluence_areas_for_each_day(all_points)

    # with open(f'Confluence_Points_{ticker}_test.pkl', 'wb') as file:
    #     pickle.dump(confluence_points, file)

    ax = plt.subplot(1, 1, 1)
    ax2 = plt.subplot(1, 1, 1)
    for i in list(confluence_points.keys()):
        confluence_point = []
        confluence_strength = []
        for j in list(confluence_points[i].keys()):
            confluence_point.append(j)
            confluence_strength.append(-confluence_points[i][j])
        ax.scatter([i] * len(confluence_point), confluence_point, c=confluence_strength, cmap="gray")
    ax.set_facecolor('pink')
    ax.plot()
    # ax2.plot(df_daily["Datetime"], df_daily["Close"])
    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]

    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(ax2, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)
    ax2.grid(True)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    ax2.xaxis.set_major_formatter(date_format)
    # fig.autofmt_xdate()

    # fig.tight_layout()

    # plt.show()
    plt.savefig(f"{plot_path}")
    plt.close()
    return


def plot_confluence_areas_with_weighted_confluence_strength(df_daily, ticker, cache_path, plot_path="test.jpg"):
    with open(f'{cache_path}', 'rb') as file:
        confluence_points = pickle.load(file)

    days_to_plot = {}
    for i in confluence_points.keys():
        if i.year == 2020 and i.month <= 3:
            days_to_plot[i] = confluence_points[i]

    df_daily = df_daily[(df_daily["Datetime"] <= max(days_to_plot.keys()).strftime("%Y-%m-%d")) & (
                df_daily["Datetime"] >= min(days_to_plot.keys()).strftime("%Y-%m-%d"))]

    ax = plt.subplot(1, 1, 1)
    ax2 = plt.subplot(1, 1, 1)
    for i in list(days_to_plot.keys()):
        confluence_point = []
        confluence_strength = []
        for j in list(days_to_plot[i].keys()):
            confluence_point.append(j)
            if days_to_plot[i][j]["confluence_strength"] > 3:
                confluence_strength.append(-300)
            elif days_to_plot[i][j]["confluence_strength"] < 1:
                confluence_strength.append(300)
            else:
                confluence_strength.append(-days_to_plot[i][j]["confluence_strength"])
        ax.scatter([i] * len(confluence_point), confluence_point, c=confluence_strength, cmap="gray")
    ax.set_facecolor('pink')
    ax.plot()
    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]

    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(ax2, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)
    ax2.grid(True)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    ax2.xaxis.set_major_formatter(date_format)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='g', markersize=5),
                       Patch(facecolor='orange', edgecolor='r',
                             label='Color Patch')]

    plt.savefig(f"{plot_path}")
    plt.close()
    return


def plot_confluence_areas_and_constance_brown(df_daily, ticker, cache_path, plot_path="test.jpg"):
    with open(f'{cache_path}', 'rb') as file:
        volume_levels_df = pickle.load(file)
    volume_levels_df = volume_levels_df.iloc[-number_of_days_for_confluence_level_plot:]
    df_daily = df_daily[(df_daily["Datetime"] <= max(volume_levels_df.index).strftime("%Y-%m-%d")) & (
                df_daily["Datetime"] >= min(volume_levels_df.index).strftime("%Y-%m-%d"))]
    df_constant_brown = df_daily.copy()
    price_level_columns = [i for i in volume_levels_df.columns if i.startswith("PriceLevels")]

    confluence_points = {}
    for i in volume_levels_df.index:
        all_points = []
        for j in price_level_columns:
            all_points.append(volume_levels_df[j][i]["poc"])
            all_points.append(volume_levels_df[j][i]["val"])
            all_points.append(volume_levels_df[j][i]["vah"])
            all_points.append(volume_levels_df[j][i]["point_of_least_volume_in_va"])
        confluence_points[i] = matching_logic_for_confluence_areas_for_each_day(all_points)

    with open(f'Confluence_Points_{ticker}_test.pkl', 'wb') as file:
        pickle.dump(confluence_points, file)

    gs_kw = dict(width_ratios=[1.4, 1], height_ratios=[1, 2])
    # ax = plt.subplots(2, 1, 1)
    # ax2 = plt.subplot(2, 1, 1)
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
    # ax3 = plt.subplot(2, 1, 2)
    for i in list(confluence_points.keys()):
        confluence_point = []
        confluence_strength = []
        for j in list(confluence_points[i].keys()):
            confluence_point.append(j)
            confluence_strength.append(-confluence_points[i][j])
        a0.scatter([i] * len(confluence_point), confluence_point, c=confluence_strength, cmap="gray")
    a0.set_facecolor('pink')
    a0.plot()
    # ax2.plot(df_daily["Datetime"], df_daily["Close"])
    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]

    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(a0, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)
    a0.grid(True)
    a0.set_xlabel('Date')
    a0.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    a1.plot(df_constant_brown["Datetime"], df_constant_brown["CB"], label="Constance-Brown Index")
    a1.plot(df_constant_brown["Datetime"], df_constant_brown["FMACB"], label="FMA Constance-Brown Index")
    a1.plot(df_constant_brown["Datetime"], df_constant_brown["SMACB"], label="SMA Constance-Brown Index")
    # fig.autofmt_xdate()

    # fig.tight_layout()

    # plt.show()
    plt.legend()
    plt.savefig(f"{plot_path}", dpi=300)
    plt.close()
    return


def cache_confluence_areas_with_matching_of_poc_and_poc_or_poc_and_point_of_least_volume(ticker):
    with open(f'VolumeFeatures_{ticker}_Daily_test.pkl', 'rb') as file:
        volume_levels_df = pickle.load(file)

    price_level_columns = [i for i in volume_levels_df.columns if i.startswith("PriceLevels")]
    confluence_points = {}
    for i in volume_levels_df.index:
        try:
            all_points = []
            for j in price_level_columns:
                all_points.append(volume_levels_df[j][i]["poc"])
                all_points.append(volume_levels_df[j][i]["point_of_least_volume_in_va"])
            confluence_points[i] = matching_logic_for_confluence_areas_for_each_day(all_points)
        except:
            continue

    with open(f'Confluence_Points_{ticker}_with_matching_of_poc_and_poc_or_poc_and_point_of_least_volume.pkl',
              'wb') as file:
        pickle.dump(confluence_points, file)

    return


def RSI(data_df, period):
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
    return rsi_df.fillna(0).iloc[:, 0]


def constance_brown(data_df):
    r = RSI(data_df, 14)
    rsidelta = [0] * len(data_df)

    for i in range(len(data_df)):
        if i < rsi_mom_length:
            rsidelta[i] = np.nan
        else:
            rsidelta[i] = r[i] - r[i - rsi_mom_length]

    rsisigma = RSI(data_df, rsi_ma_length).rolling(window=ma_length).mean()
    rsidelta = [0 if math.isnan(x) else x for x in rsidelta]
    s = [0] * len(data_df)
    for i in range(len(rsidelta)):
        s[i] = rsidelta[i] + rsisigma[i]
    # s = [0 if math.isnan(x) else x for x in s]

    data_df["CB"] = s
    data_df[f"FMACB_{fastLength}"] = pd.DataFrame(s).rolling(window=fastLength).mean()
    data_df[f"SMACB_{slowLength}"] = pd.DataFrame(s).rolling(window=slowLength).mean()
    return data_df


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

    return fish1


def plot_price_cb(data):
    fig = make_subplots(rows=2, cols=1, row_heights=[1, 1], shared_xaxes=True, shared_yaxes=True,
                        specs=[[{"secondary_y": True}], [{"secondary_y": True}]])

    fig.update_layout(xaxis2={'anchor': 'y', 'overlaying': 'x', 'side': 'top'},
                      yaxis_domain=[0, 1])

    fig.add_trace(go.Candlestick(name=".NSEI", x=data['Datetime'],
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close']), row=1, col=1)

    fig.add_trace(go.Scatter(x=data["Datetime"], y=data["CB"], name="Constance-Brown Index", mode='lines',
                             marker=dict(color='darkorchid')), row=2,
                  col=1,
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=data["Datetime"], y=data["FMACB"], name="FMA Constance-Brown Index", mode='lines',
                             marker=dict(color='orange')), row=2,
                  col=1,
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=data["Datetime"], y=data["SMACB"], name="SMA Constance-Brown Index", mode='lines',
                             marker=dict(color='aquamarine')), row=2,
                  col=1,
                  secondary_y=False)

    fig.data[1].update(xaxis='x2')
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(template="plotly_dark", width=1500, height=1000)
    fig.show()
    fig.write_html("constance_brown.html")


def detect_pivots(df_daily, confluence_points):
    high_pivot_bars = []
    low_pivot_bars = []

    for i in confluence_points.keys():
        for j in confluence_points[i].keys():
            if (df_daily[df_daily["Datetime"] == i].iloc[0]["Close"] < df_daily[df_daily["Datetime"] == i].iloc[0][
                "Open"]) and (((j <= df_daily[df_daily["Datetime"] == i].iloc[0]["High"]) and (
                    j >= max(df_daily[df_daily["Datetime"] == i].iloc[0]["Open"],
                             df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]))) or \
                              (j <= df_daily[df_daily["Datetime"] == i].iloc[0]["High"] * (1 + 0.0001) and j >=
                               df_daily[df_daily["Datetime"] == i].iloc[0]["High"] * (1 - 0.0001))):
                high_pivot_bars.append(i)
                break

    for i in confluence_points.keys():
        for j in confluence_points[i].keys():
            if (df_daily[df_daily["Datetime"] == i].iloc[0]["Close"] > df_daily[df_daily["Datetime"] == i].iloc[0][
                "Open"]) and (((j >= df_daily[df_daily["Datetime"] == i].iloc[0]["Low"]) and (
                    j <= min(df_daily[df_daily["Datetime"] == i].iloc[0]["Open"],
                             df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]))) or \
                              (j <= df_daily[df_daily["Datetime"] == i].iloc[0]["Low"] * (1 + 0.001) and j >=
                               df_daily[df_daily["Datetime"] == i].iloc[0]["Low"] * (1 - 0.001))):
                low_pivot_bars.append(i)
                break

    high_pivots = [x for x in high_pivot_bars if x not in low_pivot_bars]
    low_pivots = [x for x in low_pivot_bars if x not in high_pivot_bars]

    return high_pivots, low_pivots


def cache_pivots(df_daily, confluence_points, ticker):
    high_pivot_bars = []
    low_pivot_bars = []

    for i in confluence_points.keys():
        for j in confluence_points[i].keys():
            if (df_daily[df_daily["Datetime"] == i].iloc[0]["Close"] < df_daily[df_daily["Datetime"] == i].iloc[0][
                "Open"]) and (((j <= df_daily[df_daily["Datetime"] == i].iloc[0]["High"]) and (
                    j >= max(df_daily[df_daily["Datetime"] == i].iloc[0]["Open"],
                             df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]))) or \
                              (j <= df_daily[df_daily["Datetime"] == i].iloc[0]["High"] * (1 + 0.001) and j >=
                               df_daily[df_daily["Datetime"] == i].iloc[0]["High"] * (1 - 0.001))):
                high_pivot_bars.append(i)
                break

    for i in confluence_points.keys():
        for j in confluence_points[i].keys():
            if (df_daily[df_daily["Datetime"] == i].iloc[0]["Close"] > df_daily[df_daily["Datetime"] == i].iloc[0][
                "Open"]) and (((j >= df_daily[df_daily["Datetime"] == i].iloc[0]["Low"]) and (
                    j <= min(df_daily[df_daily["Datetime"] == i].iloc[0]["Open"],
                             df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]))) or \
                              (j <= df_daily[df_daily["Datetime"] == i].iloc[0]["Low"] * (1 + 0.001) and j >=
                               df_daily[df_daily["Datetime"] == i].iloc[0]["Low"] * (1 - 0.001))):
                low_pivot_bars.append(i)
                break

    high_pivots = [x for x in high_pivot_bars if x not in low_pivot_bars]
    low_pivots = [x for x in low_pivot_bars if x not in high_pivot_bars]

    with open(f'{ticker}_high_pivots.pkl', 'wb') as file:
        pickle.dump(high_pivots, file)

    with open(f'{ticker}_low_pivots.pkl', 'wb') as file:
        pickle.dump(low_pivots, file)

    return


def cache_fisher_pivots(df_daily, ticker):
    df_daily[f"{fisher_lookback}_day_fisher"] = fisher(df_daily, fisher_lookback)

    fisher_low_pivots = []
    fisher_high_pivots = []
    for i in df_daily.index:
        try:
            if df_daily.loc[i - 1][f"{fisher_lookback}_day_fisher"] > df_daily.loc[i][
                f"{fisher_lookback}_day_fisher"] and df_daily.loc[i + 1][f"{fisher_lookback}_day_fisher"] > \
                    df_daily.loc[i][f"{fisher_lookback}_day_fisher"]:
                fisher_low_pivots.append(df_daily.loc[i]["Datetime"])
                continue
            if df_daily.loc[i - 1][f"{fisher_lookback}_day_fisher"] < df_daily.loc[i][
                f"{fisher_lookback}_day_fisher"] and \
                    df_daily.loc[i + 1][f"{fisher_lookback}_day_fisher"] < df_daily.loc[i][
                f"{fisher_lookback}_day_fisher"]:
                fisher_high_pivots.append(df_daily.loc[i]["Datetime"])
        except Exception as e:
            print(e)

    with open(f'{ticker}_fisher_{fisher_lookback}_lb_low_pivots.pkl', 'wb') as file:
        pickle.dump(fisher_low_pivots, file)

    with open(f'{ticker}_fisher_{fisher_lookback}_lb_high_pivots.pkl', 'wb') as file:
        pickle.dump(fisher_high_pivots, file)

    return


def cache_pivots_along_with_confluence(df_daily, confluence_points, ticker):
    high_pivot_bars = {}
    low_pivot_bars = {}

    for i in confluence_points.keys():
        for j in confluence_points[i].keys():
            if (df_daily[df_daily["Datetime"] == i].iloc[0]["Close"] < df_daily[df_daily["Datetime"] == i].iloc[0][
                "Open"]) and (((j <= df_daily[df_daily["Datetime"] == i].iloc[0]["High"]) and (
                    j >= max(df_daily[df_daily["Datetime"] == i].iloc[0]["Open"],
                             df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]))) or \
                              (j <= df_daily[df_daily["Datetime"] == i].iloc[0]["High"] * (1 + 0.001) and j >=
                               df_daily[df_daily["Datetime"] == i].iloc[0]["High"] * (1 - 0.001))):
                high_pivot_bars[i] = j
                break

    for i in confluence_points.keys():
        for j in confluence_points[i].keys():
            if (df_daily[df_daily["Datetime"] == i].iloc[0]["Close"] > df_daily[df_daily["Datetime"] == i].iloc[0][
                "Open"]) and (((j >= df_daily[df_daily["Datetime"] == i].iloc[0]["Low"]) and (
                    j <= min(df_daily[df_daily["Datetime"] == i].iloc[0]["Open"],
                             df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]))) or \
                              (j <= df_daily[df_daily["Datetime"] == i].iloc[0]["Low"] * (1 + 0.001) and j >=
                               df_daily[df_daily["Datetime"] == i].iloc[0]["Low"] * (1 - 0.001))):
                low_pivot_bars[i] = j
                break

    high_pivots = {x: high_pivot_bars[x] for x in list(high_pivot_bars.keys()) if x not in list(low_pivot_bars.keys())}
    low_pivots = {x: low_pivot_bars[x] for x in list(low_pivot_bars.keys()) if x not in list(high_pivot_bars.keys())}

    with open(f'{ticker}_high_pivots_with_confluence_point_responsible.pkl', 'wb') as file:
        pickle.dump(high_pivots, file)

    with open(f'{ticker}_low_pivots_with_confluence_point_responsible.pkl', 'wb') as file:
        pickle.dump(low_pivots, file)

    return


def cache_pivots_along_with_confluence_point_and_its_strength(df_daily, confluence_points, ticker):
    high_pivot_bars = {}
    low_pivot_bars = {}

    for i in confluence_points.keys():
        for j in confluence_points[i].keys():
            if (df_daily[df_daily["Datetime"] == i].iloc[0]["Close"] < df_daily[df_daily["Datetime"] == i].iloc[0][
                "Open"]) and (((j <= df_daily[df_daily["Datetime"] == i].iloc[0]["High"]) and (
                    j >= max(df_daily[df_daily["Datetime"] == i].iloc[0]["Open"],
                             df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]))) or \
                              (j <= df_daily[df_daily["Datetime"] == i].iloc[0]["High"] * (1 + 0.001) and j >=
                               df_daily[df_daily["Datetime"] == i].iloc[0]["High"] * (1 - 0.001))):
                high_pivot_bars[i] = [j, confluence_points[i][j]]
                break

    for i in confluence_points.keys():
        for j in confluence_points[i].keys():
            if (df_daily[df_daily["Datetime"] == i].iloc[0]["Close"] > df_daily[df_daily["Datetime"] == i].iloc[0][
                "Open"]) and (((j >= df_daily[df_daily["Datetime"] == i].iloc[0]["Low"]) and (
                    j <= min(df_daily[df_daily["Datetime"] == i].iloc[0]["Open"],
                             df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]))) or \
                              (j <= df_daily[df_daily["Datetime"] == i].iloc[0]["Low"] * (1 + 0.001) and j >=
                               df_daily[df_daily["Datetime"] == i].iloc[0]["Low"] * (1 - 0.001))):
                low_pivot_bars[i] = [j, confluence_points[i][j]]
                break

    high_pivots = {x: high_pivot_bars[x] for x in list(high_pivot_bars.keys()) if x not in list(low_pivot_bars.keys())}
    low_pivots = {x: low_pivot_bars[x] for x in list(low_pivot_bars.keys()) if x not in list(high_pivot_bars.keys())}

    with open(f'{ticker}_high_pivots_with_confluence_point_responsible_and_strength.pkl', 'wb') as file:
        pickle.dump(high_pivots, file)

    with open(f'{ticker}_low_pivots_with_confluence_point_responsible_and_strength.pkl', 'wb') as file:
        pickle.dump(low_pivots, file)


def cache_pivots_along_with_confluence_point_and_its_lookback(df_daily, confluence_points, ticker):
    high_pivot_bars = {}
    low_pivot_bars = {}

    for i in confluence_points.keys():
        for j in confluence_points[i].keys():
            if (df_daily[df_daily["Datetime"] == i].iloc[0]["Close"] < df_daily[df_daily["Datetime"] == i].iloc[0][
                "Open"]) and (((j <= df_daily[df_daily["Datetime"] == i].iloc[0]["High"]) and (
                    j >= max(df_daily[df_daily["Datetime"] == i].iloc[0]["Open"],
                             df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]))) or \
                              (j <= df_daily[df_daily["Datetime"] == i].iloc[0]["High"] * (1 + 0.001) and j >=
                               df_daily[df_daily["Datetime"] == i].iloc[0]["High"] * (1 - 0.001))):
                high_pivot_bars[i] = [j, confluence_points[i][j]]
                break

    for i in confluence_points.keys():
        for j in confluence_points[i].keys():
            if (df_daily[df_daily["Datetime"] == i].iloc[0]["Close"] > df_daily[df_daily["Datetime"] == i].iloc[0][
                "Open"]) and (((j >= df_daily[df_daily["Datetime"] == i].iloc[0]["Low"]) and (
                    j <= min(df_daily[df_daily["Datetime"] == i].iloc[0]["Open"],
                             df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]))) or \
                              (j <= df_daily[df_daily["Datetime"] == i].iloc[0]["Low"] * (1 + 0.001) and j >=
                               df_daily[df_daily["Datetime"] == i].iloc[0]["Low"] * (1 - 0.001))):
                low_pivot_bars[i] = [j, confluence_points[i][j]]
                break

    high_pivots = {x: high_pivot_bars[x] for x in list(high_pivot_bars.keys()) if x not in list(low_pivot_bars.keys())}
    low_pivots = {x: low_pivot_bars[x] for x in list(low_pivot_bars.keys()) if x not in list(high_pivot_bars.keys())}

    with open(f'{ticker}_high_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'wb') as file:
        pickle.dump(high_pivots, file)

    with open(f'{ticker}_low_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'wb') as file:
        pickle.dump(low_pivots, file)


def cache_pivots_along_with_confluence_point_of_poc_and_point_of_lowest_volume_and_its_lookback(df_daily,
                                                                                                confluence_points,
                                                                                                ticker):
    high_pivot_bars = {}
    low_pivot_bars = {}

    for i in confluence_points.keys():
        for j in confluence_points[i].keys():
            if (df_daily[df_daily["Datetime"] == i].iloc[0]["Close"] < df_daily[df_daily["Datetime"] == i].iloc[0][
                "Open"]) and (((j <= df_daily[df_daily["Datetime"] == i].iloc[0]["High"]) and (
                    j >= max(df_daily[df_daily["Datetime"] == i].iloc[0]["Open"],
                             df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]))) or \
                              (j <= df_daily[df_daily["Datetime"] == i].iloc[0]["High"] * (1 + 0.001) and j >=
                               df_daily[df_daily["Datetime"] == i].iloc[0]["High"] * (1 - 0.001))):
                high_pivot_bars[i] = [j, confluence_points[i][j]]
                break

    for i in confluence_points.keys():
        for j in confluence_points[i].keys():
            if (df_daily[df_daily["Datetime"] == i].iloc[0]["Close"] > df_daily[df_daily["Datetime"] == i].iloc[0][
                "Open"]) and (((j >= df_daily[df_daily["Datetime"] == i].iloc[0]["Low"]) and (
                    j <= min(df_daily[df_daily["Datetime"] == i].iloc[0]["Open"],
                             df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]))) or \
                              (j <= df_daily[df_daily["Datetime"] == i].iloc[0]["Low"] * (1 + 0.001) and j >=
                               df_daily[df_daily["Datetime"] == i].iloc[0]["Low"] * (1 - 0.001))):
                low_pivot_bars[i] = [j, confluence_points[i][j]]
                break

    high_pivots = {x: high_pivot_bars[x] for x in list(high_pivot_bars.keys()) if x not in list(low_pivot_bars.keys())}
    low_pivots = {x: low_pivot_bars[x] for x in list(low_pivot_bars.keys()) if x not in list(high_pivot_bars.keys())}

    with open(
            f'{ticker}_high_pivots_with_confluence_point_of_poc_and_point_of_lowest_volume_responsible_and_its_lookbacks.pkl',
            'wb') as file:
        pickle.dump(high_pivots, file)

    with open(
            f'{ticker}_low_pivots_with_confluence_point_of_poc_and_point_of_lowest_volume_responsible_and_its_lookbacks.pkl',
            'wb') as file:
        pickle.dump(low_pivots, file)


def cache_pivots_along_with_confluence_point_of_poc_point_of_low_vol_and_its_strength(df_daily, confluence_points,
                                                                                      ticker):
    high_pivot_bars = {}
    low_pivot_bars = {}

    for i in confluence_points.keys():
        for j in confluence_points[i].keys():
            if (df_daily[df_daily["Datetime"] == i].iloc[0]["Close"] < df_daily[df_daily["Datetime"] == i].iloc[0][
                "Open"]) and (((j <= df_daily[df_daily["Datetime"] == i].iloc[0]["High"]) and (
                    j >= max(df_daily[df_daily["Datetime"] == i].iloc[0]["Open"],
                             df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]))) or \
                              (j <= df_daily[df_daily["Datetime"] == i].iloc[0]["High"] * (1 + 0.001) and j >=
                               df_daily[df_daily["Datetime"] == i].iloc[0]["High"] * (1 - 0.001))):
                high_pivot_bars[i] = [j, confluence_points[i][j]]
                break

    for i in confluence_points.keys():
        for j in confluence_points[i].keys():
            if (df_daily[df_daily["Datetime"] == i].iloc[0]["Close"] > df_daily[df_daily["Datetime"] == i].iloc[0][
                "Open"]) and (((j >= df_daily[df_daily["Datetime"] == i].iloc[0]["Low"]) and (
                    j <= min(df_daily[df_daily["Datetime"] == i].iloc[0]["Open"],
                             df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]))) or \
                              (j <= df_daily[df_daily["Datetime"] == i].iloc[0]["Low"] * (1 + 0.001) and j >=
                               df_daily[df_daily["Datetime"] == i].iloc[0]["Low"] * (1 - 0.001))):
                low_pivot_bars[i] = [j, confluence_points[i][j]]
                break

    high_pivots = {x: high_pivot_bars[x] for x in list(high_pivot_bars.keys()) if x not in list(low_pivot_bars.keys())}
    low_pivots = {x: low_pivot_bars[x] for x in list(low_pivot_bars.keys()) if x not in list(high_pivot_bars.keys())}

    with open(f'{ticker}_high_pivots_with_confluence_point_of_poc_point_of_low_vol_and_strength.pkl', 'wb') as file:
        pickle.dump(high_pivots, file)

    with open(f'{ticker}_low_pivots_with_confluence_point_of_poc_point_of_low_vol_and_strength.pkl', 'wb') as file:
        pickle.dump(low_pivots, file)

    return


def cache_pivots_along_with_confluence_and_lookback_of_confluence(df_daily, confluence_points, ticker):
    high_pivot_bars = {}
    low_pivot_bars = {}

    for i in confluence_points.keys():
        for j in confluence_points[i].keys():
            if (df_daily[df_daily["Datetime"] == i].iloc[0]["Close"] < df_daily[df_daily["Datetime"] == i].iloc[0][
                "Open"]) and (((j <= df_daily[df_daily["Datetime"] == i].iloc[0]["High"]) and (
                    j >= max(df_daily[df_daily["Datetime"] == i].iloc[0]["Open"],
                             df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]))) or \
                              (j <= df_daily[df_daily["Datetime"] == i].iloc[0]["High"] * (1 + 0.001) and j >=
                               df_daily[df_daily["Datetime"] == i].iloc[0]["High"] * (1 - 0.001))):
                high_pivot_bars[i] = j
                break

    for i in confluence_points.keys():
        for j in confluence_points[i].keys():
            if (df_daily[df_daily["Datetime"] == i].iloc[0]["Close"] > df_daily[df_daily["Datetime"] == i].iloc[0][
                "Open"]) and (((j >= df_daily[df_daily["Datetime"] == i].iloc[0]["Low"]) and (
                    j <= min(df_daily[df_daily["Datetime"] == i].iloc[0]["Open"],
                             df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]))) or \
                              (j <= df_daily[df_daily["Datetime"] == i].iloc[0]["Low"] * (1 + 0.001) and j >=
                               df_daily[df_daily["Datetime"] == i].iloc[0]["Low"] * (1 - 0.001))):
                low_pivot_bars[i] = j
                break

    high_pivots = {x: high_pivot_bars[x] for x in list(high_pivot_bars.keys()) if x not in list(low_pivot_bars.keys())}
    low_pivots = {x: low_pivot_bars[x] for x in list(low_pivot_bars.keys()) if x not in list(high_pivot_bars.keys())}

    with open(f'{ticker}_high_pivots_with_confluence_point_responsible.pkl', 'wb') as file:
        pickle.dump(high_pivots, file)

    with open(f'{ticker}_low_pivots_with_confluence_point_responsible.pkl', 'wb') as file:
        pickle.dump(low_pivots, file)

    return


def cache_pivots_along_with_confluence_and_confluence_strength_based_on_mcmc_weights(ticker):
    with open(f'{ticker}_high_pivots_with_confluence_point_responsible_and_its_lookbacks.pkl', 'rb') as file:
        low_pivots = pickle.load(file)

    return


def cache_percentiles_for_each_day_based_on_historical_returns(ticker, number_of_days_for_forward_returns_calculation,
                                                               forward_returns_type):
    with open(f'{ticker}_forward_returns.pkl', 'rb') as file:
        forward_returns = pickle.load(file)

    percentile_for_each_day = {}
    returns_list = []

    for i in forward_returns.index:
        try:
            current_return = forward_returns[f"{forward_returns_type}"][i][
                f"{number_of_days_for_forward_returns_calculation}_day"]
            returns_list.append(current_return)
            percentile_for_each_day[i] = stat.percentileofscore(returns_list, current_return)
        except Exception as e:
            continue

    with open(
            f'{ticker}_percentiles_for_each_day_based_on_{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type}.pkl',
            'wb') as file:
        pickle.dump(percentile_for_each_day, file)

    return


def cache_percentiles_for_each_day_based_on_absolute_historical_returns(ticker,
                                                                        number_of_days_for_forward_returns_calculation,
                                                                        forward_returns_type):
    with open(f'{ticker}_forward_returns.pkl', 'rb') as file:
        forward_returns = pickle.load(file)

    percentile_for_each_day = {}
    returns_list = []

    for i in forward_returns.index:
        try:
            current_return = abs(
                forward_returns[f"{forward_returns_type}"][i][f"{number_of_days_for_forward_returns_calculation}_day"])
            returns_list.append(current_return)
            percentile_for_each_day[i] = stat.percentileofscore(returns_list, current_return)
        except Exception as e:
            continue

    with open(
            f'{ticker}_percentiles_for_each_day_based_on_{number_of_days_for_forward_returns_calculation}_day_absolute_{forward_returns_type}.pkl',
            'wb') as file:
        pickle.dump(percentile_for_each_day, file)

    return


def cache_percentiles_for_each_day_based_on_all_days_returns(ticker, number_of_days_for_forward_returns_calculation,
                                                             forward_returns_type):
    with open(f'{ticker}_forward_returns.pkl', 'rb') as file:
        forward_returns = pickle.load(file)

    percentile_for_each_day = {}
    returns_list = []

    for i in forward_returns.index:
        try:
            current_return = forward_returns[f"{forward_returns_type}"][i][
                f"{number_of_days_for_forward_returns_calculation}_day"]
            returns_list.append(current_return)
            # percentile_for_each_day[i] = stat.percentileofscore(returns_list,current_return)
        except Exception as e:
            continue

    for i in forward_returns.index:
        try:
            current_return = forward_returns[f"{forward_returns_type}"][i][
                f"{number_of_days_for_forward_returns_calculation}_day"]
            percentile_for_each_day[i] = stat.percentileofscore(returns_list, current_return)
        except Exception as e:
            continue

    with open(
            f'{ticker}_percentiles_based_on_all_days_{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type}_returns.pkl',
            'wb') as file:
        pickle.dump(percentile_for_each_day, file)

    return


def cache_percentiles_of_distance_of_closest_confluence_point_for_each_day(ticker, confluence_points_cache):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    df_daily = get_data(ticker, "D")
    closest_confluence_points_with_distance_and_constituents_and_lookbacks_and_distance_percentile = {}

    for i in confluence_points.keys():
        day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
        try:
            closest_confluence = list(confluence_points[i].keys())[0]
            for j in confluence_points[i].keys():
                if abs(day_close - j) < abs(day_close - closest_confluence):
                    closest_confluence = j
            closest_confluence_points_with_distance_and_constituents_and_lookbacks_and_distance_percentile[i] = {
                "closest_confluence": closest_confluence, "distance": abs(day_close - closest_confluence),
                "details": confluence_points[i][closest_confluence]}
        except Exception as e:
            print(i)
            # print(e)
            continue

    distance_list = []
    for i in closest_confluence_points_with_distance_and_constituents_and_lookbacks_and_distance_percentile.keys():
        current_distance = \
        closest_confluence_points_with_distance_and_constituents_and_lookbacks_and_distance_percentile[i]["distance"]
        distance_list.append(current_distance)
        closest_confluence_points_with_distance_and_constituents_and_lookbacks_and_distance_percentile[i][
            "distance_percentile_score"] = stat.percentileofscore(distance_list, current_distance)

    with open(f'{ticker}_percentiles_of_distance_of_closest_confluence_point_for_each_day.pkl', 'wb') as file:
        pickle.dump(closest_confluence_points_with_distance_and_constituents_and_lookbacks_and_distance_percentile,
                    file)

    return


def cache_top_most_confluence_point_for_each_day(ticker, confluence_points_cache):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    df_daily = get_data(ticker, "D")
    top_most_confluence_points_with_distance_and_constituents_and_lookbacks = {}

    for i in confluence_points.keys():
        day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
        try:
            top_most_confluence = max(confluence_points[i].keys())
            top_most_confluence_points_with_distance_and_constituents_and_lookbacks[i] = {
                "top_most_confluence": top_most_confluence, "distance": abs(day_close - top_most_confluence),
                "details": confluence_points[i][top_most_confluence]}
        except Exception as e:
            print(i)
            # print(e)
            continue

    with open(f'{ticker}_top_most_confluence_point_for_each_day.pkl', 'wb') as file:
        pickle.dump(top_most_confluence_points_with_distance_and_constituents_and_lookbacks, file)

    return


def cache_top_most_non_white_confluence_point_for_each_day(ticker, confluence_points_cache):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    df_daily = get_data(ticker, "D")
    top_most_non_white_confluence_points_with_distance_and_constituents_and_lookbacks = {}

    for i in confluence_points.keys():
        day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
        try:
            non_white_confluence_points = []
            for j in confluence_points[i].keys():
                if confluence_points[i][j]["confluence_strength"] > 1:
                    non_white_confluence_points.append(j)
            top_non_white_most_confluence = max(non_white_confluence_points)
            top_most_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[i] = {
                "top_most_confluence": top_non_white_most_confluence,
                "distance": abs(day_close - top_non_white_most_confluence),
                "details": confluence_points[i][top_non_white_most_confluence]}
        except Exception as e:
            print(i)
            # print(e)
            continue

    with open(f'{ticker}_top_most_non_white_confluence_point_for_each_day.pkl', 'wb') as file:
        pickle.dump(top_most_non_white_confluence_points_with_distance_and_constituents_and_lookbacks, file)

    return


def cache_percentiles_of_non_absolute_distance_for_closest_non_white_confluence_point_for_each_day(ticker,
                                                                                                   confluence_points_cache):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    df_daily = get_data(ticker, "D")
    closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks = {}

    for i in confluence_points.keys():
        day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
        try:
            closest_non_white_confluence = list(confluence_points[i].keys())[0]
            for j in confluence_points[i].keys():
                if abs(day_close - j) < abs(day_close - closest_non_white_confluence) and confluence_points[i][j][
                    "confluence_strength"] > 1:
                    closest_non_white_confluence = j
            closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[i] = {
                "closest_confluence": closest_non_white_confluence,
                "distance": day_close - closest_non_white_confluence,
                "details": confluence_points[i][closest_non_white_confluence]}
        except Exception as e:
            print(i)
            # print(e)
            continue

    distance_list = []
    for i in closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks.keys():
        current_distance = closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[i][
            "distance"]
        distance_list.append(current_distance)
        closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[i][
            "distance_percentile_score"] = stat.percentileofscore(distance_list, current_distance)

    with open(f'{ticker}_percentiles_of_non_absolute_distance_for_closest_non_white_confluence_point_for_each_day.pkl',
              'wb') as file:
        pickle.dump(closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks, file)

    return


def cache_percentiles_of_absolute_distance_for_closest_non_white_confluence_point_for_each_day(ticker,
                                                                                               confluence_points_cache):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    df_daily = get_data(ticker, "D")
    closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks = {}

    for i in confluence_points.keys():
        day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
        try:
            closest_non_white_confluence = day_close * 1000
            for j in confluence_points[i].keys():
                if abs(day_close - j) < abs(day_close - closest_non_white_confluence) and confluence_points[i][j][
                    "confluence_strength"] >= 1:
                    closest_non_white_confluence = j
            closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[i] = {
                "closest_confluence": closest_non_white_confluence,
                "distance": abs(day_close - closest_non_white_confluence),
                "details": confluence_points[i][closest_non_white_confluence]}
        except Exception as e:
            print(i)
            # print(e)
            continue

    distance_list = []
    for i in closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks.keys():
        current_distance = closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[i][
            "distance"]
        distance_list.append(current_distance)
        closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[i][
            "distance_percentile_score"] = stat.percentileofscore(distance_list, current_distance)

    with open(f'{ticker}_percentiles_of_absolute_distance_for_closest_non_white_confluence_point_for_each_day.pkl',
              'wb') as file:
        pickle.dump(closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks, file)

    return


def cache_percentiles_of_minimum_8_absolute_distances_for_closest_non_white_confluence_point_for_each_day(ticker,
                                                                                                          confluence_points_cache):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    df_daily = get_data(ticker, "D")
    closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks = {}

    for i in confluence_points.keys():
        day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
        try:
            closest_non_white_confluence = day_close * 1000
            for j in confluence_points[i].keys():
                if abs(day_close - j) < abs(day_close - closest_non_white_confluence) and confluence_points[i][j][
                    "confluence_strength"] >= 1:
                    closest_non_white_confluence = j
            closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[i] = {
                "closest_confluence": closest_non_white_confluence,
                "distance": abs(day_close - closest_non_white_confluence),
                "details": confluence_points[i][closest_non_white_confluence]}
        except Exception as e:
            print(i)
            # print(e)
            continue

    distance_list = []
    for i in df_daily.index:
        try:
            closest_non_white_confluence = \
            closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[
                df_daily.loc[i]["Datetime"]]["closest_confluence"]
            current_min_distance = min(abs(closest_non_white_confluence - df_daily.loc[i]["Open"]),
                                       abs(closest_non_white_confluence - df_daily.loc[i]["High"]),
                                       abs(closest_non_white_confluence - df_daily.loc[i]["Low"]),
                                       abs(closest_non_white_confluence - df_daily.loc[i]["Close"]),
                                       abs(closest_non_white_confluence - df_daily.loc[i - 1]["Open"]),
                                       abs(closest_non_white_confluence - df_daily.loc[i - 1]["High"]),
                                       abs(closest_non_white_confluence - df_daily.loc[i - 1]["Low"]),
                                       abs(closest_non_white_confluence - df_daily.loc[i - 1]["Close"]))
            distance_list.append(current_min_distance)
            closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[
                df_daily.loc[i]["Datetime"]][
                "distance_percentile_score"] = stat.percentileofscore(distance_list, current_min_distance)
            closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[
                df_daily.loc[i]["Datetime"]][
                "min_of_abs_8_ohlc_data"] = current_min_distance
        except Exception as e:
            # print(i)
            print(e)
            continue

    with open(
            f'{ticker}_percentiles_of_minimum_8_absolute_distance_for_closest_non_white_confluence_point_for_each_day.pkl',
            'wb') as file:
        pickle.dump(closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks, file)

    return


def cache_percentiles_of_minimum_same_day_ohlc_absolute_distances_for_closest_non_white_confluence_point_for_each_day(
        ticker, confluence_points_cache):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    df_daily = get_data(ticker, "D")
    closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks = {}

    for i in confluence_points.keys():
        day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
        try:
            closest_non_white_confluence = day_close * 1000
            for j in confluence_points[i].keys():
                if abs(day_close - j) < abs(day_close - closest_non_white_confluence) and confluence_points[i][j][
                    "confluence_strength"] >= 1:
                    closest_non_white_confluence = j
            closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[i] = {
                "closest_confluence": closest_non_white_confluence,
                "distance": abs(day_close - closest_non_white_confluence),
                "details": confluence_points[i][closest_non_white_confluence]}
        except Exception as e:
            print(i)
            # print(e)
            continue

    distance_list = []
    for i in df_daily.index:
        try:
            closest_non_white_confluence = \
            closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[
                df_daily.loc[i]["Datetime"]]["closest_confluence"]
            current_min_distance = min(abs(closest_non_white_confluence - df_daily.loc[i]["Open"]),
                                       abs(closest_non_white_confluence - df_daily.loc[i]["High"]),
                                       abs(closest_non_white_confluence - df_daily.loc[i]["Low"]),
                                       abs(closest_non_white_confluence - df_daily.loc[i]["Close"]))
            distance_list.append(current_min_distance)
            closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[
                df_daily.loc[i]["Datetime"]][
                "distance_percentile_score"] = stat.percentileofscore(distance_list, current_min_distance)
            closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[
                df_daily.loc[i]["Datetime"]][
                "min_of_abs_same_day_ohlc_data"] = current_min_distance
        except Exception as e:
            # print(i)
            print(e)
            continue

    with open(
            f'{ticker}_percentiles_of_minimum_same_day_ohlc_absolute_distance_for_closest_non_white_confluence_point_for_each_day.pkl',
            'wb') as file:
        pickle.dump(closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks, file)

    return


def cache_percentiles_of_minimum_same_day_ohlc_absolute_distances_and_0_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day(
        ticker, confluence_points_cache):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    df_daily = get_data(ticker, "D")
    closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks = {}

    for i in confluence_points.keys():
        day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
        try:
            closest_non_white_confluence = day_close * 1000
            for j in confluence_points[i].keys():
                if abs(day_close - j) < abs(day_close - closest_non_white_confluence) and confluence_points[i][j][
                    "confluence_strength"] >= 1:
                    closest_non_white_confluence = j
            closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[i] = {
                "closest_confluence": closest_non_white_confluence,
                "distance": abs(day_close - closest_non_white_confluence),
                "details": confluence_points[i][closest_non_white_confluence]}
        except Exception as e:
            print(i)
            # print(e)
            continue

    distance_list = []
    for i in df_daily.index:
        try:
            closest_non_white_confluence = \
            closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[
                df_daily.loc[i]["Datetime"]]["closest_confluence"]
            if closest_non_white_confluence >= df_daily.loc[i]["Low"] and closest_non_white_confluence <= \
                    df_daily.loc[i]["High"]:
                closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[
                    df_daily.loc[i]["Datetime"]]["min_of_abs_same_day_ohlc_data"] = -1
            else:
                current_min_distance = min(abs(closest_non_white_confluence - df_daily.loc[i]["Open"]),
                                           abs(closest_non_white_confluence - df_daily.loc[i]["High"]),
                                           abs(closest_non_white_confluence - df_daily.loc[i]["Low"]),
                                           abs(closest_non_white_confluence - df_daily.loc[i]["Close"]))
                distance_list.append(current_min_distance)
                closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[
                    df_daily.loc[i]["Datetime"]][
                    "distance_percentile_score"] = stat.percentileofscore(distance_list, current_min_distance)
                closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks[
                    df_daily.loc[i]["Datetime"]][
                    "min_of_abs_same_day_ohlc_data"] = current_min_distance
        except Exception as e:
            # print(i)
            print(e)
            continue

    with open(
            f'{ticker}_percentiles_of_minimum_same_day_ohlc_absolute_distances_and_0_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day.pkl',
            'wb') as file:
        pickle.dump(closest_non_white_confluence_points_with_distance_and_constituents_and_lookbacks, file)

    return


def cache_days_with_percentiles_of_minimum_same_day_ohlc_absolute_distances_between_0_and_20_and_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day(
        ticker, percentile_cache):
    with open(f'{percentile_cache}', 'rb') as file:
        absolute_percentile_closest_non_white = pickle.load(file)

    days_with_percentiles_of_minimum_same_day_ohlc_absolute_distances_between_0_and_20_and_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point = []

    for i in absolute_percentile_closest_non_white.keys():
        if (absolute_percentile_closest_non_white[i]["min_of_abs_same_day_ohlc_data"] == -1) or (
                absolute_percentile_closest_non_white[i]["distance_percentile_score"] >= 0 and
                absolute_percentile_closest_non_white[i]["distance_percentile_score"] <= 20):
            days_with_percentiles_of_minimum_same_day_ohlc_absolute_distances_between_0_and_20_and_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point.append(
                i)

    with open(
            f'{ticker}_days_with_percentiles_absolute_distances_between_0_and_20_and_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day.pkl',
            'wb') as file:
        pickle.dump(
            days_with_percentiles_of_minimum_same_day_ohlc_absolute_distances_between_0_and_20_and_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point,
            file)

    return


def cache_indicator_data_for_days_with_percentiles_of_minimum_same_day_ohlc_absolute_distances_between_0_and_20_and_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day(
        df_daily, ticker, percentile_day_cache, distance_percentage_cache):
    # with open(f'{percentile_day_cache}', 'rb') as file:
    #     days_with_percentile_between_0_and_20 = pickle.load(file)

    with open(f'{distance_percentage_cache}', 'rb') as file:
        distance_percentage_for_days_with_percentile_between_0_and_20 = pickle.load(file)

    for i in fisher_lookback_list:
        df_daily[f"{i}_day_fisher"] = fisher(df_daily, i)

    df_daily = constance_brown(df_daily)
    df_daily.drop(["Open", "High", "Low", "Close", "Volume"], axis=1, inplace=True)
    df_daily.set_index("Datetime", inplace=True)
    indicator_data_for_days_with_percentile_between_0_and_20_df = df_daily.merge(
        distance_percentage_for_days_with_percentile_between_0_and_20, left_index=True, right_index=True, how='inner')

    with open(f'{ticker}_indicator_data_for_orange_days.pkl', 'wb') as file:
        pickle.dump(indicator_data_for_days_with_percentile_between_0_and_20_df, file)

    return


def cache_absolute_percentage_of_poc_and_days_close_across_various_lookbacks(ticker):
    with open(f'VolumeFeatures_Daily_{ticker}.pkl', 'rb') as file:
        volume_levels_df = pickle.load(file)

    df_daily = get_data(ticker, "D")

    price_level_columns = [i for i in volume_levels_df.columns if i.startswith("PriceLevels")]
    absolute_poc_distance_percentages = {}
    for i in volume_levels_df.index:
        try:
            absolute_distance_percentages_across_lookbacks = {}
            days_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
            for j in price_level_columns:
                distance_percentage = abs(float(days_close - volume_levels_df[j][i]["poc"]) / float(days_close)) * 100
                absolute_distance_percentages_across_lookbacks[j.split("_")[1]] = distance_percentage
            absolute_poc_distance_percentages[i] = absolute_distance_percentages_across_lookbacks
        except:
            absolute_poc_distance_percentages[i] = absolute_distance_percentages_across_lookbacks
            continue

    with open(f'{ticker}_absolute_poc_distance_percentages.pkl', 'wb') as file:
        pickle.dump(absolute_poc_distance_percentages, file)

    return


def cache_non_absolute_percentage_of_poc_and_days_close_across_various_lookbacks(ticker):
    with open(f'VolumeFeatures_Daily_{ticker}.pkl', 'rb') as file:
        volume_levels_df = pickle.load(file)

    df_daily = get_data(ticker, "D")

    price_level_columns = [i for i in volume_levels_df.columns if i.startswith("PriceLevels")]
    non_absolute_poc_distance_percentages = {}
    for i in volume_levels_df.index:
        try:
            non_absolute_distance_percentages_across_lookbacks = {}
            days_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
            for j in price_level_columns:
                distance_percentage = (float(days_close - volume_levels_df[j][i]["poc"]) / float(days_close)) * 100
                non_absolute_distance_percentages_across_lookbacks[j.split("_")[1]] = distance_percentage
            non_absolute_poc_distance_percentages[i] = non_absolute_distance_percentages_across_lookbacks
        except:
            non_absolute_poc_distance_percentages[i] = non_absolute_distance_percentages_across_lookbacks
            continue

    with open(f'{ticker}_non_absolute_poc_distance_percentages.pkl', 'wb') as file:
        pickle.dump(non_absolute_poc_distance_percentages, file)

    return


def cache_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days(
        ticker, days_list_cache):
    with open(f'VolumeFeatures_Daily_{ticker}.pkl', 'rb') as file:
        volume_levels_df = pickle.load(file)

    with open(f'{days_list_cache}', 'rb') as file:
        days_list_for_calculation = pickle.load(file)

    df_daily = get_data(ticker, "D")

    price_level_columns = [i for i in volume_levels_df.columns if i.startswith("PriceLevels")]
    non_absolute_poc_distance_percentages = {}
    for i in volume_levels_df.index:
        if i in days_list_for_calculation:
            non_absolute_distance_percentages_across_lookbacks = {}
            day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
            day_open = df_daily[df_daily["Datetime"] == i].iloc[0]["Open"]
            day_high = df_daily[df_daily["Datetime"] == i].iloc[0]["High"]
            day_low = df_daily[df_daily["Datetime"] == i].iloc[0]["Low"]
            ohlc_avg = (day_open + day_close + day_high + day_low) / 4
            try:
                for j in price_level_columns:
                    distance_percentage = (float(ohlc_avg - volume_levels_df.loc[i][j]["poc"]) / float(ohlc_avg)) * 100
                    non_absolute_distance_percentages_across_lookbacks[
                        j.split("_")[1] + "_day_lookback"] = distance_percentage
                non_absolute_poc_distance_percentages[i] = non_absolute_distance_percentages_across_lookbacks
            except Exception as e:
                print(e)
                continue

    non_absolute_poc_distance_percentages_df = pd.DataFrame().from_dict(non_absolute_poc_distance_percentages,
                                                                        orient='index')

    with open(
            f'{ticker}_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days.pkl',
            'wb') as file:
        pickle.dump(non_absolute_poc_distance_percentages_df, file)

    return


def cache_kmeans_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks(df_daily, ticker,
                                                                                                     percentage_distance_cache,
                                                                                                     number_of_clusters,
                                                                                                     train_start_year=None,
                                                                                                     train_end_year=None):
    with open(f'{percentage_distance_cache}', 'rb') as file:
        distance_percentages_df = pickle.load(file)

    distance_percentages_df = distance_percentages_df[(distance_percentages_df.index.year >= train_start_year) & (
                distance_percentages_df.index.year <= train_end_year)]

    kmeans = KMeans(number_of_clusters)
    kmeans.fit(distance_percentages_df)
    identified_clusters = kmeans.fit_predict(distance_percentages_df)
    distance_percentages_df_with_clusters = distance_percentages_df.copy()
    distance_percentages_df_with_clusters['Clusters'] = identified_clusters

    with open(
            f"{ticker}_k-means_clusters_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters.pkl",
            'wb') as file:
        pickle.dump([kmeans, distance_percentages_df_with_clusters], file)

    return


def cache_gmm_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks(df_daily, ticker,
                                                                                                  percentage_distance_cache,
                                                                                                  number_of_clusters,
                                                                                                  train_start_year=None,
                                                                                                  train_end_year=None):
    with open(f'{percentage_distance_cache}', 'rb') as file:
        distance_percentages_df = pickle.load(file)

    distance_percentages_df = distance_percentages_df[(distance_percentages_df.index.year >= train_start_year) & (
                distance_percentages_df.index.year <= train_end_year)]

    gmm = GaussianMixture(n_components=number_of_clusters)
    gmm.fit(distance_percentages_df)
    identified_clusters = gmm.predict(distance_percentages_df)
    distance_percentages_df_with_clusters = distance_percentages_df.copy()
    distance_percentages_df_with_clusters['Clusters'] = identified_clusters

    with open(
            f"{ticker}_gmm_clusters_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters.pkl",
            'wb') as file:
        pickle.dump([gmm, distance_percentages_df_with_clusters], file)

    return


# deprecated
def forward_statistical_analysis_for_points_in_each_cluster_helper_old(ticker, clusters_df, forward_statistics_cache,
                                                                       number_of_clusters, train_start_year,
                                                                       train_end_year, clustering_type):
    with open(f'{forward_statistics_cache}', 'rb') as file:
        forward_statistics_df = pickle.load(file)

    statistical_analysis_df = pd.DataFrame(
        columns=["ticker", "clustering_technique", "training_start", "training_end", "total_number_of_clusters",
                 "cluster_number", "min", "max", "std_dev", "median", "max/min"])

    df_list = []

    for i in range(number_of_clusters):
        cluster_dates = []
        for j in clusters_df.index:
            if clusters_df.loc[j]["Clusters"] == i:
                cluster_dates.append(j)
        statistical_analysis_dict = {}
        for k in ["min_returns", "max_returns", "end_to_end_returns", "end_to_end_sharpe"]:
            for l in forward_return_list_in_days:
                statistical_analysis_list = []
                for m in cluster_dates:
                    statistical_analysis_list.append(forward_statistics_df.loc[m][k][f"{l}_day"])
                statistical_analysis_dict[str(l) + "_day_" + k] = statistical_analysis_list
        df_row = {"ticker": ticker, "clustering_technique": clustering_type, "training_start": train_start_year,
                  "training_end": train_end_year, "total_number_of_clusters": number_of_clusters,
                  "cluster_number": i,
                  "min": {key: round(min(value), 2) for key, value in statistical_analysis_dict.items()},
                  "max": {key: round(max(value), 2) for key, value in statistical_analysis_dict.items()}
            , "std_dev": {key: round(np.std(value), 2) for key, value in statistical_analysis_dict.items()},
                  "median": {key: round(statistics.median(value), 2) for key, value in
                             statistical_analysis_dict.items()},
                  "max/min": {key: round(max(value) / min(value), 2) for key, value in
                              statistical_analysis_dict.items()}}
        df_list.append(df_row)

    statistical_analysis_df = pd.DataFrame(df_list)

    return statistical_analysis_df


# deprecated
def cache_forward_statistical_analysis_for_points_in_each_cluster_old(ticker, forward_statistics_cache):
    statistical_analysis_df = pd.DataFrame(
        columns=["ticker", "clustering_technique", "training_start", "training_end", "total_number_of_clusters",
                 "cluster_number", "min", "max", "std_dev", "median", "max/min"])

    for i in tqdm(train_start_date):
        for j in tqdm(train_end_date):
            for k in number_of_clusters_config:
                for c in clustering_type:
                    with open(
                            f'{ticker}_{c}_clusters_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{i}_to_{j}_with_{k}_clusters.pkl',
                            'rb') as file:
                        clusters_df = pickle.load(file)
                    clusters_df = clusters_df[["Clusters"]]
                    temp_df = forward_statistical_analysis_for_points_in_each_cluster_helper_old(ticker, clusters_df,
                                                                                                 forward_statistics_cache,
                                                                                                 k, i, j, c)
                    statistical_analysis_df = pd.concat([statistical_analysis_df, temp_df], ignore_index=True)

    with open(f"{ticker}_statistical_analysis_all_combinations_of_clusters.pkl", 'wb') as file:
        pickle.dump(statistical_analysis_df, file)

    return


def forward_statistical_analysis_for_points_in_each_cluster_helper(ticker, clusters_df, forward_statistics_cache,
                                                                   number_of_clusters, train_start_year, train_end_year,
                                                                   clustering_type):
    with open(f'{forward_statistics_cache}', 'rb') as file:
        forward_statistics_df = pickle.load(file)

    df_list = []

    for i in range(number_of_clusters):
        cluster_dates = []
        for j in clusters_df.index:
            if clusters_df.loc[j]["Clusters"] == i:
                cluster_dates.append(j)
        statistical_analysis_dict = {}
        for k in ["min_returns", "max_returns", "end_to_end_returns", "end_to_end_sharpe"]:
            for l in forward_return_list_in_days:
                statistical_analysis_list = []
                for m in cluster_dates:
                    statistical_analysis_list.append(forward_statistics_df.loc[m][k][f"{l}_day"])
                statistical_analysis_dict[str(l) + "_day_" + k] = statistical_analysis_list
        df_row = {"ticker": ticker, "clustering_technique": clustering_type, "training_start": train_start_year,
                  "training_end": train_end_year, "total_number_of_clusters": number_of_clusters,
                  "cluster_number": i, }
        for n in statistical_analysis_dict.keys():
            df_row[n + "_min"] = round(min(statistical_analysis_dict[n]), 4)
            df_row[n + "_max"] = round(max(statistical_analysis_dict[n]), 4)
            df_row[n + "_std_dev"] = round(np.std(statistical_analysis_dict[n]), 4)
            df_row[n + "_median"] = round(statistics.median(statistical_analysis_dict[n]), 4)
            df_row[n + "_max/min"] = round(max(statistical_analysis_dict[n]) / min(statistical_analysis_dict[n]), 4)

        df_list.append(df_row)

    statistical_analysis_df = pd.DataFrame(df_list)

    return statistical_analysis_df


def cache_forward_statistical_analysis_for_points_in_each_cluster(ticker, forward_statistics_cache):
    statistical_analysis_df = pd.DataFrame()

    for i in tqdm(train_start_date):
        for j in tqdm(train_end_date):
            for k in number_of_clusters_config:
                for c in clustering_type:
                    with open(
                            f'{ticker}_{c}_clusters_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{i}_to_{j}_with_{k}_clusters.pkl',
                            'rb') as file:
                        clusters_df = pickle.load(file)
                    clusters_df = clusters_df[1][["Clusters"]]
                    temp_df = forward_statistical_analysis_for_points_in_each_cluster_helper(ticker, clusters_df,
                                                                                             forward_statistics_cache,
                                                                                             k, i, j, c)
                    statistical_analysis_df = pd.concat([statistical_analysis_df, temp_df], ignore_index=True)

    with open(f"{ticker}_statistical_analysis_all_combinations_of_clusters.pkl", 'wb') as file:
        pickle.dump(statistical_analysis_df, file)

    return


def cache_top_5_and_bottom_5_of_forward_statistical_analysis_for_points_in_each_cluster(ticker,
                                                                                        forward_statistics_cache):
    with open(f'{forward_statistics_cache}', 'rb') as file:
        forward_statistics_df = pickle.load(file)

    for i in forward_statistics_df.index:
        for j in ["min", "max", "std_dev", "median", "max/min"]:
            unsorted_dictionary = forward_statistics_df.loc[i][j]
            sorted_dictionary = {k: v for k, v in sorted(unsorted_dictionary.items(), key=lambda item: item[1])}
            keys_top_5 = list(sorted_dictionary.keys())[:5]
            keys_bottom_5 = list(sorted_dictionary.keys())[-5:]
            top_and_bottom_5 = {k: sorted_dictionary[k] for k in keys_top_5 + keys_bottom_5}
            forward_statistics_df.loc[i][j] = top_and_bottom_5

    with open(f"{ticker}_top_5_and_bottom_5_of_forward_statistical_analysis_for_points_in_each_cluster.pkl",
              'wb') as file:
        pickle.dump(forward_statistics_df, file)

    return


def cache_top_5_and_bottom_5_of_forward_statistical_measures_across_all_cluster_combinations(ticker,
                                                                                             forward_statistics_cache):
    with open(f'{forward_statistics_cache}', 'rb') as file:
        forward_statistics_df = pickle.load(file)

    top_5_bottom_5_dict = {}
    for i in forward_statistics_df.columns:
        if i not in ["ticker", "clustering_technique", "training_start", "training_end", "total_number_of_clusters",
                     "cluster_number"]:
            sorted_df = forward_statistics_df.sort_values(by=[i])
            top_5 = sorted_df.iloc[-5:]
            bottom_5 = sorted_df.iloc[:5]
            top_5_bottom_5_list = []
            for j in top_5.index:
                cluster_details = top_5.loc[j]["ticker"] + "_" + top_5.loc[j][
                    "clustering_technique"] + "_trained_from_" + str(top_5.loc[j]["training_start"]) + "_to_" + str(
                    top_5.loc[j]["training_end"]) + "_total_clusters_" + \
                                  str(top_5.loc[j]["total_number_of_clusters"]) + "_cluster_number_" + str(
                    top_5.loc[j]["cluster_number"]) + f"_with_value = " + str(top_5.loc[j][i])
                top_5_bottom_5_list.append(cluster_details)
            for k in bottom_5.index:
                cluster_details = bottom_5.loc[k]["ticker"] + "_" + bottom_5.loc[k][
                    "clustering_technique"] + "_trained_from_" + str(bottom_5.loc[k]["training_start"]) + "_to_" + str(
                    bottom_5.loc[k][
                        "training_end"]) + "_total_clusters_" + \
                                  str(bottom_5.loc[k]["total_number_of_clusters"]) + "_cluster_number_" + str(
                    bottom_5.loc[k][
                        "cluster_number"]) + f"_with_value = " + str(bottom_5.loc[k][i])
                top_5_bottom_5_list.append(cluster_details)
            top_5_bottom_5_dict[i] = top_5_bottom_5_list

    top_5_bottom_5_df = pd.DataFrame(top_5_bottom_5_dict)

    with open(f"{ticker}_top_5_and_bottom_5_of_forward_statistical_measures_across_all_cluster_combinations.pkl",
              'wb') as file:
        pickle.dump(top_5_bottom_5_df, file)

    return


def cache_out_of_sample_days_which_are_part_of_clusters_of_top_5_and_bottom_5_of_forward_statistical_measures_removing_duplicate_clusters(
        ticker, top_and_bottom_5_forward_statistics_cache, poc_distance_percentage_cache_as_features_for_clustering):
    with open(f'{top_and_bottom_5_forward_statistics_cache}', 'rb') as file:
        top_and_bottom_5_forward_statistics_df = pickle.load(file)

    with open(f'{poc_distance_percentage_cache_as_features_for_clustering}', 'rb') as file:
        distance_percentage_cache = pickle.load(file)

    all_types_of_clusters = []
    for i in top_and_bottom_5_forward_statistics_df.index:
        for j in top_and_bottom_5_forward_statistics_df.columns:
            all_types_of_clusters.append(top_and_bottom_5_forward_statistics_df.loc[i][j].split("=")[0])

    all_types_of_clusters = list(set(all_types_of_clusters))

    out_of_sample_clusters_dict = {}
    out_of_sample_clusters_df = distance_percentage_cache[distance_percentage_cache.index.year >= 2021]
    for i in all_types_of_clusters:
        cluster_details = i.split("_")
        cluster_cache_path = f"{cluster_details[0]}_{cluster_details[1]}_clusters_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{cluster_details[4]}_{cluster_details[5]}_" \
                             f"{cluster_details[6]}_with_{cluster_details[9]}_clusters.pkl"
        cluster_number = cluster_details[12]
        with open(f'{cluster_cache_path}', 'rb') as file:
            cluster_cache = pickle.load(file)
        cluster_model = cluster_cache[0]
        out_of_sample_clusters_df_with_cluster_number = out_of_sample_clusters_df.copy()
        out_of_sample_clusters_df_with_cluster_number["Clusters"] = cluster_model.predict(out_of_sample_clusters_df)
        for j in out_of_sample_clusters_df_with_cluster_number.index:
            # print(str(out_of_sample_clusters_df_with_cluster_number.loc[j]["Clusters"]))
            if str(int(out_of_sample_clusters_df_with_cluster_number.loc[j]["Clusters"])) == cluster_number:
                if j in out_of_sample_clusters_dict.keys():
                    out_of_sample_clusters_dict[j].append(i)
                else:
                    out_of_sample_clusters_dict[j] = [i]

    with open(
            f"{ticker}_out_of_sample_days_which_are_part_of_clusters_of_top_5_and_bottom_5_of_forward_statistical_measures.pkl",
            'wb') as file:
        pickle.dump(out_of_sample_clusters_dict, file)

    return


def cache_out_of_sample_days_which_are_part_of_clusters_of_top_40_and_bottom_40_of_forward_statistical_measures_by_combining_over_look_forwards(
        ticker, top_and_bottom_5_forward_statistics_cache, poc_distance_percentage_cache_as_features_for_clustering):
    with open(f'{top_and_bottom_5_forward_statistics_cache}', 'rb') as file:
        top_and_bottom_5_forward_statistics_df = pickle.load(file)

    with open(f'{poc_distance_percentage_cache_as_features_for_clustering}', 'rb') as file:
        distance_percentage_cache = pickle.load(file)

    clusters_in_different_forward_stats_aggregated_over_lookbacks = {}
    for i in top_and_bottom_5_forward_statistics_df.columns:
        remove_lookback = i.split("_")
        remove_lookback = "_".join(remove_lookback[2:])
        for j in top_and_bottom_5_forward_statistics_df.index:
            if remove_lookback not in clusters_in_different_forward_stats_aggregated_over_lookbacks.keys():
                clusters_in_different_forward_stats_aggregated_over_lookbacks[remove_lookback] = [(
                                                                                                  top_and_bottom_5_forward_statistics_df.loc[
                                                                                                      j][i].split("=")[
                                                                                                      0], float(
                                                                                                      top_and_bottom_5_forward_statistics_df.loc[
                                                                                                          j][i].split(
                                                                                                          "=")[1]),
                                                                                                  "_".join(i.split("_")[
                                                                                                           :2]))]
            else:
                clusters_in_different_forward_stats_aggregated_over_lookbacks[remove_lookback].append((
                                                                                                      top_and_bottom_5_forward_statistics_df.loc[
                                                                                                          j][i].split(
                                                                                                          "=")[0],
                                                                                                      float(
                                                                                                          top_and_bottom_5_forward_statistics_df.loc[
                                                                                                              j][
                                                                                                              i].split(
                                                                                                              "=")[1]),
                                                                                                      "_".join(
                                                                                                          i.split("_")[
                                                                                                          :2])))

    for i in clusters_in_different_forward_stats_aggregated_over_lookbacks.keys():
        clusters_in_different_forward_stats_aggregated_over_lookbacks[i] = sorted(
            clusters_in_different_forward_stats_aggregated_over_lookbacks[i], key=lambda x: x[1], reverse=True)
        # clusters_in_different_forward_stats_aggregated_over_lookbacks[i] = clusters_in_different_forward_stats_aggregated_over_lookbacks[i][:5] + clusters_in_different_forward_stats_aggregated_over_lookbacks[i][-5:]
        # clusters_in_different_forward_stats_aggregated_over_lookbacks_descending = sorted(clusters_in_different_forward_stats_aggregated_over_lookbacks[i], key=lambda x: x[1],reverse=True)
        # clusters_in_different_forward_stats_aggregated_over_lookbacks_ascending = sorted(clusters_in_different_forward_stats_aggregated_over_lookbacks[i], key=lambda x: x[1])

    with open(
            f"{ticker}_clusters_of_top_40_and_bottom_40_of_forward_statistical_measures_by_combining_over_look_forwards.pkl",
            'wb') as file:
        pickle.dump(clusters_in_different_forward_stats_aggregated_over_lookbacks, file)

    out_of_sample_clusters_dict_for_combined_look_forwards = {}
    for i in clusters_in_different_forward_stats_aggregated_over_lookbacks.keys():
        top_bottom_counter = 0
        out_of_sample_clusters_dict = {}
        out_of_sample_clusters_df = distance_percentage_cache[distance_percentage_cache.index.year >= 2021]
        for j in clusters_in_different_forward_stats_aggregated_over_lookbacks[i]:
            cluster_details = j[0].split("_")
            cluster_cache_path = f"{cluster_details[0]}_{cluster_details[1]}_clusters_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{cluster_details[4]}_{cluster_details[5]}_" \
                                 f"{cluster_details[6]}_with_{cluster_details[9]}_clusters.pkl"
            cluster_number = cluster_details[12]
            with open(f'{cluster_cache_path}', 'rb') as file:
                cluster_cache = pickle.load(file)
            cluster_model = cluster_cache[0]
            out_of_sample_clusters_df_with_cluster_number = out_of_sample_clusters_df.copy()
            out_of_sample_clusters_df_with_cluster_number["Clusters"] = cluster_model.predict(out_of_sample_clusters_df)
            for k in out_of_sample_clusters_df_with_cluster_number.index:
                # print(str(out_of_sample_clusters_df_with_cluster_number.loc[j]["Clusters"]))
                if str(int(out_of_sample_clusters_df_with_cluster_number.loc[k]["Clusters"])) == cluster_number:
                    if k in out_of_sample_clusters_dict.keys():
                        if top_bottom_counter < 40:
                            out_of_sample_clusters_dict[k].append((j[0], "top_40"))
                        else:
                            out_of_sample_clusters_dict[k].append((j[0], "bottom_40"))
                    else:
                        if top_bottom_counter < 40:
                            out_of_sample_clusters_dict[k] = [(j[0], "top_40")]
                        else:
                            out_of_sample_clusters_dict[k] = [(j[0], "bottom_40")]
            top_bottom_counter += 1
        out_of_sample_clusters_dict_for_combined_look_forwards[i] = out_of_sample_clusters_dict

    out_of_sample_clusters_dict_for_combined_look_forwards_filtered_top_and_bottom_based_on_count = {}
    for i in out_of_sample_clusters_dict_for_combined_look_forwards.keys():
        out_of_sample_clusters_dict_for_combined_look_forwards_filtered_top_and_bottom_based_on_count[i] = {}
        for j in out_of_sample_clusters_dict_for_combined_look_forwards[i]:
            out_of_sample_clusters_dict_for_combined_look_forwards_filtered_top_and_bottom_based_on_count[i][j] = ""
            top_and_bottom_list = []
            for k in out_of_sample_clusters_dict_for_combined_look_forwards[i][j]:
                top_and_bottom_list.append(k[1])
            if top_and_bottom_list.count("top_40") > top_and_bottom_list.count("bottom_40"):
                out_of_sample_clusters_dict_for_combined_look_forwards_filtered_top_and_bottom_based_on_count[i][
                    j] = "top_40"
            else:
                out_of_sample_clusters_dict_for_combined_look_forwards_filtered_top_and_bottom_based_on_count[i][
                    j] = "bottom_40"

    with open(
            f"{ticker}_out_of_sample_days_which_are_part_of_clusters_of_top_40_and_bottom_40_of_forward_statistical_measures_by_combining_over_look_forwards.pkl",
            'wb') as file:
        pickle.dump(out_of_sample_clusters_dict_for_combined_look_forwards_filtered_top_and_bottom_based_on_count, file)

    return


def cache_SVR_regression_for_days_with_percentiles_of_ohlc_absolute_distances_between_0_and_20_and_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day(
        ticker, days_cache, poc_percentage_distance_cache, forward_statistics_cache):
    with open(f'{days_cache}', 'rb') as file:
        orange_days = pickle.load(file)

    with open(f'{forward_statistics_cache}', 'rb') as file:
        forward_statistics_df = pickle.load(file)

    with open(f'{poc_percentage_distance_cache}', 'rb') as file:
        poc_percentage_distance_df = pickle.load(file)

    print("Hi")

    return


def cache_confluence_areas_along_with_lookbacks(ticker):
    with open(f'VolumeFeatures_{ticker}_Daily_test.pkl', 'rb') as file:
        volume_levels_df = pickle.load(file)

    price_level_columns = [i for i in volume_levels_df.columns if i.startswith("PriceLevels")]
    confluence_points = {}
    for i in volume_levels_df.index:
        try:
            all_points = []
            for j in price_level_columns:
                all_points.append((j.split("_")[1], volume_levels_df[j][i]["poc"]))
                all_points.append((j.split("_")[1], volume_levels_df[j][i]["vah"]))
                all_points.append((j.split("_")[1], volume_levels_df[j][i]["val"]))
                all_points.append((j.split("_")[1], volume_levels_df[j][i]["point_of_least_volume_in_va"]))
            confluence_points[i] = matching_logic_for_confluence_areas_for_each_day_return_with_lookback(all_points)
        except Exception as e:
            # print(e)
            continue

    with open(f'Confluence_Points_{ticker}_along_with_lookbacks.pkl', 'wb') as file:
        pickle.dump(confluence_points, file)

    return


def cache_confluence_areas_along_with_its_constituents_and_their_lookbacks(ticker):
    with open(f'VolumeFeatures_Daily_{ticker}.pkl', 'rb') as file:
        volume_levels_df = pickle.load(file)

    # volume_levels_df = pd.read_csv(f'VolumeFeatures_Daily_{ticker}_test.csv')

    price_level_columns = [i for i in volume_levels_df.columns if i.startswith("PriceLevels")]
    confluence_points = {}
    for i in volume_levels_df.index:
        all_points = []
        try:
            for j in price_level_columns:
                all_points.append([j.split("_")[1], volume_levels_df[j][i]["poc"], "poc"])
                # all_points.append([j.split("_")[1],volume_levels_df[j][i]["vah"]])
                # all_points.append([j.split("_")[1],volume_levels_df[j][i]["val"]])
                all_points.append(
                    [j.split("_")[1], volume_levels_df[j][i]["point_of_least_volume_in_va"], "polv_in_va"])
            confluence_points[
                i] = matching_logic_for_confluence_areas_for_each_day_return_with_constituent_and_their_lookback(
                all_points)
        except Exception as e:
            # print(e)
            confluence_points[
                i] = matching_logic_for_confluence_areas_for_each_day_return_with_constituent_and_their_lookback(
                all_points)
            continue

    with open(f'Confluence_Points_{ticker}_along_with_its_constituents_and_their_lookbacks.pkl', 'wb') as file:
        pickle.dump(confluence_points, file)

    return


def cache_confluence_areas_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength(ticker,
                                                                                                             vah_val_inclusion=True):
    with open(f'VolumeFeatures_Daily_{ticker}.pkl', 'rb') as file:
        volume_levels_df = pickle.load(file)

    # volume_levels_df = pd.read_csv(f'VolumeFeatures_Daily_{ticker}_test.csv')
    # volume_levels_df = volume_levels_df[:-2]

    price_level_columns = [i for i in volume_levels_df.columns if i.startswith("PriceLevels")]
    confluence_points = {}
    for i in volume_levels_df.index:
        all_points = []
        try:
            for j in price_level_columns:
                all_points.append([j.split("_")[1], volume_levels_df[j][i]["poc"], "poc"])
                all_points.append(
                    [j.split("_")[1], volume_levels_df[j][i]["point_of_least_volume_in_va"], "polv_in_va"])
                if vah_val_inclusion == True:
                    all_points.append([j.split("_")[1], volume_levels_df[j][i]["vah"], "vah"])
                    all_points.append([j.split("_")[1], volume_levels_df[j][i]["val"], "val"])
            confluence_points[
                i] = matching_logic_for_confluence_areas_for_each_day_return_with_constituent_and_their_lookback_with_weighted_confluence_strength(
                all_points)
        except Exception as e:
            # print(e)
            confluence_points[
                i] = matching_logic_for_confluence_areas_for_each_day_return_with_constituent_and_their_lookback_with_weighted_confluence_strength(
                all_points)
            continue

    if vah_val_inclusion == True:
        with open(
                f'Confluence_Points_{ticker}_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl',
                'wb') as file:
            pickle.dump(confluence_points, file)
    else:
        with open(
                f'Confluence_Points_{ticker}_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl',
                'wb') as file:
            pickle.dump(confluence_points, file)

    return


def exponential_function_for_confluence_constituents(x):
    return math.exp(-abs(x))


def cache_closest_confluence_areas_along_with_its_constituents_and_their_lookbacks_and_exponential_function_value(
        ticker):
    with open(f'Confluence_Points_{ticker}_along_with_its_constituents_and_their_lookbacks.pkl', 'rb') as file:
        confluence_points = pickle.load(file)

    df_daily = get_data(ticker, "D")
    closest_confluence_points_with_constituents_and_lookbacks_and_function_value = {}

    for i in confluence_points.keys():
        day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
        closest_confluence = list(confluence_points[i].keys())[0]
        for j in confluence_points[i].keys():
            if abs(day_close - j) < closest_confluence:
                closest_confluence = j
        closest_confluence_points_with_constituents_and_lookbacks_and_function_value[i] = [closest_confluence,
                                                                                           confluence_points[i][
                                                                                               closest_confluence]]

    for i in closest_confluence_points_with_constituents_and_lookbacks_and_function_value.keys():
        day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
        for j in range(len(closest_confluence_points_with_constituents_and_lookbacks_and_function_value[i][1])):
            closest_confluence_points_with_constituents_and_lookbacks_and_function_value[i][1][j].append(
                exponential_function_for_confluence_constituents(
                    closest_confluence_points_with_constituents_and_lookbacks_and_function_value[i][1][j][
                        1] - day_close))

    closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df = pd.DataFrame().from_dict(
        closest_confluence_points_with_constituents_and_lookbacks_and_function_value, orient='index',
        columns=["Closest_Confluence", "Constituents_of_Closest_Confluence"])

    with open(
            f'Closest_Confluence_Point_{ticker}_along_with_its_constituents_and_their_lookbacks_and_exponential_function_value.pkl',
            'wb') as file:
        pickle.dump(closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df, file)

    return


def cache_closest_confluence_areas_along_with_its_constituents_and_their_lookbacks_and_exponential_function_value_in_form_of_numpy_array(
        ticker):
    with open(f'Confluence_Points_{ticker}_along_with_its_constituents_and_their_lookbacks.pkl', 'rb') as file:
        confluence_points = pickle.load(file)

    df_daily = get_data(ticker, "D")
    closest_confluence_points_with_constituents_and_lookbacks_and_function_value = {}

    for i in confluence_points.keys():
        day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
        closest_confluence = list(confluence_points[i].keys())[0]
        for j in confluence_points[i].keys():
            if abs(day_close - j) < closest_confluence:
                closest_confluence = j
        closest_confluence_points_with_constituents_and_lookbacks_and_function_value[i] = [closest_confluence,
                                                                                           confluence_points[i][
                                                                                               closest_confluence]]

    for i in closest_confluence_points_with_constituents_and_lookbacks_and_function_value.keys():
        day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
        for j in range(len(closest_confluence_points_with_constituents_and_lookbacks_and_function_value[i][1])):
            closest_confluence_points_with_constituents_and_lookbacks_and_function_value[i][1][j].append(
                exponential_function_for_confluence_constituents(
                    closest_confluence_points_with_constituents_and_lookbacks_and_function_value[i][1][j][
                        1] - day_close))

    closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df = pd.DataFrame().from_dict(
        closest_confluence_points_with_constituents_and_lookbacks_and_function_value, orient='index',
        columns=["Closest_Confluence", "Constituents_of_Closest_Confluence"])

    # closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df = closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df[
    #     (closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df.index >= datetime.datetime.strptime('2012-01-01 00:00:00',"%Y-%m-%d %H:%M:%S")) &
    #     (closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df.index <= datetime.datetime.strptime('2022-02-01 00:00:00', "%Y-%m-%d %H:%M:%S"))
    #     ]

    lis = []
    for i in closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df.index:
        arr = lis.append(closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df.loc[i][
                             "Constituents_of_Closest_Confluence"])

    inside_lis_len = []
    for i in range(len(lis)):
        inside_lis_len.append(len(lis[i]))

    master_array = np.zeros(
        (len(closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df), max(inside_lis_len), 3))
    master_array_appender = 0

    for i in closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df[
        "Constituents_of_Closest_Confluence"]:
        innermost_list = []
        for j in i:
            innermost_list.append(j)
        for j in range(max(inside_lis_len) - len(innermost_list)):
            innermost_list.append([np.nan, np.nan, np.nan])
        arr = np.array(innermost_list)
        master_array[master_array_appender] = arr
        master_array_appender += 1

    with open(
            f'Closest_Confluence_Point_{ticker}_along_with_its_constituents_and_their_lookbacks_and_exponential_function_value_in_form_of_numpy_array.pkl',
            'wb') as file:
        pickle.dump(master_array, file)

    return


def cache_closest_confluence_areas_along_with_its_constituents_and_their_lookbacks_and_exponential_function_value_in_form_of_numpy_array_for_percentile_above_85(
        ticker, number_of_days_for_forward_returns_calculation, forward_returns_type):
    with open(f'Confluence_Points_{ticker}_along_with_its_constituents_and_their_lookbacks.pkl', 'rb') as file:
        confluence_points = pickle.load(file)

    with open(
            f'{ticker}_percentiles_for_each_day_based_on_{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type}_returns.pkl',
            'rb') as file:
        percentiles_for_each_day = pickle.load(file)

    days_above_85_percentile = []

    for i in percentiles_for_each_day.keys():
        if percentiles_for_each_day[i] >= 85:
            days_above_85_percentile.append(i)

    df_daily = get_data(ticker, "D")
    closest_confluence_points_with_constituents_and_lookbacks_and_function_value = {}

    for i in confluence_points.keys():
        if i in days_above_85_percentile:
            day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
            closest_confluence = list(confluence_points[i].keys())[0]
            for j in confluence_points[i].keys():
                if abs(day_close - j) < closest_confluence:
                    closest_confluence = j
            closest_confluence_points_with_constituents_and_lookbacks_and_function_value[i] = [closest_confluence,
                                                                                               confluence_points[i][
                                                                                                   closest_confluence]]

    for i in closest_confluence_points_with_constituents_and_lookbacks_and_function_value.keys():
        day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
        for j in range(len(closest_confluence_points_with_constituents_and_lookbacks_and_function_value[i][1])):
            closest_confluence_points_with_constituents_and_lookbacks_and_function_value[i][1][j].append(
                exponential_function_for_confluence_constituents(
                    closest_confluence_points_with_constituents_and_lookbacks_and_function_value[i][1][j][
                        1] - day_close))

    closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df = pd.DataFrame().from_dict(
        closest_confluence_points_with_constituents_and_lookbacks_and_function_value, orient='index',
        columns=["Closest_Confluence", "Constituents_of_Closest_Confluence"])

    lis = []
    for i in closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df.index:
        arr = lis.append(closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df.loc[i][
                             "Constituents_of_Closest_Confluence"])

    inside_lis_len = []
    for i in range(len(lis)):
        inside_lis_len.append(len(lis[i]))

    master_array = np.zeros(
        (len(closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df), max(inside_lis_len), 3))
    master_array_appender = 0

    for i in closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df[
        "Constituents_of_Closest_Confluence"]:
        innermost_list = []
        for j in i:
            innermost_list.append(j)
        for j in range(max(inside_lis_len) - len(innermost_list)):
            innermost_list.append([np.nan, np.nan, np.nan])
        arr = np.array(innermost_list)
        master_array[master_array_appender] = arr
        master_array_appender += 1

    with open(
            f'Closest_Confluence_Point_{ticker}_its_constituents_lookbacks_and_exponential_function_value_in_form_of_numpy_array_for_{number_of_days_for_forward_returns_calculation}_D_{forward_returns_type}_percentile_above_85.pkl',
            'wb') as file:
        pickle.dump(master_array, file)

    with open(f'{ticker}_forward_returns.pkl', 'rb') as file:
        forward_returns = pickle.load(file)

    target = {}
    target_values = []
    for i in forward_returns.index:
        if i in closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df.index:
            try:
                target[i] = forward_returns[f"{forward_returns_type}"][i][
                    f"{number_of_days_for_forward_returns_calculation}_day"]
                target_values.append(forward_returns[f"{forward_returns_type}"][i][
                                         f"{number_of_days_for_forward_returns_calculation}_day"])
            except Exception as e:
                continue

    with open(
            f'target_values_{ticker}_{number_of_days_for_forward_returns_calculation}_D_{forward_returns_type}_percentile_above_85.pkl',
            'wb') as file:
        pickle.dump(target_values, file)

    return


def cache_closest_confluence_areas_along_with_its_constituents_and_their_lookbacks_and_exponential_function_value_in_form_of_numpy_array_for_all_days_percentile_above_85(
        ticker, number_of_days_for_forward_returns_calculation, forward_returns_type):
    with open(f'Confluence_Points_{ticker}_along_with_its_constituents_and_their_lookbacks.pkl', 'rb') as file:
        confluence_points = pickle.load(file)

    with open(
            f'{ticker}_percentiles_based_on_all_days_{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type}_returns.pkl',
            'rb') as file:
        percentiles_for_each_day = pickle.load(file)

    days_above_85_percentile = []

    for i in percentiles_for_each_day.keys():
        if percentiles_for_each_day[i] >= 85:
            days_above_85_percentile.append(i)

    df_daily = get_data(ticker, "D")
    closest_confluence_points_with_constituents_and_lookbacks_and_function_value = {}

    for i in confluence_points.keys():
        if i in days_above_85_percentile:
            day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
            closest_confluence = list(confluence_points[i].keys())[0]
            for j in confluence_points[i].keys():
                if abs(day_close - j) < closest_confluence:
                    closest_confluence = j
            closest_confluence_points_with_constituents_and_lookbacks_and_function_value[i] = [closest_confluence,
                                                                                               confluence_points[i][
                                                                                                   closest_confluence]]

    for i in closest_confluence_points_with_constituents_and_lookbacks_and_function_value.keys():
        day_close = df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
        for j in range(len(closest_confluence_points_with_constituents_and_lookbacks_and_function_value[i][1])):
            closest_confluence_points_with_constituents_and_lookbacks_and_function_value[i][1][j].append(
                exponential_function_for_confluence_constituents(
                    closest_confluence_points_with_constituents_and_lookbacks_and_function_value[i][1][j][
                        1] - day_close))

    closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df = pd.DataFrame().from_dict(
        closest_confluence_points_with_constituents_and_lookbacks_and_function_value, orient='index',
        columns=["Closest_Confluence", "Constituents_of_Closest_Confluence"])

    lis = []
    for i in closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df.index:
        arr = lis.append(closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df.loc[i][
                             "Constituents_of_Closest_Confluence"])

    inside_lis_len = []
    for i in range(len(lis)):
        inside_lis_len.append(len(lis[i]))

    master_array = np.zeros(
        (len(closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df), max(inside_lis_len), 3))
    master_array_appender = 0

    for i in closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df[
        "Constituents_of_Closest_Confluence"]:
        innermost_list = []
        for j in i:
            innermost_list.append(j)
        for j in range(max(inside_lis_len) - len(innermost_list)):
            innermost_list.append([np.nan, np.nan, np.nan])
        arr = np.array(innermost_list)
        master_array[master_array_appender] = arr
        master_array_appender += 1

    with open(
            f'Closest_Confluence_Point_{ticker}_constituents_lb_and_exponential_function_value_in_form_of_numpy_array_for_{number_of_days_for_forward_returns_calculation}_D_{forward_returns_type}_all_days_percentile_above_85.pkl',
            'wb') as file:
        pickle.dump(master_array, file)

    with open(f'{ticker}_forward_returns.pkl', 'rb') as file:
        forward_returns = pickle.load(file)

    target = {}
    target_values = []
    for i in forward_returns.index:
        if i in closest_confluence_points_with_constituents_and_lookbacks_and_function_value_df.index:
            try:
                target[i] = forward_returns[f"{forward_returns_type}"][i][
                    f"{number_of_days_for_forward_returns_calculation}_day"]
                target_values.append(forward_returns[f"{forward_returns_type}"][i][
                                         f"{number_of_days_for_forward_returns_calculation}_day"])
            except Exception as e:
                continue

    with open(
            f'target_values_{ticker}_{number_of_days_for_forward_returns_calculation}_D_{forward_returns_type}_all_days_percentile_above_85.pkl',
            'wb') as file:
        pickle.dump(target_values, file)

    return


def cache_forward_returns_and_fisher_for_non_white_confluence_areas_and_absolute_distance_percentile_between_0_and_20_and_distance_of_confluence_between_high_low_manually_marked_as_0(
        ticker, distance_percentile_cache_path, forward_return_days, forward_return_type, fisher_lookback):
    with open(f'{distance_percentile_cache_path}', 'rb') as file:
        absolute_percentile_closest_non_white = pickle.load(file)

    with open(f'{ticker}_forward_returns.pkl', 'rb') as file:
        forward_returns = pickle.load(file)

    df_daily = get_data("NIFc1", "D")
    df_daily[f"{fisher_lookback}_day_fisher"] = fisher(df_daily, fisher_lookback)

    days_with_less_than_20_percentile_or_with_confluenc_between_low_and_high = []

    for i in list(absolute_percentile_closest_non_white.keys()):
        if absolute_percentile_closest_non_white[i]["details"]["confluence_strength"] > 3:
            if (absolute_percentile_closest_non_white[i]["min_of_abs_same_day_ohlc_data"] == -1) or (
                    absolute_percentile_closest_non_white[i]["distance_percentile_score"] >= 0 and
                    absolute_percentile_closest_non_white[i]["distance_percentile_score"] <= 20):
                days_with_less_than_20_percentile_or_with_confluenc_between_low_and_high.append(i)
        elif absolute_percentile_closest_non_white[i]["details"]["confluence_strength"] < 1:
            continue
        else:
            if (absolute_percentile_closest_non_white[i]["min_of_abs_same_day_ohlc_data"] == -1) or (
                    absolute_percentile_closest_non_white[i]["distance_percentile_score"] >= 0 and
                    absolute_percentile_closest_non_white[i]["distance_percentile_score"] <= 20):
                days_with_less_than_20_percentile_or_with_confluenc_between_low_and_high.append(i)

    returns_and_fisher_dict = {}
    for i in days_with_less_than_20_percentile_or_with_confluenc_between_low_and_high:
        returns_and_fisher_dict[i] = {}

    for i in days_with_less_than_20_percentile_or_with_confluenc_between_low_and_high:
        for j in forward_return_days:
            for k in forward_return_type:
                returns_and_fisher_dict[i][f"{j}_day_{k}_returns"] = forward_returns.loc[i][f"{k}_returns"][f"{j}_day"]
        returns_and_fisher_dict[i][f"{fisher_lookback}_day_fisher"] = df_daily[df_daily["Datetime"] == i].iloc[0][
            f"{fisher_lookback}_day_fisher"]

    returns_and_fisher_df = pd.DataFrame.from_dict(returns_and_fisher_dict, orient='index')

    returns_and_fisher_df.to_csv(
        f"{ticker}forward_returns_and_fisher_for_non_white_confluence_areas_and_absolute_distance_percentile_between_0_and_20_confluence_between_high_low_manually_marked.csv")

    with open(
            f'{ticker}forward_returns_and_fisher_for_non_white_confluence_areas_and_absolute_distance_percentile_between_0_and_20_confluence_between_high_low_manually_marked.pkl',
            'wb') as file:
        pickle.dump(returns_and_fisher_df, file)

    return


def cache_confluence_areas_along_with_matching_of_poc_and_point_of_least_volume_with_lookbacks(ticker):
    with open(f'VolumeFeatures_{ticker}_Daily_test.pkl', 'rb') as file:
        volume_levels_df = pickle.load(file)

    price_level_columns = [i for i in volume_levels_df.columns if i.startswith("PriceLevels")]
    confluence_points = {}
    for i in volume_levels_df.index:
        try:
            all_points = []
            for j in price_level_columns:
                all_points.append((j.split("_")[1], volume_levels_df[j][i]["poc"]))
                all_points.append((j.split("_")[1], volume_levels_df[j][i]["point_of_least_volume_in_va"]))
            confluence_points[i] = matching_logic_for_confluence_areas_for_each_day_return_with_lookback(all_points)
        except Exception as e:
            # print(e)
            continue

    with open(f'Confluence_Points_{ticker}_with_matching_of_poc_and_point_of_least_volume_along_with_lookbacks.pkl',
              'wb') as file:
        pickle.dump(confluence_points, file)

    return


def plot_confluence_areas_and_constance_brown_and_pivots(df_daily, ticker, cache_path, plot_path="test.jpg"):
    with open(f'{cache_path}', 'rb') as file:
        volume_levels_df = pickle.load(file)
    volume_levels_df = volume_levels_df.iloc[-number_of_days_for_confluence_level_plot:]
    df_daily = df_daily[(df_daily["Datetime"] <= max(volume_levels_df.index).strftime("%Y-%m-%d")) & (
                df_daily["Datetime"] >= min(volume_levels_df.index).strftime("%Y-%m-%d"))]
    df_constant_brown = df_daily.copy()

    with open(f'Confluence_Points_.NSEI_test.pkl', 'rb') as file:
        confluence_points = pickle.load(file)

    high_pivots, low_pivots = detect_pivots(df_daily, confluence_points)
    dates = df_daily["Datetime"]

    f, (a0, a1, a2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [4, 1, 1]})
    # ax3 = plt.subplot(2, 1, 2)
    for i in list(confluence_points.keys()):
        confluence_point = []
        confluence_strength = []
        for j in list(confluence_points[i].keys()):
            if j > 15500:
                confluence_point.append(j)
                confluence_strength.append(-confluence_points[i][j])
        a0.scatter([i] * len(confluence_point), confluence_point, c=confluence_strength, cmap="gray")
    a0.set_facecolor('pink')
    a0.plot()
    # ax2.plot(df_daily["Datetime"], df_daily["Close"])
    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]

    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(a0, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)
    a0.grid(True)
    a0.set_xlabel('Date')
    a0.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    a1_yaxis = []
    for i in dates:
        if i in high_pivots:
            a1_yaxis.append(1)
        elif i in low_pivots:
            a1_yaxis.append(-1)
        else:
            a1_yaxis.append(0)
    a1.plot(dates, a1_yaxis, label="Pivots")
    a2.plot(df_constant_brown["Datetime"], df_constant_brown["CB"], label="Constance-Brown Index")
    a2.plot(df_constant_brown["Datetime"], df_constant_brown["FMACB"], label="FMA Constance-Brown Index")
    a2.plot(df_constant_brown["Datetime"], df_constant_brown["SMACB"], label="SMA Constance-Brown Index")

    plt.legend()
    plt.savefig(f"{plot_path}", dpi=300)
    plt.close()
    return


def calculate_forward_returns(df_daily, ticker):
    forward_returns_dictonary = {}
    for i in df_daily.index:
        forward_returns_dictonary[df_daily["Datetime"][i]] = {"max_returns": {}, "min_returns": {},
                                                              "end_to_end_returns": {}}

    for i in df_daily.index:
        for j in forward_return_list_in_days:
            try:
                list_of_close_values = list(df_daily.loc[i + 1:i + j + 1]["Close"])
                forward_returns_dictonary[df_daily["Datetime"][i]]["max_returns"][f"{j}_day"] = float(
                    max(list_of_close_values) - df_daily.loc[i]["Close"]) / float(df_daily.loc[i]["Close"])
                forward_returns_dictonary[df_daily["Datetime"][i]]["min_returns"][f"{j}_day"] = float(
                    min(list_of_close_values) - df_daily.loc[i]["Close"]) / float(df_daily.loc[i]["Close"])
                forward_returns_dictonary[df_daily["Datetime"][i]]["end_to_end_returns"][f"{j}_day"] = float(
                    df_daily.loc[i + j]["Close"] - df_daily.loc[i]["Close"]) / float(df_daily.loc[i]["Close"])
            except Exception as e:
                print(e)
                continue

    forward_returns_df = pd.DataFrame.from_dict(forward_returns_dictonary, orient='index')

    with open(f'{ticker}_forward_returns.pkl', 'wb') as file:
        pickle.dump(forward_returns_df, file)

    return


def cache_forward_statistics(df_daily, ticker):
    forward_statistics_dictonary = {}
    for i in df_daily.index:
        forward_statistics_dictonary[df_daily["Datetime"][i]] = {"max_returns": {}, "min_returns": {},
                                                                 "end_to_end_returns": {}, "end_to_end_sharpe": {}}

    for i in tqdm(df_daily.index):
        for j in forward_return_list_in_days:
            try:
                list_of_close_values = list(df_daily.loc[i + 1:i + j + 1]["Close"])
                forward_statistics_dictonary[df_daily["Datetime"][i]]["max_returns"][f"{j}_day"] = float(
                    max(list_of_close_values) - df_daily.loc[i]["Close"]) / float(df_daily.loc[i]["Close"])
                forward_statistics_dictonary[df_daily["Datetime"][i]]["min_returns"][f"{j}_day"] = float(
                    min(list_of_close_values) - df_daily.loc[i]["Close"]) / float(df_daily.loc[i]["Close"])
                forward_statistics_dictonary[df_daily["Datetime"][i]]["end_to_end_returns"][f"{j}_day"] = float(
                    df_daily.loc[i + j]["Close"] - df_daily.loc[i]["Close"]) / float(df_daily.loc[i]["Close"])
            except Exception as e:
                print(e)
                continue

    for i in tqdm(df_daily.index):
        try:
            forward_statistics_dictonary[df_daily["Datetime"][i]]["daily_returns"] = float(
                df_daily.loc[i + 1]["Close"] - df_daily.loc[i]["Close"]) / float(df_daily.loc[i]["Close"])
        except Exception as e:
            print(e)
            continue

    forward_statistics_df = pd.DataFrame.from_dict(forward_statistics_dictonary, orient='index')

    for i in tqdm(range(len(df_daily.index))):
        for j in forward_return_list_in_days:
            try:
                forward_sharpe = np.nan
                if i + j + 1 < len(df_daily.index):
                    forward_sharpe = (forward_statistics_df.iloc[i:i + j + 1][
                                          "daily_returns"].mean() - risk_free_interest / 25200) / \
                                     forward_statistics_df.iloc[i:i + j + 1]["daily_returns"].std() * (252 ** .5)
                if np.isnan(forward_sharpe) == False:
                    forward_statistics_df.iloc[i]["end_to_end_sharpe"][f"{j}_day"] = forward_sharpe
            except Exception as e:
                print(e)
                continue

    with open(f'{ticker}_forward_statistics.pkl', 'wb') as file:
        pickle.dump(forward_statistics_df, file)


def plot_confluence_areas_distance_percentile_and_returns_percentile(df_daily, ticker, distance_percentile_cache_path,
                                                                     confluence_points_cache,
                                                                     number_of_days_for_forward_returns_calculation,
                                                                     forward_returns_type):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    with open(
            f'{ticker}_percentiles_for_each_day_based_on_{number_of_days_for_forward_returns_calculation}_day_absolute_{forward_returns_type}_returns.pkl',
            'rb') as file:
        returns_percentile = pickle.load(file)

    with open(f'{distance_percentile_cache_path}', 'rb') as file:
        distance_percentile = pickle.load(file)

    days_to_plot = {}
    for i in confluence_points.keys():
        if i in distance_percentile.keys() and i.year == 2020 and i.month <= 4:
            days_to_plot[i] = confluence_points[i]
    df_daily = df_daily[(df_daily["Datetime"] <= max(days_to_plot.keys()).strftime("%Y-%m-%d")) & (
                df_daily["Datetime"] >= min(days_to_plot.keys()).strftime("%Y-%m-%d"))]

    f, (a0, a1, a2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [4, 1, 1]})
    # ax3 = plt.subplot(2, 1, 2)
    for i in list(days_to_plot.keys()):
        confluence_point = []
        confluence_strength = []
        for j in list(days_to_plot[i].keys()):
            confluence_point.append(j)
            if days_to_plot[i][j]["confluence_strength"] > 3:
                confluence_strength.append(-300)
            elif days_to_plot[i][j]["confluence_strength"] < 1:
                confluence_strength.append(300)
            else:
                confluence_strength.append(-days_to_plot[i][j]["confluence_strength"])
        a0.scatter([i] * len(confluence_point), confluence_point, c=confluence_strength, cmap="gray")

    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]
    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(a0, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)
    a0.grid(True)
    a0.set_xlabel('Date')
    a0.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    a0.set_facecolor('pink')
    a0.plot()
    a1_yaxis = []
    a2_yaxis = []
    for i in days_to_plot.keys():
        a1_yaxis.append(distance_percentile[i]["distance_percentile_score"])
        a2_yaxis.append(returns_percentile[i])
    a1.plot(days_to_plot.keys(), a1_yaxis, label="closest_confluence_distance_percentile")
    a1.legend()
    a2.plot(days_to_plot.keys(), a2_yaxis)
    a2.set_xlabel(
        f"{number_of_days_for_forward_returns_calculation}_day_absolute_{forward_returns_type}_returns_percentile")
    plt.legend()
    plt.savefig(
        f"confluence_points_with_distance_percentile_and_{number_of_days_for_forward_returns_calculation}_day_absolute_{forward_returns_type}_returns_percentile.jpg")
    plt.close()

    return


def plot_confluence_areas_distance_percentile_and_returns_percentile_scatter_plot(df_daily, ticker,
                                                                                  distance_percentile_cache_path,
                                                                                  confluence_points_cache,
                                                                                  number_of_days_for_forward_returns_calculation,
                                                                                  forward_returns_type):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    with open(
            f'{ticker}_percentiles_for_each_day_based_on_{number_of_days_for_forward_returns_calculation}_day_absolute_{forward_returns_type}_returns.pkl',
            'rb') as file:
        returns_percentile = pickle.load(file)

    with open(f'{distance_percentile_cache_path}', 'rb') as file:
        distance_percentile = pickle.load(file)

    days_to_plot = {}
    for i in confluence_points.keys():
        if i in distance_percentile.keys() and i.year >= 2021:
            days_to_plot[i] = confluence_points[i]
    df_daily = df_daily[(df_daily["Datetime"] <= max(days_to_plot.keys()).strftime("%Y-%m-%d")) & (
                df_daily["Datetime"] >= min(days_to_plot.keys()).strftime("%Y-%m-%d"))]

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
    # ax3 = plt.subplot(2, 1, 2)
    for i in list(days_to_plot.keys()):
        confluence_point = []
        confluence_strength = []
        for j in list(days_to_plot[i].keys()):
            confluence_point.append(j)
            if days_to_plot[i][j]["confluence_strength"] > 3:
                confluence_strength.append(-300)
            elif days_to_plot[i][j]["confluence_strength"] < 1:
                confluence_strength.append(300)
            else:
                confluence_strength.append(-days_to_plot[i][j]["confluence_strength"])
        a0.scatter([i] * len(confluence_point), confluence_point, c=confluence_strength, cmap="gray")

    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]
    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(a0, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)
    a0.grid(True)
    a0.set_xlabel('Date')
    a0.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    a0.set_facecolor('pink')
    a0.plot()
    a1_xaxis = []
    a1_yaxis = []
    for i in days_to_plot.keys():
        if i in returns_percentile.keys():
            a1_xaxis.append(returns_percentile[i])
            a1_yaxis.append(distance_percentile[i]["distance_percentile_score"])
    a1.scatter(a1_xaxis, a1_yaxis)
    a1.set_xlabel(
        f"{number_of_days_for_forward_returns_calculation}_day_absolute_{forward_returns_type}_returns_percentile")
    a1.set_ylabel("closest_confluence_distance_percentile")
    plt.legend()
    plt.savefig(
        f"confluence_points_with_distance_percentile_and_{number_of_days_for_forward_returns_calculation}_day_absolute_{forward_returns_type}_returns_percentile.jpg")
    plt.close()

    return


def plot_non_white_confluence_areas_and_distance_percentile_for_closest_non_white_confluence_and_returns_percentile_scatter_plot(
        df_daily, ticker, distance_percentile_cache_path, confluence_points_cache,
        number_of_days_for_forward_returns_calculation, forward_returns_type):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    with open(
            f'{ticker}_percentiles_for_each_day_based_on_{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type}_returns.pkl',
            'rb') as file:
        returns_percentile = pickle.load(file)

    with open(f'{distance_percentile_cache_path}', 'rb') as file:
        distance_percentile = pickle.load(file)

    with open(f'{ticker}_percentiles_of_distance_closest_non_white_confluence_point_for_each_day.pkl', 'rb') as file:
        absolute_percentile_closest_non_white = pickle.load(file)

    days_to_plot = {}
    for i in confluence_points.keys():
        if i in distance_percentile.keys() and i.year >= 2021:
            days_to_plot[i] = confluence_points[i]
    df_daily = df_daily[(df_daily["Datetime"] <= max(days_to_plot.keys()).strftime("%Y-%m-%d")) & (
                df_daily["Datetime"] >= min(days_to_plot.keys()).strftime("%Y-%m-%d"))]

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
    # ax3 = plt.subplot(2, 1, 2)
    for i in list(days_to_plot.keys()):
        # confluence_point = []
        # confluence_strength = []
        for j in list(days_to_plot[i].keys()):
            # confluence_point.append(j)
            if days_to_plot[i][j]["confluence_strength"] > 3:
                a0.scatter(i, j, color="black")
            elif days_to_plot[i][j]["confluence_strength"] < 1:
                continue
            else:
                a0.scatter(i, j, color="gray")
        # a0.scatter([i] * len(confluence_point), confluence_point, c=confluence_strength, cmap="gray")

    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]
    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(a0, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)
    a0.grid(True)
    a0.set_xlabel('Date')
    a0.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    a0.set_facecolor('pink')
    a0.plot()
    a1_xaxis = []
    a1_yaxis = []
    for i in days_to_plot.keys():
        if i in returns_percentile.keys():
            a1_xaxis.append(returns_percentile[i])
            a1_yaxis.append(distance_percentile[i]["distance_percentile_score"])
    a1.scatter(a1_xaxis, a1_yaxis)
    a1.set_xlabel(f"{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type}_returns_percentile")
    a1.set_ylabel("closest_non_white_confluence_non_absolute_distance_percentile")
    plt.title(
        f"non_white_confluence_areas_with_non_absolute_distance_percentile_and_{number_of_days_for_forward_returns_calculation}_day_absolute_{forward_returns_type}_returns_percentile",
        fontdict={'fontsize': 20})
    plt.legend()
    plt.savefig(
        f"non_white_confluence_areas_with_non_absolute_distance_percentile_and_{number_of_days_for_forward_returns_calculation}_day_absolute_{forward_returns_type}_returns_percentile.jpg",
        dpi=300)
    plt.close()

    return


def plot_non_white_confluence_areas_and_mark_points_absolute_distance_percentile_between_0_and_30(df_daily, ticker,
                                                                                                  distance_percentile_cache_path,
                                                                                                  confluence_points_cache):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    with open(f'{distance_percentile_cache_path}', 'rb') as file:
        absolute_percentile_closest_non_white = pickle.load(file)

    days_to_plot = {}
    for i in confluence_points.keys():
        if i in absolute_percentile_closest_non_white.keys() and i.year >= 2021:
            days_to_plot[i] = confluence_points[i]
    df_daily = df_daily[(df_daily["Datetime"] <= max(days_to_plot.keys()).strftime("%Y-%m-%d")) & (
                df_daily["Datetime"] >= min(days_to_plot.keys()).strftime("%Y-%m-%d"))]

    f, (a0) = plt.subplots(1, 1, )
    # ax3 = plt.subplot(2, 1, 2)
    for i in list(days_to_plot.keys()):
        # confluence_point = []
        # confluence_strength = []
        for j in list(days_to_plot[i].keys()):
            # confluence_point.append(j)
            if days_to_plot[i][j]["confluence_strength"] > 3:
                if j == absolute_percentile_closest_non_white[i]["closest_confluence"] and \
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] >= 0 and \
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] <= 30:
                    a0.scatter(i, j, s=10, color="orange")
                    a0.axvline(x=i, color="orange")
                else:
                    a0.scatter(i, j, s=10, color="black")
            elif days_to_plot[i][j]["confluence_strength"] < 1:
                continue
            else:
                if j == absolute_percentile_closest_non_white[i]["closest_confluence"] and \
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] >= 0 and \
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] <= 30:
                    a0.scatter(i, j, s=10, color="orange")
                    a0.axvline(x=i, color="orange")
                else:
                    a0.scatter(i, j, s=10, color="gray")
        # a0.scatter([i] * len(confluence_point), confluence_point, c=confluence_strength, cmap="gray")

    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]
    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(a0, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)
    a0.grid(True)
    a0.set_xlabel('Date')
    a0.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    # a0.set_facecolor('pink')
    a0.plot()
    # a1_xaxis = []
    # a1_yaxis = []
    # percentile_0_to_30_xaxis = []
    # percentile_0_to_30_yaxis = []
    # for i in days_to_plot.keys():
    #     if i in returns_percentile.keys():
    #         if absolute_percentile_closest_non_white[i]["distance_percentile_score"] >= 0 and absolute_percentile_closest_non_white[i]["distance_percentile_score"] <= 30:
    #             percentile_0_to_30_xaxis.append(returns_percentile[i])
    #             percentile_0_to_30_yaxis.append(distance_percentile[i]["distance_percentile_score"])
    #         else:
    #             a1_xaxis.append(returns_percentile[i])
    #             a1_yaxis.append(distance_percentile[i]["distance_percentile_score"])
    # a1.scatter(a1_xaxis,a1_yaxis)
    # a1.scatter(percentile_0_to_30_xaxis,percentile_0_to_30_yaxis,c="coral")
    # a1.set_xlabel(f"{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type}_returns_percentile")
    # a1.set_ylabel("closest_non_white_confluence_non_absolute_distance_percentile")
    plt.title(
        f"{ticker}_non_white_confluence_areas_and_min_of_same_day_ohlc_absolute_distance_percentile_between_0_to_30_marked_orange",
        fontdict={'fontsize': 15})
    plt.legend()
    plt.savefig(
        f"{ticker}_non_white_confluence_areas_and_min_of_same_day_ohlc_absolute_distance_percentile_between_0_to_30_marked_orange.jpg",
        dpi=300)
    plt.close()

    return


def plot_non_white_confluence_areas_and_mark_points_absolute_distance_percentile_between_0_and_30_for_fisher_pivots(
        df_daily, ticker, distance_percentile_cache_path, confluence_points_cache, fisher_lookback):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    with open(f'{distance_percentile_cache_path}', 'rb') as file:
        absolute_percentile_closest_non_white = pickle.load(file)

    with open(f'{ticker}_fisher_{fisher_lookback}_lb_low_pivots.pkl', 'rb') as file:
        fisher_low_pivots = pickle.load(file)

    days_to_plot = {}
    for i in confluence_points.keys():
        if i in absolute_percentile_closest_non_white.keys() and i.year >= 2021:
            days_to_plot[i] = confluence_points[i]
    df_daily = df_daily[(df_daily["Datetime"] <= max(days_to_plot.keys()).strftime("%Y-%m-%d")) & (
                df_daily["Datetime"] >= min(days_to_plot.keys()).strftime("%Y-%m-%d"))]

    f, (a0) = plt.subplots(1, 1, )
    # ax3 = plt.subplot(2, 1, 2)
    for i in list(days_to_plot.keys()):
        # confluence_point = []
        # confluence_strength = []
        for j in list(days_to_plot[i].keys()):
            # confluence_point.append(j)
            if days_to_plot[i][j]["confluence_strength"] > 3:
                if j == absolute_percentile_closest_non_white[i]["closest_confluence"] and \
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] >= 0 and \
                        absolute_percentile_closest_non_white[i][
                            "distance_percentile_score"] <= 30 and i in fisher_low_pivots:
                    a0.scatter(i, j, s=10, color="orange")
                    a0.axvline(x=i, color="orange")
                else:
                    a0.scatter(i, j, s=10, color="black")
            elif days_to_plot[i][j]["confluence_strength"] < 1:
                continue
            else:
                if j == absolute_percentile_closest_non_white[i]["closest_confluence"] and \
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] >= 0 and \
                        absolute_percentile_closest_non_white[i][
                            "distance_percentile_score"] <= 30 and i in fisher_low_pivots:
                    a0.scatter(i, j, s=10, color="orange")
                    a0.axvline(x=i, color="orange")
                else:
                    a0.scatter(i, j, s=10, color="gray")
        # a0.scatter([i] * len(confluence_point), confluence_point, c=confluence_strength, cmap="gray")

    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]
    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(a0, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)
    a0.grid(True)
    a0.set_xlabel('Date')
    a0.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    # a0.set_facecolor('pink')
    a0.plot()
    # a1_xaxis = []
    # a1_yaxis = []
    # percentile_0_to_30_xaxis = []
    # percentile_0_to_30_yaxis = []
    # for i in days_to_plot.keys():
    #     if i in returns_percentile.keys():
    #         if absolute_percentile_closest_non_white[i]["distance_percentile_score"] >= 0 and absolute_percentile_closest_non_white[i]["distance_percentile_score"] <= 30:
    #             percentile_0_to_30_xaxis.append(returns_percentile[i])
    #             percentile_0_to_30_yaxis.append(distance_percentile[i]["distance_percentile_score"])
    #         else:
    #             a1_xaxis.append(returns_percentile[i])
    #             a1_yaxis.append(distance_percentile[i]["distance_percentile_score"])
    # a1.scatter(a1_xaxis,a1_yaxis)
    # a1.scatter(percentile_0_to_30_xaxis,percentile_0_to_30_yaxis,c="coral")
    # a1.set_xlabel(f"{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type}_returns_percentile")
    # a1.set_ylabel("closest_non_white_confluence_non_absolute_distance_percentile")
    plt.title(
        f"{ticker}_non_white_confluence_areas_and_min_of_same_day_ohlc_absolute_distance_percentile_between_0_to_30_and_fisher_{fisher_lookback}_lb_low_pivots_marked_orange",
        fontdict={'fontsize': 15})
    plt.legend()
    plt.savefig(
        f"{ticker}_non_white_confluence_areas_and_min_of_same_day_ohlc_absolute_distance_percentile_between_0_to_30_and_fisher_{fisher_lookback}_lb_low_pivots_marked_orange.jpg",
        dpi=300)
    plt.close()

    return


def plot_non_white_confluence_areas_and_mark_points_absolute_distance_percentile_between_0_and_30_and_distance_manually_marked_as_0(
        df_daily, ticker, distance_percentile_cache_path, confluence_points_cache):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    with open(f'{distance_percentile_cache_path}', 'rb') as file:
        absolute_percentile_closest_non_white = pickle.load(file)

    days_to_plot = {}
    for i in confluence_points.keys():
        if i in absolute_percentile_closest_non_white.keys() and i.year >= 2021:
            days_to_plot[i] = confluence_points[i]
    df_daily = df_daily[(df_daily["Datetime"] <= max(days_to_plot.keys()).strftime("%Y-%m-%d")) & (
                df_daily["Datetime"] >= min(days_to_plot.keys()).strftime("%Y-%m-%d"))]

    f, (a0) = plt.subplots(1, 1, )
    for i in list(days_to_plot.keys()):
        for j in list(days_to_plot[i].keys()):
            if days_to_plot[i][j]["confluence_strength"] > 3:
                if (absolute_percentile_closest_non_white[i]["min_of_abs_same_day_ohlc_data"] == -1 and j ==
                    absolute_percentile_closest_non_white[i]["closest_confluence"]) or (
                        j == absolute_percentile_closest_non_white[i]["closest_confluence"] and
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] >= 0 and
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] <= 30):
                    a0.scatter(i, j, s=10, color="orange")
                    a0.axvline(x=i, color="orange")
                else:
                    a0.scatter(i, j, s=10, color="black")
            elif days_to_plot[i][j]["confluence_strength"] < 1:
                continue
            else:
                if (absolute_percentile_closest_non_white[i]["min_of_abs_same_day_ohlc_data"] == -1 and j ==
                    absolute_percentile_closest_non_white[i]["closest_confluence"]) or (
                        j == absolute_percentile_closest_non_white[i]["closest_confluence"] and
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] >= 0 and
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] <= 30):
                    a0.scatter(i, j, s=10, color="orange")
                    a0.axvline(x=i, color="orange")
                else:
                    a0.scatter(i, j, s=10, color="gray")
        # a0.scatter([i] * len(confluence_point), confluence_point, c=confluence_strength, cmap="gray")

    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]
    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(a0, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)
    a0.grid(True)
    a0.set_xlabel('Date')
    a0.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    # a0.set_facecolor('pink')
    a0.plot()
    # a1_xaxis = []
    # a1_yaxis = []
    # percentile_0_to_30_xaxis = []
    # percentile_0_to_30_yaxis = []
    # for i in days_to_plot.keys():
    #     if i in returns_percentile.keys():
    #         if absolute_percentile_closest_non_white[i]["distance_percentile_score"] >= 0 and absolute_percentile_closest_non_white[i]["distance_percentile_score"] <= 30:
    #             percentile_0_to_30_xaxis.append(returns_percentile[i])
    #             percentile_0_to_30_yaxis.append(distance_percentile[i]["distance_percentile_score"])
    #         else:
    #             a1_xaxis.append(returns_percentile[i])
    #             a1_yaxis.append(distance_percentile[i]["distance_percentile_score"])
    # a1.scatter(a1_xaxis,a1_yaxis)
    # a1.scatter(percentile_0_to_30_xaxis,percentile_0_to_30_yaxis,c="coral")
    # a1.set_xlabel(f"{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type}_returns_percentile")
    # a1.set_ylabel("closest_non_white_confluence_non_absolute_distance_percentile")
    plt.title(
        f"{ticker}_non_white_confluence_areas_and_mark_points_absolute_distance_percentile_between_0_and_10_and_distance_manually_marked_as_0_marked_orange",
        fontdict={'fontsize': 15})
    plt.legend()
    plt.savefig(
        f"{ticker}_non_white_confluence_areas_and_mark_points_absolute_distance_percentile_between_0_and_10_and_distance_manually_marked_as_0_marked_orange.jpg",
        dpi=300)
    plt.close()

    return


def plot_non_white_confluence_areas_and_mark_points_absolute_distance_percentile_between_0_and_20_and_distance_manually_marked_as_0_and_fisher_score_subplot(
        df_daily, ticker, distance_percentile_cache_path, confluence_points_cache, fisher_lookback):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    with open(f'{distance_percentile_cache_path}', 'rb') as file:
        absolute_percentile_closest_non_white = pickle.load(file)

    days_to_plot = {}
    df_daily[f"{fisher_lookback}_day_fisher"] = fisher(df_daily, fisher_lookback)
    df_daily_fisher = df_daily
    for i in confluence_points.keys():
        if i in absolute_percentile_closest_non_white.keys() and i.year >= 2021:
            days_to_plot[i] = confluence_points[i]
    df_daily = df_daily[(df_daily["Datetime"] <= max(days_to_plot.keys()).strftime("%Y-%m-%d")) & (
                df_daily["Datetime"] >= min(days_to_plot.keys()).strftime("%Y-%m-%d"))]

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    for i in list(days_to_plot.keys()):
        for j in list(days_to_plot[i].keys()):
            if days_to_plot[i][j]["confluence_strength"] > 3:
                if (absolute_percentile_closest_non_white[i]["min_of_abs_same_day_ohlc_data"] == -1 and j ==
                    absolute_percentile_closest_non_white[i]["closest_confluence"]) or (
                        j == absolute_percentile_closest_non_white[i]["closest_confluence"] and
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] >= 0 and
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] <= 20):
                    a0.scatter(i, j, s=10, color="orange")
                    a0.axvline(x=i, color="orange")
                else:
                    a0.scatter(i, j, s=10, color="black")
            elif days_to_plot[i][j]["confluence_strength"] < 1:
                continue
            else:
                if (absolute_percentile_closest_non_white[i]["min_of_abs_same_day_ohlc_data"] == -1 and j ==
                    absolute_percentile_closest_non_white[i]["closest_confluence"]) or (
                        j == absolute_percentile_closest_non_white[i]["closest_confluence"] and
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] >= 0 and
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] <= 20):
                    a0.scatter(i, j, s=10, color="orange")
                    a0.axvline(x=i, color="orange")
                else:
                    a0.scatter(i, j, s=10, color="gray")
        # a0.scatter([i] * len(confluence_point), confluence_point, c=confluence_strength, cmap="gray")

    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]
    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(a0, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)
    a0.grid(True)
    a0.set_xlabel('Date')
    a0.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    # a0.set_facecolor('pink')
    a0.plot()
    a1_xaxis = []
    a1_yaxis = []
    for i in days_to_plot.keys():
        a1_xaxis.append(i)
        a1_yaxis.append(df_daily_fisher[df_daily_fisher['Datetime'] == i].iloc[0][f"{fisher_lookback}_day_fisher"])
    a1.plot(a1_xaxis, a1_yaxis)
    plt.title(
        f"{ticker}_non_white_confluence_areas_and_mark_points_absolute_distance_percentile_between_0_and_20_and_distance_manually_marked_as_0_marked_orange_fisher_{fisher_lookback}",
        fontdict={'fontsize': 15})
    plt.legend()
    plt.savefig(
        f"{ticker}_non_white_confluence_areas_and_mark_points_absolute_distance_percentile_between_0_and_20_and_distance_manually_marked_as_0_marked_orange_fisher_{fisher_lookback}.jpg",
        dpi=300)
    plt.close()

    return


def plot_non_white_confluence_areas_and_mark_points_absolute_distance_percentile_between_0_and_20_and_distance_manually_marked_as_0_and_constance_brown_subplot(
        df_daily, ticker, distance_percentile_cache_path, confluence_points_cache):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    with open(f'{distance_percentile_cache_path}', 'rb') as file:
        absolute_percentile_closest_non_white = pickle.load(file)

    days_to_plot = {}
    constance_brown_df = constance_brown(df_daily)
    for i in confluence_points.keys():
        if i in absolute_percentile_closest_non_white.keys() and i.year >= 2021:
            days_to_plot[i] = confluence_points[i]
    df_daily = df_daily[(df_daily["Datetime"] <= max(days_to_plot.keys()).strftime("%Y-%m-%d")) & (
                df_daily["Datetime"] >= min(days_to_plot.keys()).strftime("%Y-%m-%d"))]

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    for i in list(days_to_plot.keys()):
        for j in list(days_to_plot[i].keys()):
            if days_to_plot[i][j]["confluence_strength"] > 3:
                if (absolute_percentile_closest_non_white[i]["min_of_abs_same_day_ohlc_data"] == -1 and j ==
                    absolute_percentile_closest_non_white[i]["closest_confluence"]) or (
                        j == absolute_percentile_closest_non_white[i]["closest_confluence"] and
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] >= 0 and
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] <= 20):
                    a0.scatter(i, j, s=10, color="orange")
                    a0.axvline(x=i, color="orange")
                else:
                    a0.scatter(i, j, s=10, color="black")
            elif days_to_plot[i][j]["confluence_strength"] < 1:
                continue
            else:
                if (absolute_percentile_closest_non_white[i]["min_of_abs_same_day_ohlc_data"] == -1 and j ==
                    absolute_percentile_closest_non_white[i]["closest_confluence"]) or (
                        j == absolute_percentile_closest_non_white[i]["closest_confluence"] and
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] >= 0 and
                        absolute_percentile_closest_non_white[i]["distance_percentile_score"] <= 20):
                    a0.scatter(i, j, s=10, color="orange")
                    a0.axvline(x=i, color="orange")
                else:
                    a0.scatter(i, j, s=10, color="gray")
        # a0.scatter([i] * len(confluence_point), confluence_point, c=confluence_strength, cmap="gray")

    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]
    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(a0, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)
    a0.grid(True)
    a0.set_xlabel('Date')
    a0.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    # a0.set_facecolor('pink')
    a0.plot()
    a1_xaxis = []
    a1_yaxis_cb = []
    a1_yaxis_fmacb = []
    a1_yaxis_smacb = []
    for i in days_to_plot.keys():
        a1_xaxis.append(i)
        a1_yaxis_cb.append(constance_brown_df[constance_brown_df['Datetime'] == i].iloc[0]["CB"])
        a1_yaxis_fmacb.append(constance_brown_df[constance_brown_df['Datetime'] == i].iloc[0]["FMACB"])
        a1_yaxis_smacb.append(constance_brown_df[constance_brown_df['Datetime'] == i].iloc[0]["SMACB"])
    a1.plot(a1_xaxis, a1_yaxis_cb, label="Constance-Brown Index")
    a1.plot(a1_xaxis, a1_yaxis_fmacb, label="FMA Constance-Brown Index")
    a1.plot(a1_xaxis, a1_yaxis_smacb, label="SMA Constance-Brown Index")
    plt.title(
        f"{ticker}_non_white_confluence_areas_and_mark_points_absolute_distance_percentile_between_0_and_20_and_distance_manually_marked_as_0_marked_orange_and_constance_brown",
        fontdict={'fontsize': 15})
    plt.legend()
    plt.savefig(
        f"{ticker}_non_white_confluence_areas_and_mark_points_absolute_distance_percentile_between_0_and_20_and_distance_manually_marked_as_0_marked_orange_and_CB.jpg",
        dpi=300)
    plt.close()

    return


def plot_gmm_and_kmeans_scatter_plot_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks(
        df_daily, ticker, percentage_distance_cache, number_of_clusters, train_start_year=None, train_end_year=None):
    with open(f'{percentage_distance_cache}', 'rb') as file:
        distance_percentages_df = pickle.load(file)

    distance_percentages_df = distance_percentages_df[(distance_percentages_df.index.year >= train_start_year) & (
                distance_percentages_df.index.year <= train_end_year)]
    df_daily = df_daily[(df_daily["Datetime"] >= (str(train_start_year) + "-02-01")) & (
                df_daily["Datetime"] <= (str(train_end_year) + "-12-31"))]

    gmm = GaussianMixture(n_components=number_of_clusters)
    gmm.fit(distance_percentages_df)
    identified_clusters = gmm.predict(distance_percentages_df)
    distance_percentages_df_with_clusters = distance_percentages_df.copy()
    distance_percentages_df_with_clusters['Clusters'] = identified_clusters
    distance_percentages_df_with_clusters["avg_across_lookbacks"] = distance_percentages_df_with_clusters.sum(
        axis=1) / 11

    f, (a0) = plt.subplots(1, 1)
    for i in distance_percentages_df_with_clusters.index:
        # if i.year == train_end_year:
        a0.scatter(x=i, y=distance_percentages_df_with_clusters.loc[i]["avg_across_lookbacks"],
                   c=gmm_cluster_colors[int(distance_percentages_df_with_clusters.loc[i]["Clusters"])])

    # a0.grid(True)
    a0.set_xlabel('date')
    a0.set_ylabel('avg_across_lookbacks')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    a0.plot()
    plt.title(
        f"{ticker}_gmm_clusters_scatter_plot_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters",
        fontdict={'fontsize': 15})
    plt.savefig(
        f"{ticker}_gmm_clustering_scatter_plot_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters.jpg",
        dpi=300)
    plt.close()

    with open(f'{percentage_distance_cache}', 'rb') as file:
        distance_percentages_df = pickle.load(file)
    kmeans = KMeans(number_of_clusters)
    kmeans.fit(distance_percentages_df)

    identified_clusters = kmeans.fit_predict(distance_percentages_df)
    distance_percentages_df_with_clusters = distance_percentages_df.copy()
    distance_percentages_df_with_clusters['Clusters'] = identified_clusters
    distance_percentages_df_with_clusters["avg_across_lookbacks"] = distance_percentages_df_with_clusters.sum(
        axis=1) / 11

    f, (a0) = plt.subplots(1, 1)
    for i in distance_percentages_df_with_clusters.index:
        # if i.year == train_end_year:
        a0.scatter(x=i, y=distance_percentages_df_with_clusters.loc[i]["avg_across_lookbacks"],
                   c=gmm_cluster_colors[int(distance_percentages_df_with_clusters.loc[i]["Clusters"])])

    # a0.grid(True)
    a0.set_xlabel('date')
    a0.set_ylabel('avg_across_lookbacks')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    a0.plot()
    plt.title(
        f"{ticker}_kmeans_clusters_scatter_plot_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters",
        fontdict={'fontsize': 15})
    plt.savefig(
        f"{ticker}_kmeans_clustering_scatter_plot_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters.jpg",
        dpi=300)
    plt.close()

    return


def plot_gmm_and_kmeans_scatter_plot_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_and_indicator_values_various_lookbacks(
        df_daily, ticker, percentage_distance_cache, number_of_clusters, train_start_year=None, train_end_year=None):
    with open(f'{percentage_distance_cache}', 'rb') as file:
        distance_percentages_df = pickle.load(file)

    distance_percentages_df = distance_percentages_df[(distance_percentages_df.index.year >= train_start_year) & (
                distance_percentages_df.index.year <= train_end_year)]
    df_daily = df_daily[(df_daily["Datetime"] >= (str(train_start_year) + "-02-01")) & (
                df_daily["Datetime"] <= (str(train_end_year) + "-12-31"))]

    gmm = GaussianMixture(n_components=number_of_clusters)
    gmm.fit(distance_percentages_df)
    identified_clusters = gmm.predict(distance_percentages_df)
    distance_percentages_df_with_clusters = distance_percentages_df.copy()
    distance_percentages_df_with_clusters['Clusters'] = identified_clusters
    distance_percentages_df_with_clusters["avg_across_columns"] = distance_percentages_df_with_clusters.sum(
        axis=1) / len(distance_percentages_df.columns)

    f, (a0) = plt.subplots(1, 1)
    for i in distance_percentages_df_with_clusters.index:
        # if i.year == train_end_year:
        a0.scatter(x=i, y=distance_percentages_df_with_clusters.loc[i]["avg_across_columns"],
                   c=gmm_cluster_colors[int(distance_percentages_df_with_clusters.loc[i]["Clusters"])])

    # a0.grid(True)
    a0.set_xlabel('date')
    a0.set_ylabel('avg_across_columns')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    a0.plot()
    plt.title(
        f"{ticker}_gmm_clusters_scatter_plot_for_non_absolute_percentage_of_poc_and_ohlc_avg_and_indicators_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters",
        fontdict={'fontsize': 15})
    plt.savefig(
        f"{ticker}_gmm_clustering_scatter_plot_for_non_absolute_percentage_of_poc_and_ohlc_avg_and_indicators_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters.jpg",
        dpi=300)
    plt.close()

    with open(f'{percentage_distance_cache}', 'rb') as file:
        distance_percentages_df = pickle.load(file)
    kmeans = KMeans(number_of_clusters)
    kmeans.fit(distance_percentages_df)

    identified_clusters = kmeans.fit_predict(distance_percentages_df)
    distance_percentages_df_with_clusters = distance_percentages_df.copy()
    distance_percentages_df_with_clusters['Clusters'] = identified_clusters
    distance_percentages_df_with_clusters["avg_across_columns"] = distance_percentages_df_with_clusters.sum(
        axis=1) / 11

    f, (a0) = plt.subplots(1, 1)
    for i in distance_percentages_df_with_clusters.index:
        # if i.year == train_end_year:
        a0.scatter(x=i, y=distance_percentages_df_with_clusters.loc[i]["avg_across_columns"],
                   c=gmm_cluster_colors[int(distance_percentages_df_with_clusters.loc[i]["Clusters"])])

    # a0.grid(True)
    a0.set_xlabel('date')
    a0.set_ylabel('avg_across_columns')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    a0.plot()
    plt.title(
        f"{ticker}_kmeans_clusters_scatter_plot_for_non_absolute_percentage_of_poc_and_ohlc_avg_and_indicators_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters",
        fontdict={'fontsize': 15})
    plt.savefig(
        f"{ticker}_kmeans_clustering_scatter_plot_for_non_absolute_percentage_of_poc_and_ohlc_avg_and_indicators_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters.jpg",
        dpi=300)
    plt.close()

    return


def plot_gmm_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks(df_daily, ticker,
                                                                                                 percentage_distance_cache,
                                                                                                 number_of_clusters,
                                                                                                 train_start_year=None,
                                                                                                 train_end_year=None):
    with open(f'{percentage_distance_cache}', 'rb') as file:
        distance_percentages_df = pickle.load(file)

    distance_percentages_df = distance_percentages_df[(distance_percentages_df.index.year >= train_start_year) & (
                distance_percentages_df.index.year <= train_end_year)]
    df_daily = df_daily[(df_daily["Datetime"] >= (str(train_start_year) + "-02-01")) & (
                df_daily["Datetime"] <= (str(train_end_year) + "-12-31"))]

    gmm = GaussianMixture(n_components=number_of_clusters)
    gmm.fit(distance_percentages_df)
    identified_clusters = gmm.predict(distance_percentages_df)
    distance_percentages_df_with_clusters = distance_percentages_df.copy()
    distance_percentages_df_with_clusters['Clusters'] = identified_clusters

    f, (a0) = plt.subplots(1, 1)
    for i in distance_percentages_df_with_clusters.index:
        # if i.year == train_end_year:
        a0.axvline(x=i, c=gmm_cluster_colors[int(distance_percentages_df_with_clusters.loc[i]["Clusters"])])

    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]
    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(a0, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)
    a0.grid(True)
    a0.set_xlabel('Date')
    a0.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    a0.plot()
    plt.title(
        f"{ticker}_gmm_clusters_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters",
        fontdict={'fontsize': 15})
    plt.savefig(
        f"{ticker}_gmm_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters.jpg",
        dpi=300)
    plt.close()
    return


def plot_kmeans_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks(df_daily, ticker,
                                                                                                    percentage_distance_cache,
                                                                                                    number_of_clusters,
                                                                                                    train_start_year=None,
                                                                                                    train_end_year=None):
    with open(f'{percentage_distance_cache}', 'rb') as file:
        distance_percentages_df = pickle.load(file)

    distance_percentages_df = distance_percentages_df[(distance_percentages_df.index.year >= train_start_year) & (
                distance_percentages_df.index.year <= train_end_year)]
    df_daily = df_daily[(df_daily["Datetime"] >= (str(train_start_year) + "-02-01")) & (
                df_daily["Datetime"] <= (str(train_end_year) + "-12-31"))]

    kmeans = KMeans(number_of_clusters)
    kmeans.fit(distance_percentages_df)

    identified_clusters = kmeans.fit_predict(distance_percentages_df)
    distance_percentages_df_with_clusters = distance_percentages_df.copy()
    distance_percentages_df_with_clusters['Clusters'] = identified_clusters

    f, (a0) = plt.subplots(1, 1)

    for i in distance_percentages_df_with_clusters.index:
        # if i.year == train_end_year:
        a0.axvline(x=i, c=kmeans_cluster_colors[int(distance_percentages_df_with_clusters.loc[i]["Clusters"])])

    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]
    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(a0, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)

    a0.grid(True)
    a0.set_xlabel('Date')
    a0.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    a0.plot()

    plt.title(
        f"{ticker}_k-means_clusters_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters",
        fontdict={'fontsize': 15})
    plt.savefig(
        f"{ticker}_k-means_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters.jpg",
        dpi=300)
    plt.close()

    return


def plot_gmm_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_one_chart_for_each_cluster(
        df_daily, ticker, percentage_distance_cache, number_of_clusters, train_start_year=None, train_end_year=None):
    with open(f'{percentage_distance_cache}', 'rb') as file:
        distance_percentages_df = pickle.load(file)

    distance_percentages_df = distance_percentages_df[(distance_percentages_df.index.year >= train_start_year) & (
                distance_percentages_df.index.year <= train_end_year)]
    gmm = GaussianMixture(n_components=number_of_clusters)
    gmm.fit(distance_percentages_df)
    identified_clusters = gmm.predict(distance_percentages_df)
    distance_percentages_df_with_clusters = distance_percentages_df.copy()
    distance_percentages_df_with_clusters['Clusters'] = identified_clusters

    for i in range(number_of_clusters):
        df_daily = get_data(ticker, "D")
        df_daily = df_daily[(df_daily["Datetime"] >= (str(train_start_year) + "-01-01")) & (
                df_daily["Datetime"] <= (str(train_end_year) + "-12-31"))]
        f, (a0) = plt.subplots(1, 1)
        for j in distance_percentages_df_with_clusters.index:
            if int(distance_percentages_df_with_clusters.loc[j]["Clusters"]) == i:
                a0.axvline(x=j, c=kmeans_cluster_colors[i])

        df_daily = df_daily[['Datetime', 'Open', 'High',
                             'Low', 'Close']]
        df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
        df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
        candlestick_ohlc(a0, df_daily.values, width=0.8,
                         colorup='green', colordown='red',
                         alpha=0.8)

        a0.grid(True)
        a0.set_xlabel('Date')
        a0.set_ylabel('Price')
        date_format = mpdates.DateFormatter('%d-%m-%Y')
        a0.xaxis.set_major_formatter(date_format)
        a0.plot()

        plt.title(
            f"{ticker}_gmm_clusters_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters_cluster_number_{i}",
            fontdict={'fontsize': 15})
        plt.savefig(
            f"{ticker}_gmm_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters_cluster_number_{i}.jpg",
            dpi=300)
        plt.close()

    return


def plot_kmeans_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_one_chart_for_each_cluster(
        df_daily, ticker, percentage_distance_cache, number_of_clusters, train_start_year=None, train_end_year=None):
    with open(f'{percentage_distance_cache}', 'rb') as file:
        distance_percentages_df = pickle.load(file)

    distance_percentages_df = distance_percentages_df[(distance_percentages_df.index.year >= train_start_year) & (
                distance_percentages_df.index.year <= train_end_year)]

    kmeans = KMeans(number_of_clusters)
    kmeans.fit(distance_percentages_df)

    identified_clusters = kmeans.fit_predict(distance_percentages_df)
    distance_percentages_df_with_clusters = distance_percentages_df.copy()
    distance_percentages_df_with_clusters['Clusters'] = identified_clusters

    for i in range(number_of_clusters):
        df_daily = get_data(ticker, "D")
        df_daily = df_daily[(df_daily["Datetime"] >= (str(train_start_year) + "-01-01")) & (
                df_daily["Datetime"] <= (str(train_end_year) + "-12-31"))]
        f, (a0) = plt.subplots(1, 1)
        for j in distance_percentages_df_with_clusters.index:
            if int(distance_percentages_df_with_clusters.loc[j]["Clusters"]) == i:
                a0.axvline(x=j, c=kmeans_cluster_colors[i])

        df_daily = df_daily[['Datetime', 'Open', 'High',
                             'Low', 'Close']]
        df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
        df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
        candlestick_ohlc(a0, df_daily.values, width=0.8,
                         colorup='green', colordown='red',
                         alpha=0.8)

        a0.grid(True)
        a0.set_xlabel('Date')
        a0.set_ylabel('Price')
        date_format = mpdates.DateFormatter('%d-%m-%Y')
        a0.xaxis.set_major_formatter(date_format)
        a0.plot()

        plt.title(
            f"{ticker}_k-means_clusters_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters_cluster_number_{i}",
            fontdict={'fontsize': 15})
        plt.savefig(
            f"{ticker}_k-means_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters_cluster_number_{i}.jpg",
            dpi=300)
        plt.close()

    return


def plot_gmm_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_and_indicator_values_one_chart_for_each_cluster(
        df_daily, ticker, percentage_distance_and_indicators_cache, number_of_clusters, train_start_year=None,
        train_end_year=None):
    with open(f'{percentage_distance_and_indicators_cache}', 'rb') as file:
        distance_percentagesand_indicators_df = pickle.load(file)

    distance_percentagesand_indicators_df = distance_percentagesand_indicators_df[
        (distance_percentagesand_indicators_df.index.year >= train_start_year) & (
                    distance_percentagesand_indicators_df.index.year <= train_end_year)]
    gmm = GaussianMixture(n_components=number_of_clusters)
    gmm.fit(distance_percentagesand_indicators_df)
    identified_clusters = gmm.predict(distance_percentagesand_indicators_df)
    distance_percentages_df_with_clusters = distance_percentagesand_indicators_df.copy()
    distance_percentages_df_with_clusters['Clusters'] = identified_clusters

    for i in range(number_of_clusters):
        df_daily = get_data(ticker, "D")
        df_daily = df_daily[(df_daily["Datetime"] >= (str(train_start_year) + "-01-01")) & (
                df_daily["Datetime"] <= (str(train_end_year) + "-12-31"))]
        f, (a0) = plt.subplots(1, 1)
        for j in distance_percentages_df_with_clusters.index:
            if int(distance_percentages_df_with_clusters.loc[j]["Clusters"]) == i:
                a0.axvline(x=j, c=kmeans_cluster_colors[i])

        df_daily = df_daily[['Datetime', 'Open', 'High',
                             'Low', 'Close']]
        df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
        df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
        candlestick_ohlc(a0, df_daily.values, width=0.8,
                         colorup='green', colordown='red',
                         alpha=0.8)

        a0.grid(True)
        a0.set_xlabel('Date')
        a0.set_ylabel('Price')
        date_format = mpdates.DateFormatter('%d-%m-%Y')
        a0.xaxis.set_major_formatter(date_format)
        a0.plot()

        plt.title(
            f"{ticker}_gmm_clusters_for_non_absolute_percentage_of_poc_and_ohlc_avg_and_indicators_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters_cluster_number_{i}",
            fontdict={'fontsize': 15})
        plt.savefig(
            f"{ticker}_gmm_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_and_indicators_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters_cluster_number_{i}.jpg",
            dpi=300)
        plt.close()

    return


def plot_kmeans_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_and_indicator_values_one_chart_for_each_cluster(
        df_daily, ticker, percentage_distance_and_indicators_cache, number_of_clusters, train_start_year=None,
        train_end_year=None):
    with open(f'{percentage_distance_and_indicators_cache}', 'rb') as file:
        distance_percentagesand_indicators_df = pickle.load(file)

    distance_percentagesand_indicators_df = distance_percentagesand_indicators_df[
        (distance_percentagesand_indicators_df.index.year >= train_start_year) & (
                    distance_percentagesand_indicators_df.index.year <= train_end_year)]

    kmeans = KMeans(number_of_clusters)
    kmeans.fit(distance_percentagesand_indicators_df)

    identified_clusters = kmeans.fit_predict(distance_percentagesand_indicators_df)
    distance_percentages_df_with_clusters = distance_percentagesand_indicators_df.copy()
    distance_percentages_df_with_clusters['Clusters'] = identified_clusters

    for i in range(number_of_clusters):
        df_daily = get_data(ticker, "D")
        df_daily = df_daily[(df_daily["Datetime"] >= (str(train_start_year) + "-01-01")) & (
                df_daily["Datetime"] <= (str(train_end_year) + "-12-31"))]
        f, (a0) = plt.subplots(1, 1)
        for j in distance_percentages_df_with_clusters.index:
            if int(distance_percentages_df_with_clusters.loc[j]["Clusters"]) == i:
                a0.axvline(x=j, c=kmeans_cluster_colors[i])

        df_daily = df_daily[['Datetime', 'Open', 'High',
                             'Low', 'Close']]
        df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
        df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
        candlestick_ohlc(a0, df_daily.values, width=0.8,
                         colorup='green', colordown='red',
                         alpha=0.8)

        a0.grid(True)
        a0.set_xlabel('Date')
        a0.set_ylabel('Price')
        date_format = mpdates.DateFormatter('%d-%m-%Y')
        a0.xaxis.set_major_formatter(date_format)
        a0.plot()

        plt.title(
            f"{ticker}_k-means_clusters_for_non_absolute_percentage_of_poc_and_ohlc_avg_and_indicators_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters_cluster_number_{i}",
            fontdict={'fontsize': 15})
        plt.savefig(
            f"{ticker}_k-means_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_and_indicators_trained_from_{train_start_year}_to_{train_end_year}_with_{number_of_clusters}_clusters_cluster_number_{i}.jpg",
            dpi=300)
        plt.close()

    return


def plot_out_of_sample_days_which_are_part_of_clusters_of_top_5_and_bottom_5_of_forward_statistical_measures_removing_duplicate_clusters(
        df_daily, ticker, out_of_sample_cache_path_for_top5_and_bottom_5_stats):
    with open(f'{out_of_sample_cache_path_for_top5_and_bottom_5_stats}', 'rb') as file:
        out_of_sample_for_top5_and_bottom_5_stats = pickle.load(file)

    df_daily = df_daily[(df_daily["Datetime"] >= ("2021-01-01")) & (
            df_daily["Datetime"] <= ("2022-04-22"))]
    f, (a0) = plt.subplots(1, 1)
    for i in out_of_sample_for_top5_and_bottom_5_stats.keys():
        a0.axvline(x=i, c="orange")

    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]
    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(a0, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)

    a0.grid(True)
    a0.set_xlabel('Date')
    a0.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    a0.plot()

    plt.title(
        f"{ticker}_out_of_sample_days_which_are_part_of_clusters_of_top_5_and_bottom_5_of_forward_statistical_measures",
        fontdict={'fontsize': 15})
    plt.savefig(
        f"{ticker}_out_of_sample_days_which_are_part_of_clusters_of_top_5_and_bottom_5_of_forward_statistical_measures.jpg",
        dpi=300)
    plt.close()
    return


def plot_out_of_sample_days_which_are_part_of_clusters_of_top_40_and_bottom_40_of_forward_statistical_measures_by_combining_over_look_forwards(
        df_daily, ticker, out_of_sample_cache_path_for_top_40_and_bottom_40_stats):
    with open(f'{out_of_sample_cache_path_for_top_40_and_bottom_40_stats}', 'rb') as file:
        out_of_sample_for_top_40_and_bottom_40_stats = pickle.load(file)

    for i in out_of_sample_for_top_40_and_bottom_40_stats.keys():
        df_daily_for_plot = df_daily.copy()
        df_daily_for_plot = df_daily_for_plot[(df_daily_for_plot["Datetime"] >= ("2021-01-01")) & (
                df_daily_for_plot["Datetime"] <= ("2022-04-22"))]
        f, (a0) = plt.subplots(1, 1)
        for j in out_of_sample_for_top_40_and_bottom_40_stats[i].keys():
            if out_of_sample_for_top_40_and_bottom_40_stats[i][j] == "top_40":
                a0.axvline(x=j, c="green")
            if out_of_sample_for_top_40_and_bottom_40_stats[i][j] == "bottom_40":
                a0.axvline(x=j, c="red")

        df_daily_for_plot = df_daily_for_plot[['Datetime', 'Open', 'High',
                                               'Low', 'Close']]
        df_daily_for_plot['Datetime'] = pd.to_datetime(df_daily_for_plot['Datetime'])
        df_daily_for_plot['Datetime'] = df_daily_for_plot['Datetime'].map(mpdates.date2num)
        candlestick_ohlc(a0, df_daily_for_plot.values, width=0.8,
                         colorup='green', colordown='red',
                         alpha=0.8)

        a0.grid(True)
        a0.set_xlabel('Date')
        a0.set_ylabel('Price')
        date_format = mpdates.DateFormatter('%d-%m-%Y')
        a0.xaxis.set_major_formatter(date_format)
        a0.plot()
        plt.title(
            f"{ticker}_out_of_sample_days_which_are_part_of_clusters_of_top_40_and_bottom_40_of_forward_statistical_measures_by_combining_{i}_over_look_forwards",
            fontdict={'fontsize': 15})
        if i.split("_")[-1] == "max/min":
            temp = i.split("_")[:-1]
            temp = "_".join(temp)
            plt.savefig(
                f"{ticker}_out_of_sample_days_which_are_part_of_clusters_of_top_40_and_bottom_40_of_forward_stats_by_combining_{temp}_max_div_by_min_over_look_forwards.jpg",
                dpi=300)
        else:
            plt.savefig(
                f"{ticker}_out_of_sample_days_which_are_part_of_clusters_of_top_40_and_bottom_40_of_forward_stats_by_combining_{i}_over_look_forwards.jpg",
                dpi=300)
        plt.close()
    return


def plot_non_absolute_distance_percentages_of_poc_for_various_lookbacks(ticker, percentage_cache_path):
    with open(f'{percentage_cache_path}', 'rb') as file:
        non_absolute_distance_percentages_of_poc = pickle.load(file)

    f, (a0) = plt.subplots(1, 1)

    for i in tqdm(non_absolute_distance_percentages_of_poc.index):
        lookback = []
        non_absolute_distance_percentages = []
        for j in list(non_absolute_distance_percentages_of_poc.columns):
            non_absolute_distance_percentages.append(non_absolute_distance_percentages_of_poc.loc[i][j])
            lookback.append(-int(j.split("_")[0]))
        a0.scatter([i] * len(lookback), non_absolute_distance_percentages, c=lookback, cmap="gray")
    a0.grid(True)
    a0.plot()
    a0.set_xlabel('Date')
    a0.set_ylabel(f'{ticker}_non_absolute_distance_percentages_of_poc_for_various_lookbacks')
    a0.set_facecolor('pink')
    plt.title(f"{ticker}_non_absolute_distance_percentages_of_poc_for_various_lookbacks", fontdict={'fontsize': 15})
    plt.savefig(f"{ticker}_non_absolute_distance_percentages_of_poc_for_various_lookbacks.jpg", dpi=300)
    plt.close()
    return


def plot_confluence_areas_distance_percentile_and_returns_percentile_scatter_plot_for_fisher_pivots(df_daily, ticker,
                                                                                                    distance_percentile_cache_path,
                                                                                                    confluence_points_cache,
                                                                                                    number_of_days_for_forward_returns_calculation,
                                                                                                    forward_returns_type,
                                                                                                    fisher_lb):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    with open(
            f'{ticker}_percentiles_for_each_day_based_on_{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type}_returns.pkl',
            'rb') as file:
        returns_percentile = pickle.load(file)

    with open(f'{distance_percentile_cache_path}', 'rb') as file:
        distance_percentile = pickle.load(file)

    with open(f'{ticker}_fisher_{fisher_lb}_lb_low_pivots.pkl', 'rb') as file:
        fisher_low_pivots = pickle.load(file)

    with open(f'{ticker}_fisher_{fisher_lb}_lb_high_pivots.pkl', 'rb') as file:
        fisher_high_pivots = pickle.load(file)

    days_to_plot = {}
    for i in confluence_points.keys():
        if i in distance_percentile.keys() and i.year < 2021 and i.year >= 2019:
            days_to_plot[i] = confluence_points[i]
    df_daily = df_daily[(df_daily["Datetime"] <= max(days_to_plot.keys()).strftime("%Y-%m-%d")) & (
                df_daily["Datetime"] >= min(days_to_plot.keys()).strftime("%Y-%m-%d"))]

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
    # ax3 = plt.subplot(2, 1, 2)
    for i in list(days_to_plot.keys()):
        confluence_point = []
        confluence_strength = []
        for j in list(days_to_plot[i].keys()):
            confluence_point.append(j)
            if days_to_plot[i][j]["confluence_strength"] > 3:
                confluence_strength.append(-300)
            elif days_to_plot[i][j]["confluence_strength"] < 1:
                confluence_strength.append(300)
            else:
                confluence_strength.append(-days_to_plot[i][j]["confluence_strength"])
        a0.scatter([i] * len(confluence_point), confluence_point, c=confluence_strength, cmap="gray")

    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]
    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(a0, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)
    a0.grid(True)
    a0.set_xlabel('Date')
    a0.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    a0.xaxis.set_major_formatter(date_format)
    a0.set_facecolor('pink')
    a0.plot()
    a1_xaxis = []
    a1_yaxis = []
    for i in days_to_plot.keys():
        if i in returns_percentile.keys() and (i in fisher_low_pivots):
            a1_xaxis.append(returns_percentile[i])
            a1_yaxis.append(distance_percentile[i]["distance_percentile_score"])
    a1.scatter(a1_xaxis, a1_yaxis)
    a1.set_xlabel(
        f"{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type}_returns_percentile_fisher_low_pivots")
    a1.set_ylabel("closest_confluence_distance_percentile_fisher_low_pivots")
    # plt.legend()
    plt.title(
        f"confluence_points_with_distance_percentile_and_{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type}_returns_percentile_fisher_{fisher_lb}_lb_low_pivots",
        fontdict={'fontsize': 20})
    plt.savefig(
        f"confluence_points_with_distance_percentile_and_{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type}_returns_percentile_fisher_{fisher_lb}_lb_low_pivots.jpg")
    plt.close()

    x = np.array(a1_xaxis).reshape(-1, 1)
    y = np.array(a1_yaxis).reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    r_square = reg.score(x, y)

    with open(f'{ticker}_linear_regression_r_squared.pkl', 'rb') as file:
        linear_regression_r_squared_dict = pickle.load(file)

    linear_regression_r_squared_dict[
        f"confluence_points_with_distance_percentile_and_{number_of_days_for_forward_returns_calculation}_day_{forward_returns_type}_returns_percentile_fisher_{fisher_lb}_lb_low_pivots"] = r_square

    with open(f'{ticker}_linear_regression_r_squared.pkl', 'wb') as file:
        pickle.dump(linear_regression_r_squared_dict, file)

    return


# def plot_non_white_confluence_areas_and_distance_percentile_for_closest_non_white_confluence_areas_distance_percentile_and_returns_percentile_scatter_plot_for_fisher_pivots()

def plot_confluence_areas_with_2_and_3_standard_deviations_above_n_day_close_price_over_topmost_confluence(df_daily,
                                                                                                           ticker,
                                                                                                           confluence_points_cache,
                                                                                                           top_most_confluence_point_cache,
                                                                                                           rolling_window_for_std_dev):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    with open(f'{top_most_confluence_point_cache}', 'rb') as file:
        top_most_confluence_points = pickle.load(file)

    df_daily[f"{rolling_window_for_std_dev}_day_std_dev_of_close_price"] = df_daily["Close"].rolling(
        rolling_window_for_std_dev).std()

    days_to_plot = {}
    for i in confluence_points.keys():
        if i.year >= 2021:
            days_to_plot[i] = confluence_points[i]

    df_daily = df_daily[(df_daily["Datetime"] <= max(days_to_plot.keys()).strftime("%Y-%m-%d")) & (
                df_daily["Datetime"] >= min(days_to_plot.keys()).strftime("%Y-%m-%d"))]

    f, (ax) = plt.subplots(1, 1)
    # ax3 = plt.subplot(2, 1, 2)
    for i in list(days_to_plot.keys()):
        confluence_point = []
        confluence_strength = []
        for j in list(days_to_plot[i].keys()):
            confluence_point.append(j)
            if days_to_plot[i][j]["confluence_strength"] > 3:
                confluence_strength.append(-300)
            elif days_to_plot[i][j]["confluence_strength"] < 1:
                confluence_strength.append(300)
            else:
                confluence_strength.append(-days_to_plot[i][j]["confluence_strength"])
        ax.scatter([i] * len(confluence_point), confluence_point, c=confluence_strength, cmap="gray")
        two_std_above_top_most_confluence = top_most_confluence_points[i]['top_most_confluence'] + 2 * \
                                            df_daily[df_daily["Datetime"] == i].iloc[0][
                                                f"{rolling_window_for_std_dev}_day_std_dev_of_close_price"]
        three_std_above_top_most_confluence = top_most_confluence_points[i]['top_most_confluence'] + 3 * \
                                              df_daily[df_daily["Datetime"] == i].iloc[0][
                                                  f"{rolling_window_for_std_dev}_day_std_dev_of_close_price"]
        ax.scatter(i, two_std_above_top_most_confluence, color="orange")
        ax.scatter(i, three_std_above_top_most_confluence, color="red")
    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]
    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(ax, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)
    ax.grid(True)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    ax.xaxis.set_major_formatter(date_format)
    ax.set_facecolor('lightblue')
    ax.plot()
    plt.savefig(
        f"{ticker}_confluence_areas_with_2_and_3_standard_deviations_above_{rolling_window_for_std_dev}_day_close_price_over_topmost_confluence.jpg",
        dpi=300)
    plt.close()
    return


def plot_confluence_areas_with_1_2_and_3_standard_deviations_above_n_day_close_price_over_topmost_non_white_confluence(
        df_daily, ticker, confluence_points_cache, top_most_non_white_confluence_point_cache,
        rolling_window_for_std_dev):
    with open(f'{confluence_points_cache}', 'rb') as file:
        confluence_points = pickle.load(file)

    with open(f'{top_most_non_white_confluence_point_cache}', 'rb') as file:
        top_most_confluence_points = pickle.load(file)

    df_daily[f"{rolling_window_for_std_dev}_day_std_dev_of_close_price"] = df_daily["Close"].rolling(
        rolling_window_for_std_dev).std()

    days_to_plot = {}
    for i in confluence_points.keys():
        if i.year >= 2021:
            days_to_plot[i] = confluence_points[i]

    df_daily = df_daily[(df_daily["Datetime"] <= max(days_to_plot.keys()).strftime("%Y-%m-%d")) & (
                df_daily["Datetime"] >= min(days_to_plot.keys()).strftime("%Y-%m-%d"))]

    f, (ax) = plt.subplots(1, 1)
    # ax3 = plt.subplot(2, 1, 2)
    for i in list(days_to_plot.keys()):
        confluence_point = []
        confluence_strength = []
        for j in list(days_to_plot[i].keys()):
            confluence_point.append(j)
            if days_to_plot[i][j]["confluence_strength"] > 3:
                confluence_strength.append(-300)
            elif days_to_plot[i][j]["confluence_strength"] < 1:
                confluence_strength.append(300)
            else:
                confluence_strength.append(-days_to_plot[i][j]["confluence_strength"])
        ax.scatter([i] * len(confluence_point), confluence_point, c=confluence_strength, cmap="gray")
        one_std_above_top_most_confluence = top_most_confluence_points[i]['top_most_confluence'] + 1 * \
                                            df_daily[df_daily["Datetime"] == i].iloc[0][
                                                f"{rolling_window_for_std_dev}_day_std_dev_of_close_price"]
        two_std_above_top_most_confluence = top_most_confluence_points[i]['top_most_confluence'] + 2 * \
                                            df_daily[df_daily["Datetime"] == i].iloc[0][
                                                f"{rolling_window_for_std_dev}_day_std_dev_of_close_price"]
        three_std_above_top_most_confluence = top_most_confluence_points[i]['top_most_confluence'] + 3 * \
                                              df_daily[df_daily["Datetime"] == i].iloc[0][
                                                  f"{rolling_window_for_std_dev}_day_std_dev_of_close_price"]
        ax.scatter(i, one_std_above_top_most_confluence, color="yellow")
        ax.scatter(i, two_std_above_top_most_confluence, color="orange")
        ax.scatter(i, three_std_above_top_most_confluence, color="red")

    df_daily = df_daily[['Datetime', 'Open', 'High',
                         'Low', 'Close']]
    df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime'])
    df_daily['Datetime'] = df_daily['Datetime'].map(mpdates.date2num)
    candlestick_ohlc(ax, df_daily.values, width=0.8,
                     colorup='green', colordown='red',
                     alpha=0.8)
    ax.grid(True)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    date_format = mpdates.DateFormatter('%d-%m-%Y')
    ax.xaxis.set_major_formatter(date_format)
    ax.set_facecolor('lightblue')
    ax.plot()
    plt.savefig(
        f"{ticker}_confluence_areas_with_1_2_and_3_standard_deviations_above_{rolling_window_for_std_dev}_day_close_price_over_topmost_non_white_confluence.jpg",
        dpi=600)
    plt.close()
    return


def plot_forward_returns_histogram_for_pivots(ticker):
    with open(f'Confluence_Points_{ticker}_test.pkl', 'rb') as file:
        confluence_points = pickle.load(file)

    df_daily = get_data(ticker, "D")
    df_daily = df_daily[(df_daily["Datetime"] <= max(confluence_points.keys()).strftime("%Y-%m-%d")) & (
            df_daily["Datetime"] >= min(confluence_points.keys()).strftime("%Y-%m-%d"))]

    with open(f'{ticker}_high_pivots.pkl', 'rb') as file:
        high_pivots = pickle.load(file)

    with open(f'{ticker}_low_pivots.pkl', 'rb') as file:
        low_pivots = pickle.load(file)

    with open(f'{ticker}_forward_returns.pkl', 'rb') as file:
        forward_returns_df = pickle.load(file)

    forward_returns_df = forward_returns_df[
        (forward_returns_df.index <= max(confluence_points.keys()).strftime("%Y-%m-%d")) & (
                forward_returns_df.index >= min(confluence_points.keys()).strftime("%Y-%m-%d"))]

    for i in forward_return_list_in_days:
        high_pivots_min_returns = []
        high_pivots_max_returns = []
        high_pivots_end_to_end_returns = []
        for j in high_pivots:
            try:
                high_pivots_min_returns.append(forward_returns_df.loc[j]["min_returns"][f"{i}_day"])
                high_pivots_max_returns.append(forward_returns_df.loc[j]["max_returns"][f"{i}_day"])
                high_pivots_end_to_end_returns.append(forward_returns_df.loc[j]["end_to_end_returns"][f"{i}_day"])
            except Exception as e:
                print(e)
                continue
        value_area_calculation_for_any_histogram(high_pivots_min_returns, plot=True,
                                                 plot_path=f"{ticker}_{i}_day_forward_min_returns_with_value_area_for_high_pivots.jpg",
                                                 plot_label=f"{i}_day_forward_min_returns_high_pivots")
        value_area_calculation_for_any_histogram(high_pivots_max_returns, plot=True,
                                                 plot_path=f"{ticker}_{i}_day_forward_max_returns_with_value_area_for_high_pivots.jpg",
                                                 plot_label=f"{i}_day_forward_max_returns_high_pivots")
        value_area_calculation_for_any_histogram(high_pivots_end_to_end_returns, plot=True,
                                                 plot_path=f"{ticker}_{i}_day_forward_end_to_end_returns_with_value_area_for_high_pivots.jpg",
                                                 plot_label=f"{i}_day_forward_end_to_end_returns_high_pivots")

    for i in forward_return_list_in_days:
        low_pivots_min_returns = []
        low_pivots_max_returns = []
        low_pivots_end_to_end_returns = []
        for j in low_pivots:
            try:
                low_pivots_min_returns.append(forward_returns_df.loc[j]["min_returns"][f"{i}_day"])
                low_pivots_max_returns.append(forward_returns_df.loc[j]["max_returns"][f"{i}_day"])
                low_pivots_end_to_end_returns.append(forward_returns_df.loc[j]["end_to_end_returns"][f"{i}_day"])
            except Exception as e:
                print(e)
                continue
        value_area_calculation_for_any_histogram(low_pivots_min_returns, plot=True,
                                                 plot_path=f"{ticker}_{i}_day_forward_min_returns_with_value_area_for_low_pivots.jpg",
                                                 plot_label=f"{i}_day_forward_min_returns_low_pivots")
        value_area_calculation_for_any_histogram(low_pivots_max_returns, plot=True,
                                                 plot_path=f"{ticker}_{i}_day_forward_max_returns_with_value_area_for_low_pivots.jpg",
                                                 plot_label=f"{i}_day_forward_max_returns_low_pivots")
        value_area_calculation_for_any_histogram(low_pivots_end_to_end_returns, plot=True,
                                                 plot_path=f"{ticker}_{i}_day_forward_end_to_end_returns_with_value_area_for_low_pivots.jpg",
                                                 plot_label=f"{i}_day_forward_end_to_end_returns_low_pivots")

    return


def plot_returns_histogram_for_pivots_based_on_confluence_areas(ticker):
    with open(f'Confluence_Points_{ticker}_test.pkl', 'rb') as file:
        confluence_points = pickle.load(file)

    with open(f'{ticker}_high_pivots_with_confluence_point_responsible.pkl', 'rb') as file:
        high_pivots = pickle.load(file)

    with open(f'{ticker}_low_pivots_with_confluence_point_responsible.pkl', 'rb') as file:
        low_pivots = pickle.load(file)

    df_daily = get_data(ticker, "D")
    low_pivot_returns = []
    high_pivot_returns = []

    for i in list(low_pivots.keys()):
        try:
            df_daily_for_low_pivot_returns = df_daily[df_daily["Datetime"] > i]
            starting_index = min(df_daily_for_low_pivot_returns.index)
            end_index = max(df_daily_for_low_pivot_returns.index)
            for j in df_daily_for_low_pivot_returns.index:
                if df_daily_for_low_pivot_returns.loc[j]["Close"] < (low_pivots[i] * 0.98):
                    end_index = j
                    break
            max_close = max(df_daily_for_low_pivot_returns.loc[starting_index:end_index + 1]["Close"])
            returns = (max_close - df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]) / \
                      df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
            low_pivot_returns.append(returns)
        except Exception as e:
            print(e)
    value_area_calculation_for_any_histogram(low_pivot_returns, plot=True,
                                             plot_path=f"{ticker}_low_pivot_returns_for_max_till_it_goes_below_2_percent_of_confluence_area.jpg",
                                             plot_label=f"{ticker}_low_pivot_returns_for_max_till_it_goes_below_2_percent_of_confluence_area")

    # for i in list(high_pivots.keys()):
    #     try:
    #         df_daily_for_high_pivot_returns = df_daily[df_daily["Datetime"] > i]
    #         starting_index = min(df_daily_for_high_pivot_returns.index)
    #         end_index = max(df_daily_for_high_pivot_returns.index)
    #         for j in df_daily_for_high_pivot_returns.index:
    #             if df_daily_for_high_pivot_returns.loc[j]["Close"] > high_pivots[i]:
    #                 end_index = j
    #                 break
    #         min_close = min(df_daily_for_high_pivot_returns.loc[starting_index:end_index+1]["Close"])
    #         returns = (min_close - df_daily[df_daily["Datetime"] == i].iloc[0]["Close"])/df_daily[df_daily["Datetime"] == i].iloc[0]["Close"]
    #         high_pivot_returns.append(returns)
    #     except Exception as e:
    #         print(e)
    # value_area_calculation_for_any_histogram(high_pivot_returns,plot=True,plot_path=f"{ticker}_high_pivot_returns_for_min_till_it_goes_above_confluence_area.jpg",plot_label=f"{ticker}_high_pivot_returns_for_min_till_it_goes_above_confluence_area")


def plot_returns_histogram_for_pivots_based_on_confluence_areas_of_n_days(ticker):
    return


def detect_exhaustion_bars(ticker):
    with open(f'Confluence_Points_{ticker}_test.pkl', 'rb') as file:
        confluence_points = pickle.load(file)

    df_daily = get_data(ticker, "D")
    df_daily = df_daily[(df_daily["Datetime"] <= max(confluence_points.keys()).strftime("%Y-%m-%d")) & (
            df_daily["Datetime"] >= min(confluence_points.keys()).strftime("%Y-%m-%d"))]

    exhaustion_tops = []
    exhaustion_bottoms = []

    for i in df_daily.index:
        try:
            if df_daily.loc[i - 1]["Close"] < df_daily.loc[i]["Open"]:
                if max(list(confluence_points[df_daily.loc[i]["Datetime"]].keys())) < df_daily.loc[i]["Open"]:
                    exhaustion_tops.append(df_daily.loc[i]["Datetime"])

            if df_daily.loc[i - 1]["Close"] > df_daily.loc[i]["Open"]:
                if min(list(confluence_points[df_daily.loc[i]["Datetime"]].keys())) > (df_daily.loc[i]["Open"] * 1.05):
                    exhaustion_bottoms.append(df_daily.loc[i]["Datetime"])
        except Exception as e:
            print(e)
            continue

    with open(f'{ticker}_exhaustion_tops.pkl', 'wb') as file:
        pickle.dump(exhaustion_tops, file)

    with open(f'{ticker}_exhaustion_bottoms.pkl', 'wb') as file:
        pickle.dump(exhaustion_bottoms, file)

    return


def detect_stretched_and_very_stretched_highs(ticker):
    with open(f'Confluence_Points_{ticker}_test.pkl', 'rb') as file:
        confluence_points = pickle.load(file)

    df_daily = get_data(ticker, "D")
    df_daily = df_daily[(df_daily["Datetime"] <= max(confluence_points.keys()).strftime("%Y-%m-%d")) & (
            df_daily["Datetime"] >= min(confluence_points.keys()).strftime("%Y-%m-%d"))]
    df_daily[f"{rolling_window_for_std_dev_for_stretched_high_bars}_day_std_dev"] = df_daily["Close"].rolling(
        window=rolling_window_for_std_dev_for_stretched_high_bars).std()

    stretched_highs = []
    very_stretched_highs = []

    for i in df_daily.index:
        try:
            if df_daily.loc[i - 1]["Close"] < df_daily.loc[i]["Open"]:
                if max(list(confluence_points[df_daily.loc[i]["Datetime"]].keys())) < df_daily.loc[i]["Open"]:
                    very_stretched_highs.append(df_daily.loc[i]["Datetime"])

            if df_daily.loc[i - 1]["Close"] > df_daily.loc[i]["Open"]:
                if min(list(confluence_points[df_daily.loc[i]["Datetime"]].keys())) > (df_daily.loc[i]["Open"] * 1.05):
                    very_stretched_highs.append(df_daily.loc[i]["Datetime"])
        except Exception as e:
            print(e)
            continue


def plot_confluence_area_and_constance_brown_indicator_for_all_days_plotly(df_daily_with_constance_brown,
                                                                           confluence_points_cache_path, ticker):
    with open(f'{confluence_points_cache_path}', 'rb') as file:
        confluence_points = pickle.load(file)

    df_daily_with_constance_brown = df_daily_with_constance_brown[
        (df_daily_with_constance_brown["Datetime"] <= "2022-03-24") & (
                    df_daily_with_constance_brown["Datetime"] >= "2021-10-01")]
    fig = make_subplots(rows=2, cols=1, row_heights=[1, 1], shared_xaxes=True, shared_yaxes=True,
                        specs=[[{"secondary_y": True}], [{"secondary_y": True}]])

    fig.update_layout(xaxis2={'anchor': 'y', 'overlaying': 'x', 'side': 'top'},
                      yaxis_domain=[0, 1])

    fig.add_trace(go.Candlestick(name=".NSEI", x=df_daily_with_constance_brown['Datetime'],
                                 open=df_daily_with_constance_brown['Open'],
                                 high=df_daily_with_constance_brown['High'],
                                 low=df_daily_with_constance_brown['Low'],
                                 close=df_daily_with_constance_brown['Close']), row=1, col=1)

    # fig.add_trace(go.Scatter(x=df_daily_with_constance_brown["Datetime"], y=confluence_points[df_daily_with_constance_brown["Datetime"]], name="Constance-Brown Index", mode='markers',
    #                          marker=dict(color='darkorchid')), row=1, col=1)

    for i in list(confluence_points.keys()):
        confluence_point = []
        confluence_strength = []
        for j in list(confluence_points[i].keys()):
            confluence_point.append(j)
            confluence_strength.append(-confluence_points[i][j])
        fig.add_trace(go.Scatter(x=[i] * len(confluence_point),
                                 y=confluence_point, name=None, mode='markers',
                                 marker=dict(color=confluence_strength, colorscale="gray")), row=1, col=1)
        # ax.scatter([i]*len(confluence_point),confluence_point,c=confluence_strength,cmap="gray")

    fig.add_trace(go.Scatter(x=df_daily_with_constance_brown["Datetime"], y=df_daily_with_constance_brown["CB"],
                             name="Constance-Brown Index", mode='lines',
                             marker=dict(color='darkorchid')), row=2,
                  col=1,
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df_daily_with_constance_brown["Datetime"], y=df_daily_with_constance_brown["FMACB"],
                             name="FMA Constance-Brown Index", mode='lines',
                             marker=dict(color='orange')), row=2,
                  col=1,
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df_daily_with_constance_brown["Datetime"], y=df_daily_with_constance_brown["SMACB"],
                             name="SMA Constance-Brown Index", mode='lines',
                             marker=dict(color='aquamarine')), row=2,
                  col=1,
                  secondary_y=False)

    fig.data[1].update(xaxis='x2')
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(template="plotly", width=1500, height=1000)
    fig.show()
    fig.write_html(f"confluence_areas_with_constance_brown_{ticker}.html")
    return


def filter_pivots_based_on_constance_brown_and_confluence_strength_and_fisher(ticker):
    # with open(f'{ticker}_low_pivots_with_confluence_point_of_poc_point_of_low_vol_and_strength.pkl', 'rb') as file:
    #     low_pivots = pickle.load(file)
    with open(f'{ticker}_low_pivots_with_confluence_point_responsible_and_strength.pkl', 'rb') as file:
        low_pivots = pickle.load(file)

    df_daily = get_data(".NSEI", "D")
    df_daily = constance_brown(df_daily)
    df_daily[f"{fisher_lookback}_day_fisher"] = fisher(df_daily, fisher_lookback)
    df_daily = df_daily[df_daily["Datetime"] > datetime.datetime.strptime('2012-01-01 00:00:00', "%Y-%m-%d %H:%M:%S")]
    filtered_low_pivots_based_on_confluence_strength = [i for i in low_pivots.keys() if low_pivots[i][1] > 2]

    for i in range(-80, 140, 20):
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
        for j in filtered_low_pivots_based_on_confluence_strength:
            if df_daily[df_daily["Datetime"] == j].iloc[0]["CB"] >= i and df_daily[df_daily["Datetime"] == j].iloc[0][
                "CB"] < i + 20:
                a0.axvline(x=j, color="orange")
        a0.plot(df_daily["Datetime"], df_daily["Close"], label=f"{i} <= Constance Brown < {i + 20}")
        a1.plot(df_daily["Datetime"], df_daily[f"{fisher_lookback}_day_fisher"])
        plt.savefig(f"{ticker}_low_pivots_filtered_by_confluence_strength_fisher_and_CB_gt_{i}_and_lt_{i + 20}.jpg",
                    dpi=200)
        plt.close()

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
    for j in filtered_low_pivots_based_on_confluence_strength:
        if df_daily[df_daily["Datetime"] == j].iloc[0]["CB"] < -80:
            plt.axvline(x=j, color="orange")
    a0.plot(df_daily["Datetime"], df_daily["Close"], label=f"Constance Brown < -80")
    a1.plot(df_daily["Datetime"], df_daily[f"{fisher_lookback}_day_fisher"])
    plt.savefig(f"{ticker}_low_pivots_filtered_by_confluence_strength_fisher_and_CB_lt_-80.jpg", dpi=200)
    plt.close()

    return


if __name__ == "__main__":
    start = datetime.datetime.now()

    # df_minute = get_data_ETH_minute(path=f'MinuteOHLCV.pkl')
    # df_hour = resample_data(df_minute, 60)
    #
    # vol_feat = return_volume_features_minute_hourly(df_hour, df_minute)
    # with open(f'VolumeFeatures_new.pkl', 'wb') as file:
    #     pickle.dump(vol_feat, file)
    # nifty_not_available_tickers = ['APLH.NS','BPCL.NS','BRIT.NS','EICH.NS','GRAS.NS','HDFL.NS','HALC.NS','JSTL.NS','SBIL.NS','SHCM.NS','TACN.NS']
    # #
    # for i in current_nifty_tickers:
    #     if i in nifty_not_available_tickers:
    #         df_daily = get_data(i, "D")
    #         vol_features_daily = return_volume_features_daily(df_daily,i)
    #         # ticker_file_name = i.split(".")[0]
    #         with open(f'VolumeFeatures_Daily_{i}_test.pkl', 'wb') as file:
    #             pickle.dump(vol_features_daily, file)
    #
    #         # plot_volume_level_points_for_n_days(df_daily,"VolumeFeatures_NSEI_Daily_test.pkl")
    #         plot_confluence_areas(df_daily,i, f"VolumeFeatures_Daily_{i}_test.pkl", plot_path=f"{i}.jpg")
    #     print(current_nifty_tickers.index(i),"-Done")

    # df_daily = get_data(".NSEI", "D")
    # vol_features_daily = return_volume_features_daily(df_daily)
    # with open(f'VolumeFeatures_Daily_.NSEI_test.pkl', 'wb') as file:
    #     pickle.dump(vol_features_daily, file)

    # plot_confluence_areas(df_daily,".NSEI","VolumeFeatures_.NSEI_Daily_test.pkl","2021_till_today_confluence_plot.jpg")
    # plot_confluence_areas(df_daily, ".NSEI", f"VolumeFeatures_NSEI_Daily_test.pkl", plot_path=f".NSEI.jpg")
    # constance_brown_df = constance_brown(df_daily)
    # plot_confluence_area_and_constance_brown_indicator_for_all_days_plotly(constance_brown_df,'Confluence_Points_.NSEI_test.pkl',".NSEI")
    # plot_price_cb(constance_brown_df)
    # plot_confluence_areas_and_constance_brown(constance_brown_df, ".NSEI", f"VolumeFeatures_NSEI_Daily_test.pkl")
    # plot_confluence_areas_and_constance_brown_and_pivots(constance_brown_df, ".NSEI",
    #                                                      f"VolumeFeatures_NSEI_Daily_test.pkl",
    #                                                      plot_path=f".NSEI_pivots.jpg")

    # calculate_forward_returns(df_daily,".NSEI")
    # plot_confluence_areas_and_constance_brown(constance_brown_df, ".NSEI", f"VolumeFeatures_NSEI_Daily_test.pkl")
    # with open(f'Confluence_Points_.NSEI_test.pkl', 'rb') as file:
    #     confluence_points = pickle.load(file)
    # #
    # df_daily = get_data(".NSEI", "D")
    # df_daily = df_daily[(df_daily["Datetime"] <= max(confluence_points.keys()).strftime("%Y-%m-%d")) & (
    #             df_daily["Datetime"] >= min(confluence_points.keys()).strftime("%Y-%m-%d"))]
    # # cache_pivots(df_daily,confluence_points,".NSEI")
    # # cache_pivots_along_with_confluence(df_daily,confluence_points,".NSEI")
    # cache_pivots_along_with_confluence_point_and_its_strength(df_daily,confluence_points,".NSEI")

    # with open(f'Confluence_Points_.NSEI_with_matching_of_poc_and_poc_or_poc_and_point_of_least_volume.pkl', 'rb') as file:
    #     confluence_points = pickle.load(file)
    # #
    # confluence_points = {x:confluence_points[x] for x in confluence_points.keys() if x>datetime.datetime.strptime('2012-02-01 00:00:00', "%Y-%m-%d %H:%M:%S")}
    # df_daily = get_data(".NSEI", "D")
    # df_daily = df_daily[(df_daily["Datetime"] <= max(confluence_points.keys()).strftime("%Y-%m-%d")) & (
    #             df_daily["Datetime"] >= min(confluence_points.keys()).strftime("%Y-%m-%d"))]
    # cache_pivots_along_with_confluence_point_of_poc_point_of_low_vol_and_its_strength(df_daily, confluence_points, ".NSEI")
    # plot_forward_returns_histogram_for_pivots(".NSEI")
    # detect_exhaustion_bars(".NSEI")
    # plot_returns_histogram_for_pivots_based_on_confluence_areas(".NSEI")
    # cache_confluence_areas_with_matching_of_poc_and_poc_or_poc_and_point_of_least_volume(".NSEI")
    # filter_pivots_based_on_constance_brown_and_confluence_strength_and_fisher(".NSEI")

    # cache_confluence_areas_along_with_lookbacks(".NSEI")
    # cache_confluence_areas_based_on_weights(".NSEI")
    # with open(f'Confluence_Points_.NSEI_along_with_lookbacks.pkl', 'rb') as file:
    #     confluence_points = pickle.load(file)
    # df_daily = get_data(".NSEI", "D")
    #
    # cache_pivots_along_with_confluence_point_and_its_lookback(df_daily, confluence_points, ".NSEI")

    # find_optimal_weights_for_lookbacks_for_confluence_strength_using_mcmc(".NSEI")
    # find_optimal_weights_for_lookbacks_for_confluence_strength_using_mcmc(".NSEI")

    # cache_confluence_areas_along_with_matching_of_poc_and_point_of_least_volume_with_lookbacks(".NSEI")
    #
    # with open(f'Confluence_Points_.NSEI_with_matching_of_poc_and_point_of_least_volume_along_with_lookbacks.pkl', 'rb') as file:
    #     confluence_points = pickle.load(file)
    # df_daily = get_data(".NSEI", "D")
    #
    # cache_pivots_along_with_confluence_point_of_poc_and_point_of_lowest_volume_and_its_lookback(df_daily, confluence_points, ".NSEI")

    # find_optimal_weights_for_lookbacks_for_confluence_strength_using_mcmc(".NSEI")
    # cache_confluence_areas_along_with_its_constituents_and_their_lookbacks(".NSEI")
    # cache_closest_confluence_areas_along_with_its_constituents_and_their_lookbacks_and_exponential_function_value(".NSEI")
    # cache_closest_confluence_areas_along_with_its_constituents_and_their_lookbacks_and_exponential_function_value_in_form_of_numpy_array(".NSEI")
    # find_optimal_weights_for_look backs_for_confluence_strength_using_mcmc_for_exponential_function_output(".NSEI",30,"max_returns")
    # cache_closest_confluence_areas_along_with_its_constituents_and_their_lookbacks_and_exponential_function_value_in_form_of_numpy_array(".NSEI")

    # cache_percentiles_for_each_day_based_on_historical_returns(".NSEI",30,"end_to_end_returns")
    # cache_closest_confluence_areas_along_with_its_constituents_and_their_lookbacks_and_exponential_function_value_in_form_of_numpy_array_for_percentile_above_85(".NSEI",30,"end_to_end_returns")
    # find_optimal_weights_for_lookbacks_for_confluence_strength_using_mcmc_for_exponential_function_output_percentile_gt_85(".NSEI",30,"end_to_end_returns")

    # cache_percentiles_for_each_day_based_on_all_days_returns(".NSEI", 30, "end_to_end_returns")
    # cache_closest_confluence_areas_along_with_its_constituents_and_their_lookbacks_and_exponential_function_value_in_form_of_numpy_array_for_all_days_percentile_above_85(".NSEI", 30, "end_to_end_returns")
    # find_optimal_weights_for_lookbacks_for_confluence_strength_using_mcmc_for_exponential_function_output_all_days_percentile_gt_85(".NSEI", 30, "end_to_end_returns")

    # df_daily = get_data("NIFc1", "D")
    # vol_features_daily = return_volume_features_daily(df_daily)
    # with open(f'VolumeFeatures_Daily_NIFc1.pkl', 'wb') as file:
    #     pickle.dump(vol_features_daily, file)
    # plot_confluence_areas(df_daily, "NIFc1", "VolumeFeatures_Daily_NIFc1.pkl","2020_confluence_plot.jpg")
    # cache_confluence_areas_along_with_its_constituents_and_their_lookbacks("NIFc1")

    # cache_confluence_areas_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength("NIFc1",vah_val_inclusion=False)
    # plot_confluence_areas_with_weighted_confluence_strength(df_daily,"NIFc1",f'Confluence_Points_NIFc1_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl',f"2020_confluence_plot_till_march_{confluence_area_threshold*100}_percent.jpg")
    # plot_confluence_areas_with_weighted_confluence_strength(df_daily, "NIFc1",f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl',f"2020_confluence_plot_poc_polv_till_march_{confluence_area_threshold * 100}_percent.jpg")

    # plot_confluence_areas(df_daily, "NIFc1", "VolumeFeatures_Daily_NIFc1.pkl", "2020_confluence_plot.jpg")
    # cache_percentiles_of_distance_of_closest_confluence_point_for_each_day("NIFc1",f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl')
    # calculate_forward_returns(df_daily,"NIFc1")
    # cache_percentiles_for_each_day_based_on_historical_returns("NIFc1", 10, "end_to_end_returns")
    # cache_percentiles_for_each_day_based_on_historical_returns("NIFc1", 30, "end_to_end_returns")
    # cache_percentiles_for_each_day_based_on_historical_returns("NIFc1", 60, "end_to_end_returns")
    # cache_percentiles_for_each_day_based_on_historical_returns("NIFc1", 10, "max_returns")
    # cache_percentiles_for_each_day_based_on_historical_returns("NIFc1", 30, "max_returns")
    # cache_percentiles_for_each_day_based_on_historical_returns("NIFc1", 60, "max_returns")
    # cache_percentiles_for_each_day_based_on_absolute_historical_returns("NIFc1",30,"end_to_end_returns")
    # plot_confluence_areas_distance_percentile_and_returns_percentile(df_daily,"NIFc1","NIFc1_percentiles_of_distance_of_closest_confluence_point_for_each_day.pkl",
    #                                                                  f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl',
    #                                                                  30,"end_to_end")
    # plot_confluence_areas_distance_percentile_and_returns_percentile_scatter_plot(df_daily, "NIFc1",
    #                                                                  "NIFc1_percentiles_of_distance_of_closest_confluence_point_for_each_day.pkl",
    #                                                                  f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl',
    #                                                                  30, "end_to_end")
    # df_daily = get_data("NIFc1", "D")
    # calculate_forward_returns(df_daily, "NIFc1")
    # cache_top_most_confluence_point_for_each_day("NIFc1",f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl')
    # cache_top_most_non_white_confluence_point_for_each_day("NIFc1",f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl')
    # cache_percentiles_of_non_absolute_distance_for_closest_non_white_confluence_point_for_each_day("NIFc1",f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl')
    # plot_confluence_areas_with_2_and_3_standard_deviations_above_n_day_close_price_over_topmost_confluence(df_daily,"NIFc1",
    #                                                                                                        f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl',
    #                                                                                                        'NIFc1_top_most_confluence_point_for_each_day.pkl',60)
    # plot_confluence_areas_with_1_2_and_3_standard_deviations_above_n_day_close_price_over_topmost_non_white_confluence(df_daily,"NIFc1",
    #                                                                                                        f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl',
    #                                                                                                        'NIFc1_top_most_non_white_confluence_point_for_each_day.pkl',60)
    # cache_fisher_pivots(df_daily,"NIFc1")
    # plot_confluence_areas_distance_percentile_and_returns_percentile_scatter_plot_for_fisher_pivots(df_daily, "NIFc1",
    #                                                                   "NIFc1_percentiles_of_distance_of_closest_confluence_point_for_each_day.pkl",
    #                                                                   f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl',
    #                                                                   60, "end_to_end",260)
    # plot_non_white_confluence_areas_and_distance_percentile_for_closest_non_white_confluence_and_returns_percentile_scatter_plot(df_daily, "NIFc1",
    #                                                                  "NIFc1_percentiles_of_non_absolute_distance_for_closest_non_white_confluence_point_for_each_day.pkl",
    #                                                                   f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl',
    #                                                                   60, "max")
    # cache_percentiles_of_absolute_distance_for_closest_non_white_confluence_point_for_each_day("NIFc1",f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl')
    # plot_non_white_confluence_areas_and_mark_points_absolute_distance_percentile_between_0_and_30(df_daily, "NIFc1",
    #                                                                  "NIFc1_percentiles_of_minimum_same_day_ohlc_absolute_distance_for_closest_non_white_confluence_point_for_each_day.pkl",
    #                                                                   f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl',
    #                                                                   )
    # plot_non_white_confluence_areas_and_mark_points_absolute_distance_percentile_between_0_and_30_for_fisher_pivots(df_daily, "NIFc1",
    #                                                                  "NIFc1_percentiles_of_minimum_same_day_ohlc_absolute_distance_for_closest_non_white_confluence_point_for_each_day.pkl",
    #                                                                   f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl',
    #                                                                   260)
    # cache_percentiles_of_minimum_8_absolute_distances_for_closest_non_white_confluence_point_for_each_day("NIFc1",f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl')
    # cache_percentiles_of_minimum_same_day_ohlc_absolute_distances_for_closest_non_white_confluence_point_for_each_day("NIFc1",f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl')
    # cache_percentiles_of_minimum_same_day_ohlc_absolute_distances_and_0_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day("NIFc1",f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl')
    # plot_non_white_confluence_areas_and_mark_points_absolute_distance_percentile_between_0_and_30_and_distance_manually_marked_as_0(df_daily, "NIFc1",
    #                                                                  "NIFc1_percentiles_of_minimum_same_day_ohlc_absolute_distances_and_0_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day.pkl",
    #                                                                   f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl',
    #                                                                   )
    # cache_forward_returns_and_fisher_for_non_white_confluence_areas_and_absolute_distance_percentile_between_0_and_20_and_distance_of_confluence_between_high_low_manually_marked_as_0("NIFc1",
    #                                                                  "NIFc1_percentiles_of_minimum_same_day_ohlc_absolute_distances_and_0_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day.pkl",[10,20,30],["min","max"],260)
    #
    # plot_non_white_confluence_areas_and_mark_points_absolute_distance_percentile_between_0_and_20_and_distance_manually_marked_as_0_and_fisher_score_subplot(df_daily, "NIFc1",
    #                                                                  "NIFc1_percentiles_of_minimum_same_day_ohlc_absolute_distances_and_0_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day.pkl",
    #                                                                   f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl',260)
    # calculate_forward_returns(df_daily,"NIFc1")
    # cache_absolute_percentage_of_poc_and_days_close_across_various_lookbacks("NIFc1")
    # cache_non_absolute_percentage_of_poc_and_days_close_across_various_lookbacks("NIFc1")
    # plot_non_absolute_distance_percentages_of_poc_for_various_lookbacks("NIFc1","NIFc1_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days.pkl")
    # plot_non_white_confluence_areas_and_mark_points_absolute_distance_percentile_between_0_and_20_and_distance_manually_marked_as_0_and_constance_brown_subplot(df_daily, "NIFc1",
    #                                                                  "NIFc1_percentiles_of_minimum_same_day_ohlc_absolute_distances_and_0_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day.pkl",
    #                                                                   f'Confluence_Points_NIFc1_poc_polv_along_with_its_constituents_and_their_lookbacks_with_weighted_confluence_strength_{confluence_area_threshold}_confluence_threshold.pkl')

    # cache_days_with_percentiles_of_minimum_same_day_ohlc_absolute_distances_between_0_and_20_and_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day("NIFc1","NIFc1_percentiles_of_minimum_same_day_ohlc_absolute_distances_and_0_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day.pkl")
    # cache_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days("NIFc1","NIFc1_days_with_percentiles_absolute_distances_between_0_and_20_and_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day.pkl")
    # gmm_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks(df_daily,"NIFc1","NIFc1_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days.pkl")
    df_daily = get_data("NIFc1", "D")
    # for i in tqdm([2016,2019,2021]):
    #     for j in [3,5,7,9,11]:
    #         plot_kmeans_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks(df_daily,
    #                                                                                                      "NIFc1",
    #                                                                                                      "NIFc1_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days.pkl",
    #                                                                                                      j, 2013, i)
    # for i in tqdm([2016,2019,2021]):
    #     for j in [3,5,7,9,11]:
    #         plot_gmm_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks(df_daily,
    #                                                                                                      "NIFc1",
    #                                                                                                      "NIFc1_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days.pkl",
    #                                                                                                      j, 2013, i)
    # for i in tqdm([2016,2019,2021]):
    #     for j in [3,5,7,9,11]:
    #         cache_kmeans_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks(df_daily,
    #                                                                                                      "NIFc1",
    #                                                                                                      "NIFc1_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days.pkl",
    #                                                                                                      j, 2013, i)
    #
    # for i in tqdm([2016,2019,2021]):
    #     for j in [3,5,7,9,11]:
    #         cache_gmm_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks(df_daily,
    #                                                                                                      "NIFc1",
    #                                                                                                      "NIFc1_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days.pkl",
    #                                                                                                      j, 2013, i)
    # plot_kmeans_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_one_chart_for_each_cluster(df_daily,
    #                                                                                                      "NIFc1",
    #                                                                                                      "NIFc1_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days.pkl",
    #                                                                                                      5, 2013, 2021)
    # plot_gmm_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_one_chart_for_each_cluster(
    #     df_daily,
    #     "NIFc1",
    #     "NIFc1_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days.pkl",
    #     5, 2013, 2021)

    # cache_forward_statistics(df_daily,"NIFc1")
    # cache_forward_statistical_analysis_for_points_in_each_cluster("NIFc1","NIFc1_forward_statistics.pkl")
    # cache_top_5_and_bottom_5_of_forward_statistical_analysis_for_points_in_each_cluster("NIFc1",'NIFc1_statistical_analysis_all_combinations_of_clusters.pkl')
    # cache_top_5_and_bottom_5_of_forward_statistical_measures_across_all_cluster_combinations("NIFc1",'NIFc1_statistical_analysis_all_combinations_of_clusters.pkl')
    # cache_out_of_sample_days_which_are_part_of_clusters_of_top_5_and_bottom_5_of_forward_statistical_measures_removing_duplicate_clusters("NIFc1","NIFc1_top_5_and_bottom_5_of_forward_statistical_measures_across_all_cluster_combinations.pkl","NIFc1_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days.pkl")
    # plot_out_of_sample_days_which_are_part_of_clusters_of_top_5_and_bottom_5_of_forward_statistical_measures_removing_duplicate_clusters(df_daily,"NIFc1","NIFc1_out_of_sample_days_which_are_part_of_clusters_of_top_5_and_bottom_5_of_forward_statistical_measures.pkl")
    # cache_out_of_sample_days_which_are_part_of_clusters_of_top_40_and_bottom_40_of_forward_statistical_measures_by_combining_over_look_forwards("NIFc1","NIFc1_top_5_and_bottom_5_of_forward_statistical_measures_across_all_cluster_combinations.pkl","NIFc1_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days.pkl")
    # plot_out_of_sample_days_which_are_part_of_clusters_of_top_40_and_bottom_40_of_forward_statistical_measures_by_combining_over_look_forwards(df_daily,"NIFc1","NIFc1_out_of_sample_days_which_are_part_of_clusters_of_top_40_and_bottom_40_of_forward_statistical_measures_by_combining_over_look_forwards.pkl")
    # cache_indicator_data_for_days_with_percentiles_of_minimum_same_day_ohlc_absolute_distances_between_0_and_20_and_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day(df_daily,
    #                                                                                                                                                                                                   "NIFc1",
    #                                                                                                                                                                                                   "NIFc1_days_with_percentiles_absolute_distances_between_0_and_20_and_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day.pkl",
    #                                                                                                                                                                                                   "NIFc1_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days.pkl")

    # plot_gmm_and_kmeans_scatter_plot_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_and_indicator_values_various_lookbacks(
    #     df_daily,
    #     "NIFc1",
    #     "NIFc1_indicator_data_for_orange_days.pkl",
    #     5, 2013, 2021)
    # plot_kmeans_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_one_chart_for_each_cluster(
    #     df_daily,
    #     "NIFc1",
    #     "NIFc1_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days.pkl",
    #     5, 2013, 2021)
    # plot_gmm_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_one_chart_for_each_cluster(
    #     df_daily,
    #     "NIFc1",
    #     "NIFc1_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days.pkl",
    #     5, 2013, 2021)
    # plot_kmeans_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_and_indicator_values_one_chart_for_each_cluster(df_daily,
    #     "NIFc1",
    #     "NIFc1_indicator_data_for_orange_days.pkl",
    #     5, 2013, 2021)
    # plot_gmm_clustering_for_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_and_indicator_values_one_chart_for_each_cluster(df_daily,
    #     "NIFc1",
    #     "NIFc1_indicator_data_for_orange_days.pkl",
    #     5, 2013, 2021)
    # df_hourly = get_data("NIFc1", "H")
    # vol_features_hourly = return_volume_features_daily_based_on_hourly_candles_single_lookback_which_is_number_of_hourly_candles_for_each_day(df_hourly,"NIFc1")
    # with open(f'VolumeFeatures_Daily_Based_On_Hourly_Candles_Of_That_Day_NIFc1.pkl', 'wb') as file:
    #     pickle.dump(vol_features_hourly, file)

    cache_SVR_regression_for_days_with_percentiles_of_ohlc_absolute_distances_between_0_and_20_and_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day(
        "NIFc1",
        "NIFc1_days_with_percentiles_absolute_distances_between_0_and_20_and_if_closest_is_in_high_and_low_for_closest_non_white_confluence_point_for_each_day.pkl",
        "NIFc1_non_absolute_percentage_of_poc_and_ohlc_avg_across_various_lookbacks_for_0_and_20_distance_percentiles_and_closest_is_in_high_and_low_days.pkl",
        "NIFc1_forward_statistics.pkl")

    end = datetime.datetime.now()
    print(end - start)
    # calculate_confluence_points("VolumeFeatures_NSEI_Daily.pkl")
    # plot_confluence_areas("VolumeFeatures_NSEI_Daily.pkl")
© 2022
GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact
GitHub
Pricing
API
Training
Blog
About
Loading
complete