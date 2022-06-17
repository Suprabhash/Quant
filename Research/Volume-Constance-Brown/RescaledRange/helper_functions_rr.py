import numpy as np
import ssl
import pandas as pd
import datetime
import pickle
import os

def get_data_BTC():
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

    return temp_og, temp_og1

def to_pct(x):
    pcts = x[1:] / x[:-1] - 1.
    return pcts


