from calendar import month
from email import message
import pandas as pd
import warnings
import numpy as np
import ast
import importlib.util
import time
from datetime import date,timedelta
from azure.data.tables import TableServiceClient, UpdateMode
from azure.core.credentials import AzureNamedKeyCredential
from tqdm import tqdm
warnings.filterwarnings('ignore')

spec = importlib.util.spec_from_file_location("account_name_and_key", "/nas/Algo/keys_and_passwords/Azure/account_name_and_key.py")
azure_keys = importlib.util.module_from_spec(spec)
spec.loader.exec_module(azure_keys)
_STORAGE_ACCOUNT_NAME = azure_keys._STORAGE_ACCOUNT_NAME
_STORAGE_ACCOUNT_KEY = azure_keys._STORAGE_ACCOUNT_KEY
_TABLE_SERVIVCE_ENDPOINT = azure_keys._TABLE_SERVIVCE_ENDPOINT

spec = importlib.util.spec_from_file_location("ticker_dict", "/nas/Algo/data_retrieval/Tickers/ticker_dict.py")
tickers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tickers)

tickers_all = tickers.tickers_all


def get_reuter_ticker_from_investpy_ticker(investpy_ticker):
    for i in tickers_all:
        if tickers_all[i]["ticker_investpy"] == investpy_ticker:
            return i

    raise f"{investpy_ticker} not found in ticker dictionary please update the dictionary in NAS"


def get_required_data_from_csv(filename=""):
    algo_df = pd.read_csv(filename,index_col="Date")
    portfolio_dict = {}
    constituent_list = [col for col in algo_df.columns if col.endswith("_units") == True ]

    for i in algo_df.index:
        datewise_dict = {}
        for j in constituent_list:
            ticker = get_reuter_ticker_from_investpy_ticker(j.split("_")[0])
            if np.isnan(algo_df[j][i]):
                datewise_dict[ticker] = 0
            else:
                datewise_dict[ticker] = algo_df[j][i] 
        portfolio_dict[i] = datewise_dict

    data = {"Date" : list(algo_df.index),
            "PortfolioValue": list(algo_df["Portfolio Value"])}

    required_dataframe = pd.DataFrame(data)
    required_dataframe.set_index("Date",inplace=True)
    required_dataframe["PortfolioComposition"] = ""

    for i in required_dataframe.index:
        required_dataframe["PortfolioComposition"][i] = portfolio_dict[i]

    # print(required_dataframe)
    # print(portfolio_dict)
    
    # required_dataframe.to_csv("temp.csv")

    return required_dataframe

def get_required_gold_data_from_csv(filename=""):
    algo_df = pd.read_csv(filename,index_col="Date")
    data = {}
    
    if "Portfolio Value" in algo_df.columns:
        data = {"Date" : list(algo_df.index),
            "PortfolioValue": list(algo_df["Portfolio Value"])}
    elif "Strategy_Return" in algo_df.columns:
        data = {"Date" : list(algo_df.index),
            "PortfolioValue": list(algo_df["Strategy_Return"])}

    required_dataframe = pd.DataFrame(data)
    required_dataframe.set_index("Date",inplace=True)

    return required_dataframe

def get_required_ixic_data_from_csv(filename=""):
    algo_df = pd.read_csv(filename,index_col="Date")
    algo_df["IXIC_Value_USD"] = algo_df["Strategy_Return"]/algo_df["USDINR"]

    data = {"Date" : list(algo_df.index),
            "IXIC_Value_INR" : list(algo_df["Strategy_Return"]),
            "IXIC_Value_USD" : list(algo_df["IXIC_Value_USD"])}

    required_dataframe = pd.DataFrame(data)
    required_dataframe.set_index("Date",inplace=True)

    return required_dataframe

def upload_nifty_and_midcap_first_time():
    # date_today = date.today().strftime("%Y-%m-%d")
    credential = AzureNamedKeyCredential(_STORAGE_ACCOUNT_NAME, _STORAGE_ACCOUNT_KEY)
    table_service = TableServiceClient(endpoint=_TABLE_SERVIVCE_ENDPOINT,credential=credential)
    table_name = 'PortfolioValuesOfAlpha'
    table_service.create_table_if_not_exists(table_name=table_name)
    table_client = table_service.get_table_client(table_name=table_name)

    csv_files = ["Nifty50_NonNaive.csv","Nifty50Naive.csv","MidcapNaive.csv","Midcap50NonNaive.csv"]

    for file in csv_files:
        df = get_required_data_from_csv(file)
        partition_key = file.split(".")[0]
        for row in tqdm(df.index):
            temp_dict = {"PartitionKey": partition_key,
                        "RowKey": str(row),
                        "Date": str(row),
                        "PortfolioComposition": str(df["PortfolioComposition"][row]),
                        "PortfolioValue": str(df["PortfolioValue"][row])
                        }
            table_client.create_entity(temp_dict)

    return

def upload_gold_first_time():
    credential = AzureNamedKeyCredential(_STORAGE_ACCOUNT_NAME, _STORAGE_ACCOUNT_KEY)
    table_service = TableServiceClient(endpoint=_TABLE_SERVIVCE_ENDPOINT,credential=credential)
    table_name = 'PortfolioValuesOfAlpha'
    table_service.create_table_if_not_exists(table_name=table_name)
    table_client = table_service.get_table_client(table_name=table_name)

    csv_files = ["Gold_Alpha_Midcap_Naive.csv","Gold_Alpha_Midcap_Non_Naive.csv","Gold_Alpha_Nifty_Naive.csv","Gold_Alpha_Nifty_Non_Naive.csv"]

    for file in csv_files:
        df = get_required_gold_data_from_csv(file)
        # print(df)
        partition_key = file.split(".")[0]
        for row in tqdm(df.index):
            temp_dict = {"PartitionKey": partition_key,
                        "RowKey": str(row),
                        "Date": str(row),
                        "PortfolioValue": str(df["PortfolioValue"][row]),
                        }
            table_client.create_entity(temp_dict)

    return

def upload_ixic_first_time():
    # date_today = date.today().strftime("%Y-%m-%d")
    credential = AzureNamedKeyCredential(_STORAGE_ACCOUNT_NAME, _STORAGE_ACCOUNT_KEY)
    table_service = TableServiceClient(endpoint=_TABLE_SERVIVCE_ENDPOINT,credential=credential)
    table_name = 'PortfolioValuesOfAlpha'
    table_service.create_table_if_not_exists(table_name=table_name)
    table_client = table_service.get_table_client(table_name=table_name)

    csv_files = ["^IXIC.csv"]

    for file in csv_files:
        df = get_required_ixic_data_from_csv(file)
        # print(df)
        for row in tqdm(df.index):
            temp_dict = {"PartitionKey": "^IXIC",
                        "RowKey": str(row),
                        "Date": str(row),
                        "IXIC_Value_INR": str(df["IXIC_Value_INR"][row]),
                        "IXIC_Value_USD": str(df["IXIC_Value_USD"][row])
                        }
            table_client.create_entity(temp_dict)

    return

def update_nifty_midcap_daily(file):
    status_message = ""

    credential = AzureNamedKeyCredential(_STORAGE_ACCOUNT_NAME, _STORAGE_ACCOUNT_KEY)
    table_service = TableServiceClient(endpoint=_TABLE_SERVIVCE_ENDPOINT,credential=credential)
    table_name = 'PortfolioValuesOfAlpha'
    table_service.create_table_if_not_exists(table_name=table_name)
    table_client = table_service.get_table_client(table_name=table_name)
    
    try:
        df = get_required_data_from_csv(file)
        df = df[-70:]
        partition_key = file.split(".")[0]
        for row in df.index:
            temp_dict = {"PartitionKey": partition_key,
                        "RowKey": str(row),
                        "Date": str(row),
                        "PortfolioComposition": str(df["PortfolioComposition"][row]),
                        "PortfolioValue": str(df["PortfolioValue"][row])
                        }
            table_client.upsert_entity(temp_dict)

            status_message = "Azure update of Table - PortfolioValuesOfAlpha done successfully"
    except Exception as err:
        status_message = str(err)
    return status_message

def update_ixic_daily(file):
    status_message = ""

    credential = AzureNamedKeyCredential(_STORAGE_ACCOUNT_NAME, _STORAGE_ACCOUNT_KEY)
    table_service = TableServiceClient(endpoint=_TABLE_SERVIVCE_ENDPOINT,credential=credential)
    table_name = 'PortfolioValuesOfAlpha'
    table_service.create_table_if_not_exists(table_name=table_name)
    table_client = table_service.get_table_client(table_name=table_name)
    
    try:
        df = get_required_ixic_data_from_csv(file)
        df = df[-70:]
        for row in df.index:
            temp_dict = {"PartitionKey": "^IXIC",
                        "RowKey": str(row),
                        "Date": str(row),
                        "IXIC_Value_INR": str(df["IXIC_Value_INR"][row]),
                        "IXIC_Value_USD": str(df["IXIC_Value_USD"][row])
                        }
            table_client.upsert_entity(temp_dict)

            status_message = "Azure update of Table - PortfolioValuesOfAlpha done successfully"
    except Exception as err:
        status_message = str(err)

    return status_message

def update_gold_daily(file):
    status_message = ""

    credential = AzureNamedKeyCredential(_STORAGE_ACCOUNT_NAME, _STORAGE_ACCOUNT_KEY)
    table_service = TableServiceClient(endpoint=_TABLE_SERVIVCE_ENDPOINT,credential=credential)
    table_name = 'PortfolioValuesOfAlpha'
    table_service.create_table_if_not_exists(table_name=table_name)
    table_client = table_service.get_table_client(table_name=table_name)

    try:
        df = get_required_gold_data_from_csv(file)
        df = df[-70:]
        partition_key = file.split(".")[0]
        for row in tqdm(df.index):
            temp_dict = {"PartitionKey": partition_key,
                        "RowKey": str(row),
                        "Date": str(row),
                        "PortfolioValue": str(df["PortfolioValue"][row]),
                        }
            table_client.upsert_entity(temp_dict)
        
        status_message = "Azure update of Table - PortfolioValuesOfAlpha for Gold Alpha done successfully"
    except Exception as err:
        status_message = str(err)

    return status_message

#Deprecated don't use  
def update_portfolio_values_table_daily(file):
    status_message = ""

    credential = AzureNamedKeyCredential(_STORAGE_ACCOUNT_NAME, _STORAGE_ACCOUNT_KEY)
    table_service = TableServiceClient(endpoint=_TABLE_SERVIVCE_ENDPOINT,credential=credential)
    table_name = 'PortfolioValuesOfAlpha'
    table_service.create_table_if_not_exists(table_name=table_name)
    table_client = table_service.get_table_client(table_name=table_name)

    try:
        df = get_required_data_from_csv(file)
        df = df[-70:]
        partition_key = file.split(".")[0]
        for row in df.index:
            temp_dict = {"PartitionKey": partition_key,
                        "RowKey": str(row),
                        "Date": str(row),
                        "PortfolioComposition": str(df["PortfolioComposition"][row]),
                        "PortfolioValue": str(df["PortfolioValue"][row])
                        }
            table_client.upsert_entity(temp_dict)
        
        status_message = "Azure update of Table - PortfolioValuesOfAlpha done successfully"

    except Exception as err:
        status_message = str(err)

    return status_message


# get_required_data_from_csv("MidcapNaive.csv")
# upload_nifty_and_midcap_first_time()
# print(update_portfolio_values_table_daily("Nifty50_NonNaive.csv"))
# upload_ixic_first_time()
# upload_gold_first_time()
# testing undo commit