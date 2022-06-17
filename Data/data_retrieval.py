# This file contains code that retrieves data stored in Azure Datatables.
# This wrapper can be used to return data for both daily and hourly frequencies.
# Starting date is determined by the dictionary provided in tickers.py

# Requirements: importlib, azure_data_tables==12.2.0, azure==4.0.0
import investpy
import pandas as pd
import importlib.util
spec = importlib.util.spec_from_file_location("account_name_and_key",
                                                  "Z:/Algo/keys_and_passwords/Azure/account_name_and_key.py")
azure_keys = importlib.util.module_from_spec(spec)
spec.loader.exec_module(azure_keys)
_STORAGE_ACCOUNT_NAME = azure_keys._STORAGE_ACCOUNT_NAME
_STORAGE_ACCOUNT_KEY = azure_keys._STORAGE_ACCOUNT_KEY

from azure.data.tables import TableServiceClient
from azure.core.credentials import AzureNamedKeyCredential

def get_data(ticker, frequency):
    """
    :param ticker: Ticker as on Reuters. Investpy and yfinance tickers can be passed using the lookup dict in tickers.py
    :param frequency: Frequency of the data required. Currently supports daily and hourly. Pass "D" or "H"
    :return:  Returns the OHLCV dataframe indexed by datetime
    """

    credential = AzureNamedKeyCredential(_STORAGE_ACCOUNT_NAME, _STORAGE_ACCOUNT_KEY)
    table_service = TableServiceClient(endpoint="https://acsysbatchstroageacc.table.core.windows.net/",
                                       credential=credential)
    if frequency=='H':
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
    ticker_dataframe[["Open", "High", "Low", "Close", "Volume"]] = ticker_dataframe[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    if 'Date' in ticker_dataframe.columns:
        ticker_dataframe.rename(columns={'Date': 'Datetime'}, inplace=True)
    ticker_dataframe["Datetime"] = pd.to_datetime(ticker_dataframe["Datetime"])
    return ticker_dataframe

def resample_data(ohlcv, minutes):
    """
    :param ohlcv: Pass minute-level OHLCV data to be resampled
    :param minutes: Pass the number of minutes of the bar to be resampled to. Eg For hourly bars, pass 60
    :return: Returns the resampled data
    """
    return ohlcv.set_index("Datetime").groupby(pd.Grouper(freq=f'{minutes}Min')).agg({"Open": "first",
                                                 "Close": "last",
                                                 "Low": "min",
                                                 "High": "max",
                                                "Volume": "sum"}).reset_index()


def get_data_investpy( symbol, country, from_date, to_date ):
  find = investpy.search.search_quotes(text=symbol, products =["stocks", "etfs", "indices", "currencies"] )
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

  if ret is None:
    try:
      ret = investpy.currency_crosses.get_currency_cross_historical_data(currency_cross=symbol,
                                      from_date=from_date,
                                      to_date=to_date)
    except:
      ret = None
  ret.drop(["Change Pct"], axis=1, inplace=True)
  return ret