#%%
import investpy
from datetime import date, datetime
import math
import numpy as np
import pandas as pd
from MCMC.MCMC import MCMC
import datetime as dt
import dateutil.relativedelta
import multiprocessing
import pickle
# import eikon as ek
import yfinance as yf
from datetime import timedelta
import os
import uuid
import sys
from azure.storage.blob import BlockBlobService, PublicAccess
import zipfile
from azure.data.tables import TableServiceClient
from azure.core.credentials import AzureNamedKeyCredential

import importlib.util

spec = importlib.util.spec_from_file_location("account_name_and_key", "/nas/Algo/keys_and_passwords/Azure/account_name_and_key.py")
azure_keys = importlib.util.module_from_spec(spec)
spec.loader.exec_module(azure_keys)
_STORAGE_ACCOUNT_NAME = azure_keys._STORAGE_ACCOUNT_NAME
_STORAGE_ACCOUNT_KEY = azure_keys._STORAGE_ACCOUNT_KEY

tickers_all = {
   ".NSEI": {
      "ticker_investpy": "NSEI",
      "ticker_yfinance": "^NSEI",
      "Start_Date": "2007-09-17",
      "exchange_name": "NSE"
   },
   "GBES.NS": {
      "ticker_investpy": "NA",
      "ticker_yfinance": "GOLDBEES.NS",
      "Start_Date": "2009-01-02",
      "exchange_name": "NSE"
   },
   "TLT.OQ": {
      "ticker_investpy": "TLT",
      "ticker_yfinance": "TLT",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   ".IXIC": {
      "ticker_investpy": "IXIC",
      "ticker_yfinance": "^IXIC",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   ".NIMDCP50": {
      "ticker_investpy": "NIMDCP50",
      "ticker_yfinance": "^NSEMDCP50",
      "Start_Date": "2007-09-24",
      "exchange_name": "NSE"
   },
   "ROST.OQ": {
      "ticker_investpy": "ROST",
      "ticker_yfinance": "ROST",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "MNST.OQ": {
      "ticker_investpy": "MNST",
      "ticker_yfinance": "MNST",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "CMCSA.OQ": {
      "ticker_investpy": "CMCSA",
      "ticker_yfinance": "CMCSA",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "KLAC.OQ": {
      "ticker_investpy": "KLAC",
      "ticker_yfinance": "KLAC",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "NXPI.OQ": {
      "ticker_investpy": "NXPI",
      "ticker_yfinance": "NXPI",
      "Start_Date": "2010-08-09",
      "exchange_name": "NASDAQ"
   },
   "XLNX.OQ": {
      "ticker_investpy": "XLNX",
      "ticker_yfinance": "XLNX",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "ALGN.OQ": {
      "ticker_investpy": "ALGN",
      "ticker_yfinance": "ALGN",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "MRVL.OQ": {
      "ticker_investpy": "MRVL",
      "ticker_yfinance": "MRVL",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "ISRG.OQ": {
      "ticker_investpy": "ISRG",
      "ticker_yfinance": "ISRG",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "MAT.OQ": {
      "ticker_investpy": "MAT",
      "ticker_yfinance": "MAT",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "OKTA.OQ": {
      "ticker_investpy": "OKTA",
      "ticker_yfinance": "OKTA",
      "Start_Date": "2017-04-10",
      "exchange_name": "NASDAQ"
   },
   "AVGO.OQ": {
      "ticker_investpy": "AVGO",
      "ticker_yfinance": "AVGO",
      "Start_Date": "2009-08-06",
      "exchange_name": "NASDAQ"
   },
   "DXCM.OQ": {
      "ticker_investpy": "DXCM",
      "ticker_yfinance": "DXCM",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "AMD.OQ": {
      "ticker_investpy": "AMD",
      "ticker_yfinance": "AMD",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "DOCU.OQ": {
      "ticker_investpy": "DOCU",
      "ticker_yfinance": "DOCU",
      "Start_Date": "2018-04-30",
      "exchange_name": "NASDAQ"
   },
   "INTC.OQ": {
      "ticker_investpy": "INTC",
      "ticker_yfinance": "INTC",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "UAL.OQ": {
      "ticker_investpy": "UAL",
      "ticker_yfinance": "UAL",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "KDP.OQ": {
      "ticker_investpy": "KDP",
      "ticker_yfinance": "KDP",
      "Start_Date": "2008-04-28",
      "exchange_name": "NASDAQ"
   },
   "WBA.OQ": {
      "ticker_investpy": "WBA",
      "ticker_yfinance": "WBA",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "CSCO.OQ": {
      "ticker_investpy": "CSCO",
      "ticker_yfinance": "CSCO",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "SIRI.OQ": {
      "ticker_investpy": "SIRI",
      "ticker_yfinance": "SIRI",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "LRCX.OQ": {
      "ticker_investpy": "LRCX",
      "ticker_yfinance": "LRCX",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "GILD.OQ": {
      "ticker_investpy": "GILD",
      "ticker_yfinance": "GILD",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "ADP.OQ": {
      "ticker_investpy": "ADP",
      "ticker_yfinance": "ADP",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "NLOK.OQ": {
      "ticker_investpy": "NLOK",
      "ticker_yfinance": "NLOK",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "ADSK.OQ": {
      "ticker_investpy": "ADSK",
      "ticker_yfinance": "ADSK",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "AMZN.OQ": {
      "ticker_investpy": "AMZN",
      "ticker_yfinance": "AMZN",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "QRTEA.OQ": {
      "ticker_investpy": "QRTEA",
      "ticker_yfinance": "QRTEA",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "REGN.OQ": {
      "ticker_investpy": "REGN",
      "ticker_yfinance": "REGN",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "LBTYA.OQ": {
      "ticker_investpy": "LBTYA",
      "ticker_yfinance": "LBTYA",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "TMUS.OQ": {
      "ticker_investpy": "TMUS",
      "ticker_yfinance": "TMUS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "LULU.OQ": {
      "ticker_investpy": "LULU",
      "ticker_yfinance": "LULU",
      "Start_Date": "2007-07-30",
      "exchange_name": "NASDAQ"
   },
   "SGEN.OQ": {
      "ticker_investpy": "SGEN",
      "ticker_yfinance": "SGEN",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "MDLZ.OQ": {
      "ticker_investpy": "MDLZ",
      "ticker_yfinance": "MDLZ",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "INCY.OQ": {
      "ticker_investpy": "INCY",
      "ticker_yfinance": "INCY",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "TCOM.OQ": {
      "ticker_investpy": "TCOM",
      "ticker_yfinance": "TCOM",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "STX.OQ": {
      "ticker_investpy": "STX",
      "ticker_yfinance": "STX",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "CDNS.OQ": {
      "ticker_investpy": "CDNS",
      "ticker_yfinance": "CDNS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "NTAP.OQ": {
      "ticker_investpy": "NTAP",
      "ticker_yfinance": "NTAP",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "HAS.OQ": {
      "ticker_investpy": "HAS",
      "ticker_yfinance": "HAS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "CHTR.OQ": {
      "ticker_investpy": "CHTR",
      "ticker_yfinance": "CHTR",
      "Start_Date": "2009-12-02",
      "exchange_name": "NASDAQ"
   },
   "ILMN.OQ": {
      "ticker_investpy": "ILMN",
      "ticker_yfinance": "ILMN",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "SBUX.OQ": {
      "ticker_investpy": "SBUX",
      "ticker_yfinance": "SBUX",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "PYPL.OQ": {
      "ticker_investpy": "PYPL",
      "ticker_yfinance": "PYPL",
      "Start_Date": "2015-07-07",
      "exchange_name": "NASDAQ"
   },
   "EBAY.OQ": {
      "ticker_investpy": "EBAY",
      "ticker_yfinance": "EBAY",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "AMGN.OQ": {
      "ticker_investpy": "AMGN",
      "ticker_yfinance": "AMGN",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "TEAM.OQ": {
      "ticker_investpy": "TEAM",
      "ticker_yfinance": "TEAM",
      "Start_Date": "2015-12-11",
      "exchange_name": "NASDAQ"
   },
   "MCHP.OQ": {
      "ticker_investpy": "MCHP",
      "ticker_yfinance": "MCHP",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "BIDU.OQ": {
      "ticker_investpy": "BIDU",
      "ticker_yfinance": "BIDU",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "NCLH.N": {
      "ticker_investpy": "NCLH",
      "ticker_yfinance": "NCLH",
      "Start_Date": "2013-01-18",
      "exchange_name": "NYSE"
   },
   "EA.OQ": {
      "ticker_investpy": "EA",
      "ticker_yfinance": "EA",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "XEL.OQ": {
      "ticker_investpy": "XEL",
      "ticker_yfinance": "XEL",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "CERN.OQ": {
      "ticker_investpy": "CERN",
      "ticker_yfinance": "CERN",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "CDW.OQ": {
      "ticker_investpy": "CDW",
      "ticker_yfinance": "CDW",
      "Start_Date": "2013-06-27",
      "exchange_name": "NASDAQ"
   },
   "AMAT.OQ": {
      "ticker_investpy": "AMAT",
      "ticker_yfinance": "AMAT",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "CPRT.OQ": {
      "ticker_investpy": "CPRT",
      "ticker_yfinance": "CPRT",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "BKNG.OQ": {
      "ticker_investpy": "BKNG",
      "ticker_yfinance": "BKNG",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "CTSH.OQ": {
      "ticker_investpy": "CTSH",
      "ticker_yfinance": "CTSH",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "AEP.OQ": {
      "ticker_investpy": "AEP",
      "ticker_yfinance": "AEP",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "CHKP.OQ": {
      "ticker_investpy": "CHKP",
      "ticker_yfinance": "CHKP",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "PEP.OQ": {
      "ticker_investpy": "PEP",
      "ticker_yfinance": "PEP",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "FB.OQ": {
      "ticker_investpy": "FB",
      "ticker_yfinance": "FB",
      "Start_Date": "2012-05-18",
      "exchange_name": "NASDAQ"
   },
   "JD.OQ": {
      "ticker_investpy": "JD",
      "ticker_yfinance": "JD",
      "Start_Date": "2014-05-23",
      "exchange_name": "NASDAQ"
   },
   "ANSS.OQ": {
      "ticker_investpy": "ANSS",
      "ticker_yfinance": "ANSS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "VTRS.OQ": {
      "ticker_investpy": "VTRS",
      "ticker_yfinance": "VTRS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "INTU.OQ": {
      "ticker_investpy": "INTU",
      "ticker_yfinance": "INTU",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "LILA.OQ": {
      "ticker_investpy": "LILA",
      "ticker_yfinance": "LILA",
      "Start_Date": "2015-07-01",
      "exchange_name": "NASDAQ"
   },
   "CSGP.OQ": {
      "ticker_investpy": "CSGP",
      "ticker_yfinance": "CSGP",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "NVDA.OQ": {
      "ticker_investpy": "NVDA",
      "ticker_yfinance": "NVDA",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "GOOGL.OQ": {
      "ticker_investpy": "GOOGL",
      "ticker_yfinance": "GOOGL",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "VOD.OQ": {
      "ticker_investpy": "VOD",
      "ticker_yfinance": "VOD",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "NFLX.OQ": {
      "ticker_investpy": "NFLX",
      "ticker_yfinance": "NFLX",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "JBHT.OQ": {
      "ticker_investpy": "JBHT",
      "ticker_yfinance": "JBHT",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "XRAY.OQ": {
      "ticker_investpy": "XRAY",
      "ticker_yfinance": "XRAY",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "DLTR.OQ": {
      "ticker_investpy": "DLTR",
      "ticker_yfinance": "DLTR",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "VRTX.OQ": {
      "ticker_investpy": "VRTX",
      "ticker_yfinance": "VRTX",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "COST.OQ": {
      "ticker_investpy": "COST",
      "ticker_yfinance": "COST",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "IDXX.OQ": {
      "ticker_investpy": "IDXX",
      "ticker_yfinance": "IDXX",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "TTWO.OQ": {
      "ticker_investpy": "TTWO",
      "ticker_yfinance": "TTWO",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "FISV.OQ": {
      "ticker_investpy": "FISV",
      "ticker_yfinance": "FISV",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "AKAM.OQ": {
      "ticker_investpy": "AKAM",
      "ticker_yfinance": "AKAM",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "ADBE.OQ": {
      "ticker_investpy": "ADBE",
      "ticker_yfinance": "ADBE",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "NTES.OQ": {
      "ticker_investpy": "NTES",
      "ticker_yfinance": "NTES",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "BIIB.OQ": {
      "ticker_investpy": "BIIB",
      "ticker_yfinance": "BIIB",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "SWKS.OQ": {
      "ticker_investpy": "SWKS",
      "ticker_yfinance": "SWKS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "SNPS.OQ": {
      "ticker_investpy": "SNPS",
      "ticker_yfinance": "SNPS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "AAL.OQ": {
      "ticker_investpy": "AAL",
      "ticker_yfinance": "AAL",
      "Start_Date": "2013-12-09",
      "exchange_name": "NASDAQ"
   },
   "EXC.OQ": {
      "ticker_investpy": "EXC",
      "ticker_yfinance": "EXC",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "DISH.OQ": {
      "ticker_investpy": "DISH",
      "ticker_yfinance": "DISH",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "MU.OQ": {
      "ticker_investpy": "MU",
      "ticker_yfinance": "MU",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "VRSN.OQ": {
      "ticker_investpy": "VRSN",
      "ticker_yfinance": "VRSN",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "QCOM.OQ": {
      "ticker_investpy": "QCOM",
      "ticker_yfinance": "QCOM",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "TSCO.OQ": {
      "ticker_investpy": "TSCO",
      "ticker_yfinance": "TSCO",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "MELI.OQ": {
      "ticker_investpy": "MELI",
      "ticker_yfinance": "MELI",
      "Start_Date": "2007-08-13",
      "exchange_name": "NASDAQ"
   },
   "HOLX.OQ": {
      "ticker_investpy": "HOLX",
      "ticker_yfinance": "HOLX",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "WYNN.OQ": {
      "ticker_investpy": "WYNN",
      "ticker_yfinance": "WYNN",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "EXPE.OQ": {
      "ticker_investpy": "EXPE",
      "ticker_yfinance": "EXPE",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "BMRN.OQ": {
      "ticker_investpy": "BMRN",
      "ticker_yfinance": "BMRN",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "FAST.OQ": {
      "ticker_investpy": "FAST",
      "ticker_yfinance": "FAST",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "ASML.OQ": {
      "ticker_investpy": "ASML",
      "ticker_yfinance": "ASML",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "TSLA.OQ": {
      "ticker_investpy": "TSLA",
      "ticker_yfinance": "TSLA",
      "Start_Date": "2010-06-30",
      "exchange_name": "NASDAQ"
   },
   "KHC.OQ": {
      "ticker_investpy": "KHC",
      "ticker_yfinance": "KHC",
      "Start_Date": "2009-01-23",
      "exchange_name": "NASDAQ"
   },
   "MSFT.OQ": {
      "ticker_investpy": "MSFT",
      "ticker_yfinance": "MSFT",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "ORLY.OQ": {
      "ticker_investpy": "ORLY",
      "ticker_yfinance": "ORLY",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "PAYX.OQ": {
      "ticker_investpy": "PAYX",
      "ticker_yfinance": "PAYX",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "CTXS.OQ": {
      "ticker_investpy": "CTXS",
      "ticker_yfinance": "CTXS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "PCAR.OQ": {
      "ticker_investpy": "PCAR",
      "ticker_yfinance": "PCAR",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "ULTA.OQ": {
      "ticker_investpy": "ULTA",
      "ticker_yfinance": "ULTA",
      "Start_Date": "2007-10-26",
      "exchange_name": "NASDAQ"
   },
   "CSX.OQ": {
      "ticker_investpy": "CSX",
      "ticker_yfinance": "CSX",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "DISCA.OQ": {
      "ticker_investpy": "DISCA",
      "ticker_yfinance": "DISCA",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "WDC.OQ": {
      "ticker_investpy": "WDC",
      "ticker_yfinance": "WDC",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "ADI.OQ": {
      "ticker_investpy": "ADI",
      "ticker_yfinance": "ADI",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "HSIC.OQ": {
      "ticker_investpy": "HSIC",
      "ticker_yfinance": "HSIC",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "TXN.OQ": {
      "ticker_investpy": "TXN",
      "ticker_yfinance": "TXN",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "ATVI.OQ": {
      "ticker_investpy": "ATVI",
      "ticker_yfinance": "ATVI",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "AAPL.OQ": {
      "ticker_investpy": "AAPL",
      "ticker_yfinance": "AAPL",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "CTAS.OQ": {
      "ticker_investpy": "CTAS",
      "ticker_yfinance": "CTAS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "SPLK.OQ": {
      "ticker_investpy": "SPLK",
      "ticker_yfinance": "SPLK",
      "Start_Date": "2012-04-20",
      "exchange_name": "NASDAQ"
   },
   "HON.OQ": {
      "ticker_investpy": "HON",
      "ticker_yfinance": "HON",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "VRSK.OQ": {
      "ticker_investpy": "VRSK",
      "ticker_yfinance": "VRSK",
      "Start_Date": "2009-10-08",
      "exchange_name": "NASDAQ"
   },
   "PDD.OQ": {
      "ticker_investpy": "PDD",
      "ticker_yfinance": "PDD",
      "Start_Date": "2018-07-27",
      "exchange_name": "NASDAQ"
   },
   "TRIP.OQ": {
      "ticker_investpy": "TRIP",
      "ticker_yfinance": "TRIP",
      "Start_Date": "2011-12-08",
      "exchange_name": "NASDAQ"
   },
   "MTCH.OQ": {
      "ticker_investpy": "MTCH",
      "ticker_yfinance": "MTCH",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "WDAY.OQ": {
      "ticker_investpy": "WDAY",
      "ticker_yfinance": "WDAY",
      "Start_Date": "2012-10-15",
      "exchange_name": "NASDAQ"
   },
   "SBAC.OQ": {
      "ticker_investpy": "SBAC",
      "ticker_yfinance": "SBAC",
      "Start_Date": "2007-07-01",
      "exchange_name": "NASDAQ"
   },
   "TAMO.NS": {
      "ticker_investpy": "TAMO",
      "ticker_yfinance": "TATAMOTORS.NS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "SBI.NS": {
      "ticker_investpy": "SBI",
      "ticker_yfinance": "SBI",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "NEST.NS": {
      "ticker_investpy": "NEST",
      "ticker_yfinance": "NEST",
      "Start_Date": "2010-01-11",
      "exchange_name": "NSE"
   },
   "INFY.NS": {
      "ticker_investpy": "INFY",
      "ticker_yfinance": "INFY",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "TCS.NS": {
      "ticker_investpy": "TCS",
      "ticker_yfinance": "TCS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "COAL.NS": {
      "ticker_investpy": "COAL",
      "ticker_yfinance": "COAL",
      "Start_Date": "2010-11-05",
      "exchange_name": "NSE"
   },
   "HCLT.NS": {
      "ticker_investpy": "HCLT",
      "ticker_yfinance": "HCLT",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "NTPC.NS": {
      "ticker_investpy": "NTPC",
      "ticker_yfinance": "NTPC",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "ICBK.NS": {
      "ticker_investpy": "ICBK",
      "ticker_yfinance": "ICBK",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "LART.NS": {
      "ticker_investpy": "LART",
      "ticker_yfinance": "LART",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "HDBK.NS": {
      "ticker_investpy": "HDBK",
      "ticker_yfinance": "HDBK",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "TISC.NS": {
      "ticker_investpy": "TISC",
      "ticker_yfinance": "TISC",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "BAJA.NS": {
      "ticker_investpy": "BAJA",
      "ticker_yfinance": "BAJA",
      "Start_Date": "2008-05-27",
      "exchange_name": "NSE"
   },
   "ASPN.NS": {
      "ticker_investpy": "ASPN",
      "ticker_yfinance": "ASPN",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "REDY.NS": {
      "ticker_investpy": "REDY",
      "ticker_yfinance": "REDY",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "TEML.NS": {
      "ticker_investpy": "TEML",
      "ticker_yfinance": "TEML",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "CIPL.NS": {
      "ticker_investpy": "CIPL",
      "ticker_yfinance": "CIPL",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "ULTC.NS": {
      "ticker_investpy": "ULTC",
      "ticker_yfinance": "ULTC",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "BJFS.NS": {
      "ticker_investpy": "BJFS",
      "ticker_yfinance": "BJFS",
      "Start_Date": "2008-05-27",
      "exchange_name": "NSE"
   },
   "HDFC.NS": {
      "ticker_investpy": "HDFC",
      "ticker_yfinance": "HDFC",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "SUN.NS": {
      "ticker_investpy": "SUN",
      "ticker_yfinance": "SUN",
      "Start_Date": "2010-05-21",
      "exchange_name": "NSE"
   },
   "ITC.NS": {
      "ticker_investpy": "ITC",
      "ticker_yfinance": "ITC",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "WIPR.NS": {
      "ticker_investpy": "WIPR",
      "ticker_yfinance": "WIPR",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "GAIL.NS": {
      "ticker_investpy": "GAIL",
      "ticker_yfinance": "GAIL",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "VDAN.NS": {
      "ticker_investpy": "VDAN",
      "ticker_yfinance": "VDAN",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "PGRD.NS": {
      "ticker_investpy": "PGRD",
      "ticker_yfinance": "PGRD",
      "Start_Date": "2007-10-08",
      "exchange_name": "NSE"
   },
   "HROM.NS": {
      "ticker_investpy": "HROM",
      "ticker_yfinance": "HROM",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "AXBK.NS": {
      "ticker_investpy": "AXBK",
      "ticker_yfinance": "AXBK",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "YESB.NS": {
      "ticker_investpy": "YESB",
      "ticker_yfinance": "YESB",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "ONGC.NS": {
      "ticker_investpy": "ONGC",
      "ticker_yfinance": "ONGC",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "HLL.NS": {
      "ticker_investpy": "HLL",
      "ticker_yfinance": "HLL",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "APSE.NS": {
      "ticker_investpy": "APSE",
      "ticker_yfinance": "APSE",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "BRTI.NS": {
      "ticker_investpy": "BRTI",
      "ticker_yfinance": "BRTI",
      "Start_Date": "2017-07-25",
      "exchange_name": "NSE"
   },
   "VODA.NS": {
      "ticker_investpy": "VODA",
      "ticker_yfinance": "VODA",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "BFRG.NS": {
      "ticker_investpy": "BFRG",
      "ticker_yfinance": "BFRG",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "CUMM.NS": {
      "ticker_investpy": "CUMM",
      "ticker_yfinance": "CUMM",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "CAST.NS": {
      "ticker_investpy": "CAST",
      "ticker_yfinance": "CAST",
      "Start_Date": "2007-08-16",
      "exchange_name": "NSE"
   },
   "ASOK.NS": {
      "ticker_investpy": "ASOK",
      "ticker_yfinance": "ASOK",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "AUFI.NS": {
      "ticker_investpy": "AUFI",
      "ticker_yfinance": "AUFI",
      "Start_Date": "2017-07-10",
      "exchange_name": "NSE"
   },
   "SRTR.NS": {
      "ticker_investpy": "SRTR",
      "ticker_yfinance": "SRTR",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "MAXI.NS": {
      "ticker_investpy": "MAXI",
      "ticker_yfinance": "MAXI",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "BATA.NS": {
      "ticker_investpy": "BATA",
      "ticker_yfinance": "BATA",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "MINT.NS": {
      "ticker_investpy": "MINT",
      "ticker_yfinance": "MINT",
      "Start_Date": "2009-11-18",
      "exchange_name": "NSE"
   },
   "COFO.NS": {
      "ticker_investpy": "COFO",
      "ticker_yfinance": "COFO",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "TVSM.NS": {
      "ticker_investpy": "TVSM",
      "ticker_yfinance": "TVSM",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "PAGE.NS": {
      "ticker_investpy": "PAGE",
      "ticker_yfinance": "PAGE",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "CCRI.NS": {
      "ticker_investpy": "CCRI",
      "ticker_yfinance": "CCRI",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "ESCO.NS": {
      "ticker_investpy": "ESCO",
      "ticker_yfinance": "ESCO",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "SRFL.NS": {
      "ticker_investpy": "SRFL",
      "ticker_yfinance": "SRFL",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "CNBK.NS": {
      "ticker_investpy": "CNBK",
      "ticker_yfinance": "CNBK",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "TTPW.NS": {
      "ticker_investpy": "TTPW",
      "ticker_yfinance": "TTPW",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "ZEE.NS": {
      "ticker_investpy": "ZEE",
      "ticker_yfinance": "ZEE",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "MNFL.NS": {
      "ticker_investpy": "MNFL",
      "ticker_yfinance": "MNFL",
      "Start_Date": "2010-07-01",
      "exchange_name": "NSE"
   },
   "FED.NS": {
      "ticker_investpy": "FED",
      "ticker_yfinance": "FED",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "GLEN.NS": {
      "ticker_investpy": "GLEN",
      "ticker_yfinance": "GLEN",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "CHLA.NS": {
      "ticker_investpy": "CHLA",
      "ticker_yfinance": "CHLA",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "AMAR.NS": {
      "ticker_investpy": "AMAR",
      "ticker_yfinance": "AMAR",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "APLO.NS": {
      "ticker_investpy": "APLO",
      "ticker_yfinance": "APLO",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "BAJE.NS": {
      "ticker_investpy": "BAJE",
      "ticker_yfinance": "BAJE",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "SAIL.NS": {
      "ticker_investpy": "SAIL",
      "ticker_yfinance": "SAIL",
      "Start_Date": "2017-11-17",
      "exchange_name": "NSE"
   },
   "MMFS.NS": {
      "ticker_investpy": "MMFS",
      "ticker_yfinance": "MMFS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "BLKI.NS": {
      "ticker_investpy": "BLKI",
      "ticker_yfinance": "BLKI",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "PWFC.NS": {
      "ticker_investpy": "PWFC",
      "ticker_yfinance": "PWFC",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "TOPO.NS": {
      "ticker_investpy": "TOPO",
      "ticker_yfinance": "TOPO",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "BOB.NS": {
      "ticker_investpy": "BOB",
      "ticker_yfinance": "BOB",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "GODR.NS": {
      "ticker_investpy": "GODR",
      "ticker_yfinance": "GODR",
      "Start_Date": "2010-01-06",
      "exchange_name": "NSE"
   },
   "LTFH.NS": {
      "ticker_investpy": "LTFH",
      "ticker_yfinance": "LTFH",
      "Start_Date": "2011-08-16",
      "exchange_name": "NSE"
   },
   "INBF.NS": {
      "ticker_investpy": "INBF",
      "ticker_yfinance": "INBF",
      "Start_Date": "2013-07-24",
      "exchange_name": "NSE"
   },
   "BOI.NS": {
      "ticker_investpy": "BOI",
      "ticker_yfinance": "BOI",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "JNSP.NS": {
      "ticker_investpy": "JNSP",
      "ticker_yfinance": "JNSP",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "IDFB.NS": {
      "ticker_investpy": "IDFB",
      "ticker_yfinance": "IDFB",
      "Start_Date": "2015-11-09",
      "exchange_name": "NSE"
   },
   "SUTV.NS": {
      "ticker_investpy": "SUTV",
      "ticker_yfinance": "SUTV",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "VOLT.NS": {
      "ticker_investpy": "VOLT",
      "ticker_yfinance": "VOLT",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "MGAS.NS": {
      "ticker_investpy": "MGAS",
      "ticker_yfinance": "MGAS",
      "Start_Date": "2016-07-04",
      "exchange_name": "NSE"
   },
   "RECM.NS": {
      "ticker_investpy": "RECM",
      "ticker_yfinance": "RECM",
      "Start_Date": "2008-03-13",
      "exchange_name": "NSE"
   },
   "GMRI.NS": {
      "ticker_investpy": "GMRI",
      "ticker_yfinance": "GMRI",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "BHEL.NS": {
      "ticker_investpy": "BHEL",
      "ticker_yfinance": "BHEL",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "LICH.NS": {
      "ticker_investpy": "LICH",
      "ticker_yfinance": "LICH",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "EXID.NS": {
      "ticker_investpy": "EXID",
      "ticker_yfinance": "EXID",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "TRCE.NS": {
      "ticker_investpy": "TRCE",
      "ticker_yfinance": "TRCE",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "UNBK.NS": {
      "ticker_investpy": "UNBK",
      "ticker_yfinance": "UNIONBANK.NS",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "MRTI.NS": {
      "ticker_investpy": "MRTI",
      "ticker_yfinance": "MARUTI.NS",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "TITN.NS": {
      "ticker_investpy": "TITN",
      "ticker_yfinance": "TITAN.NS",
      "Start_Date": "2008-01-02",
      "exchange_name": "NSE"
   },
   "MAHM.NS": {
      "ticker_investpy": "MAHM",
      "ticker_yfinance": "M&M.NS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "RELI.NS": {
      "ticker_investpy": "RELI",
      "ticker_yfinance": "RELIANCE.NS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "LUPN.NS": {
      "ticker_investpy": "LUPN",
      "ticker_yfinance": "LUPIN.NS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "INBK.NS": {
      "ticker_investpy": "INBK",
      "ticker_yfinance": "INDUSINDBK.NS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "KTKM.NS": {
      "ticker_investpy": "KTKM",
      "ticker_yfinance": "KOTAKBANK.NS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "BJFN.NS": {
      "ticker_investpy": "BJFN",
      "ticker_yfinance": "BAJFINANCE.NS",
      "Start_Date": "2007-07-01",
      "exchange_name": "NSE"
   },
   "INIR.NS": {
      "ticker_investpy": "INIR",
      "ticker_yfinance": "IRCTC.NS",
      "Start_Date": "2019-10-15",
      "exchange_name": "NSE"
   },
   "TAMdv.NS": {
      "ticker_investpy": "TAMdv",
      "ticker_yfinance": "TATAMTRDVR.NS",
      "Start_Date": "2013-01-14",
      "exchange_name": "NSE"
   },
   "RATB.NS": {
      "ticker_investpy": "RATB",
      "ticker_yfinance": "RBLBANK.NS",
      "Start_Date": "2016-09-01",
      "exchange_name": "NSE"
   },
   "ETH=": {
      "ticker_investpy": "ethereum",
      "ticker_yfinance": "ETH-USD",
      "Start_Date": "2018-11-01",
      "exchange_name": "Crypto"
   },
   "BTC=": {
      "ticker_investpy": "bitcoin",
      "ticker_yfinance": "BTC-USD",
      "Start_Date": "2014-07-17",
      "exchange_name": "Crypto"
   }
}


def niftydata():
    # /****************************************************************************************************************/
    # Load nifty50 list

    data_nifty = pd.read_csv("Nifty_50_Components_10Mar2021_csv.csv")
    data_nifty["Identifier (RIC)"] = data_nifty["Identifier (RIC)"].str.replace('.NS', '')
    Ticker = data_nifty["Identifier (RIC)"]

    # /****************************************************************************************************************/
    # Load Leavers and Joiners list

    data_Leavers_Joiners = pd.read_csv("Leavers_Jnrs_10_Mar_2021_csv_new.csv")
    data_Leavers_Joiners = data_Leavers_Joiners.drop(['Company'], axis=1).dropna()
    Date_req = "01-Jan-2021"  # input("Enter the date in format XX-Jan-20YY : ")
    x = data_Leavers_Joiners.apply(lambda x: datetime.strptime(x['Date'], "%d-%b-%Y"), axis=1)
    data_Leavers_Joiners['Date'] = data_Leavers_Joiners.apply(lambda x: datetime.strptime(x['Date'], "%d-%b-%Y"),
                                                              axis=1)
    row_req = data_Leavers_Joiners[data_Leavers_Joiners['Date'] > datetime.strptime(Date_req, "%d-%b-%Y")]

    rows_add = row_req[row_req["Unnamed: 0"] == '-']
    rows_sub = row_req[row_req["Unnamed: 0"] == '+']

    for i in range(len(row_req)):
        if row_req["Unnamed: 0"][i] == '-':
            Ticker = Ticker[Ticker != row_req['Code'][i]]
        else:
            A = pd.Series({'Totals (50)': [row_req["Code"][i]]})
            Ticker.append(pd.Series(A))

    return Ticker[1:].to_list()

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

def get_data(ticker, api,country):

    if api == "datatables":
        credential = AzureNamedKeyCredential(_STORAGE_ACCOUNT_NAME, _STORAGE_ACCOUNT_KEY)
        table_service = TableServiceClient(endpoint="https://acsysbatchstroageacc.table.core.windows.net/",
                                           credential=credential)
        table_client = table_service.get_table_client(table_name="DailyData")
        tasks = table_client.query_entities(query_filter=f"PartitionKey eq '{ticker}'")
        list_dict = []
        for i in tasks:
            list_dict.append(i)

        ticker_dataframe = pd.DataFrame(list_dict)
        ticker_dataframe.drop(columns=["PartitionKey", "RowKey"], inplace=True)
        temp_og = ticker_dataframe
        # temp_og.reset_index(inplace=True)
        temp_og.drop(columns="API", inplace=True)
        temp_og["Date"] = pd.to_datetime(temp_og["Date"])

    if api == "azure":

        # # Create the BlockBlobService that is used to call the Blob service for the storage account
        blob_service_client = BlockBlobService(
            account_name=_STORAGE_ACCOUNT_NAME, account_key=_STORAGE_ACCOUNT_KEY)
        container_name = 'data-india'
        blob_service_client.create_container(container_name)

        # Set the permission so the blobs are public.
        blob_service_client.set_container_acl(
            container_name, public_access=PublicAccess.Container)

        generator = blob_service_client.list_blobs(container_name)

        for blob in generator:
            if blob.name[:-4] == ticker:
                zf = zipfile.ZipFile(blob.name[:-4]+".zip",
                                     mode='w',
                                     compression=zipfile.ZIP_DEFLATED,
                                     )
                b = blob_service_client.get_blob_to_bytes(container_name, blob.name)
                zf.writestr(blob.name, b.content)
                zf.close()
                with zipfile.ZipFile(blob.name[:-4]+".zip", 'r') as zip_ref:
                    zip_ref.extractall('')
                with open(blob.name,'rb') as file:
                    temp_og = pickle.load(file)
                os.remove(f"{blob.name[:-4]}.zip")
                os.remove(f"{blob.name[:-4]}.pkl")
                temp_og.drop(columns="API", inplace=True)

                temp_og.fillna(method="ffill", inplace=True)

                temp_og = add_fisher(temp_og)
                break
            else:
                temp_og = None

        if temp_og is None:
            # # Create the BlockBlobService that is used to call the Blob service for the storage account
            blob_service_client = BlockBlobService(
                account_name=_STORAGE_ACCOUNT_NAME, account_key=_STORAGE_ACCOUNT_KEY)
            container_name = 'data-usa'
            blob_service_client.create_container(container_name)

            # Set the permission so the blobs are public.
            blob_service_client.set_container_acl(
                container_name, public_access=PublicAccess.Container)

            generator = blob_service_client.list_blobs(container_name)

            for blob in generator:
                if blob.name[:-4] == ticker:
                    zf = zipfile.ZipFile(blob.name[:-4] + ".zip",
                                         mode='w',
                                         compression=zipfile.ZIP_DEFLATED,
                                         )
                    b = blob_service_client.get_blob_to_bytes(container_name, blob.name)
                    zf.writestr(blob.name, b.content)
                    zf.close()
                    with zipfile.ZipFile(blob.name[:-4] + ".zip", 'r') as zip_ref:
                        zip_ref.extractall('')
                    with open(blob.name, 'rb') as file:
                        temp_og = pickle.load(file)
                    os.remove(f"{blob.name[:-4]}.zip")
                    os.remove(f"{blob.name[:-4]}.pkl")
                    temp_og.drop(columns="API", inplace=True)

                    temp_og.fillna(method="ffill", inplace=True)

                    temp_og = add_fisher(temp_og)

    if api == "yfinance":

        temp_og = yf.download(ticker, start = '2007-01-01', end= str(date.today()+timedelta(1)))
        if len(temp_og)==0:
            temp_og = yf.download(ticker, start='2007-01-01', end=str(date.today()))
        temp_og.reset_index(inplace=True)
        temp_og.drop(['Adj Close'], axis=1, inplace=True)
        if ticker=="GOLDBEES.NS":
            temp_og = temp_og.loc[temp_og["Close"]>1]
        temp_og = add_fisher(temp_og)

    if api =="investpy":
        temp_og = get_data_investpy(symbol=ticker, country=country, from_date="01/07/2007",to_date=(date.today()+timedelta(1)).strftime("%d/%m/%Y"))
        temp_og.reset_index(inplace=True)
        temp_og = add_fisher(temp_og)

    if api == "reuters":
        temp_og = ek.get_timeseries(ticker, start_date='2007-01-01', end_date=str(date.today() + timedelta(1)))
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

def backtest_live(input):

    date_i = input[0]
    dates = input[1]
    temp_og = input[2]
    ss_test = input[3]
    res_test = input[4]
    num_strategies = input[5]
    weights = input[6]
    recalib_months = input[7]
    dates_all = input[8]

    if (date_i - int(24 / recalib_months)) < 0:
        temp = temp_og.loc[
            (temp_og["Date"] >= str(dates_all[dates_all.index(dates[date_i]) - int(24 / 3)])) & (
                    temp_og["Date"] < str(dates[date_i + 1]))].reset_index().drop(
            ['index'], axis=1)
    else:
        temp = temp_og.loc[
            (temp_og["Date"] >= str(dates[date_i - int(24 / recalib_months)])) & (
                    temp_og["Date"] < str(dates[date_i + 1]))].reset_index().drop(
            ['index'], axis=1)

    test = temp.loc[(temp["Date"] >= str(dates[date_i]))].reset_index().drop(['index'], axis=1)

    try:
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

        _, signal = top_n_strat_params_rolling_live(temp, res, to_train=False, num_of_strat=len(selected_strategies),
                                               split_date=str(dates[date_i]))

        # Weights
        signal_final = create_final_signal_weights(signal, params, weights, len(selected_strategies))

        print_strategies = pd.concat([params.drop(["Name"], axis=1), weights.rename(columns={0: 'Weights'})], axis=1)
        print_strategies["Signal"] = signal.iloc[-1].reset_index().drop(['index'],axis=1)

        # equi-weighted
        # signal_final = create_final_signal(signal, params, len(selected_strategies))

        inp = pd.merge(test.set_index(test["Date"]), signal_final, left_index=True, right_index=True)

    except:

        inp = test.set_index(test["Date"])
        inp['signal'] = 0

    test_strategy = FISHER_MCMC_live(inp.drop(['signal', 'Date'], axis=1), inp['signal'])
    _ = test_strategy.generate_signals()
    tt = test_strategy.signal_performance(1, 6)
    return tt.set_index(pd.to_datetime(tt["Date"])).drop(columns="Date"), print_strategies

def top_n_strat_params_rolling_live(temp, res, to_train, num_of_strat, split_date):
    if len(res)>0:
        for i in range(num_of_strat):
            f_look = res.iloc[i, 0]
            bf = res.iloc[i, 1]
            sf = res.iloc[i, 2]
            temp["fisher"] = temp[f'Fisher{int(f_look)}']
            if to_train:
                train = temp.loc[(temp["Date"] < split_date)].reset_index().drop(['index'], axis=1)
            else:
                train = temp.loc[temp["Date"] >= split_date].reset_index().drop(['index'], axis=1)
            test_strategy = FISHER_bounds_strategy_opt_live(train, zone_low=bf, zone_high=sf)
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

        if len(guess)>1:
            if metric == 'rolling_sharpe':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=2000, prior=prior,
                          optimize_fn=rolling_sharpe, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'rolling_sortino':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=2000, prior=prior,
                          optimize_fn=rolling_sortino, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'rolling_cagr':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=2000, prior=prior,
                          optimize_fn=rolling_cagr, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'maxdrawup_by_maxdrawdown':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=2000, prior=prior,
                          optimize_fn=maxdrawup_by_maxdrawdown, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'outperformance':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=2000, prior=prior,
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

    for f_look in range(50, 410, 20):
        max_metric = 0
        for lb in np.round(np.arange(-7, 7, 0.25), decimals=1):
            for ub in np.round(np.arange(-7, 7, 0.25), decimals=1):
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

def get_constituents(index):
    ic, err = ek.get_data(index, ['TR.IndexConstituentRIC'])
    lj, err = ek.get_data(index,
                          ['TR.IndexJLConstituentChangeDate',
                           'TR.IndexJLConstituentRIC.change',
                           'TR.IndexJLConstituentRIC'],
                          {'SDate': '0D', 'EDate': '-55M', 'IC': 'B'})
    lj['Date'] = pd.to_datetime(lj['Date']).dt.date
    lj.sort_values(['Date', 'Change'], ascending=False, inplace=True)
    dates = [dt.date(2007, 1, 30)]
    i = 0
    while (dates[0] + dateutil.relativedelta.relativedelta(months=+i + 1)) < dt.date.today():
        dates.append(dates[0] + dateutil.relativedelta.relativedelta(months=+i + 1))
        i = i + 1
    dates.append(dt.date.today())
    df = pd.DataFrame(index=dates, columns=['Index Constituents'])
    ic_list = ic['Constituent RIC'].tolist()
    for i in range(len(dates)):
        #print(str(dates[len(dates) - i - 1]))
        df.at[dates[len(dates) - i - 1], 'Index Constituents'] = ic_list[:]
        for j in lj.index:
            if lj['Date'].loc[j] <= dates[len(dates) - i - 1]:
                if lj['Date'].loc[j] > dates[len(dates) - i - 2]:
                    if lj['Change'].loc[j] == 'Joiner':
                        #print('Removing ' + lj['Constituent RIC'].loc[j])
                        ic_list.remove(lj['Constituent RIC'].loc[j])
                    elif lj['Change'].loc[j] == 'Leaver':
                        #print('Adding ' + lj['Constituent RIC'].loc[j])
                        ic_list.append(lj['Constituent RIC'].loc[j])
                else:
                    break
    try:
        df.index = pd.date_range(start=str(df.index[0])[:10], end=str(df.index[-1].replace(month=df.index[-1].month+1))[:10], freq=f'1M')
    except:
        df.index = pd.date_range(start=str(df.index[0])[:10],end=str(df.index[-1].replace(day = 30).replace(month=df.index[-1].month + 1))[:10], freq=f'1M')
    return df

def prepare_portfolio_data(tickers, recalibrating_months,api,country):
    data = pd.DataFrame()
    for i, ticker_investing in enumerate(tickers):
        try:
            for ticker_reuters in list(tickers_all.keys()):
                if tickers_all[ticker_reuters]["ticker_investpy"] == ticker_investing:
                    ticker = ticker_reuters
                    break
                else:
                    ticker=""
            temp_og = get_data(ticker, api,country)
            temp_og = temp_og["Close"].to_frame().astype(float).rename(columns={"Close":ticker_investing}).set_index(temp_og["Date"])
            temp_og = temp_og[~temp_og.index.duplicated(keep='last')]
            data = pd.concat([data, temp_og], axis=1)
            data[f"{ticker_investing}Return"] = np.log(data[ticker_investing]/data[ticker_investing].shift(1))
            data[f"{ticker_investing}ROC0.5"] = data[ticker_investing].pct_change(10)
            data[f"{ticker_investing}ROC1"] = data[ticker_investing].pct_change(21)
            data[f"{ticker_investing}ROC3"] = data[ticker_investing].pct_change(63)
            data[f"{ticker_investing}ROC6"] = data[ticker_investing].pct_change(126)
            data[f"{ticker_investing}ROC9"] = data[ticker_investing].pct_change(189)
            data[f"{ticker_investing}ROC12"] = data[ticker_investing].pct_change(252)
            data[f"{ticker_investing}SD12"] = data[ticker_investing].pct_change().rolling(252).std()
            data[f"{ticker_investing}FReturn"] = data[ticker_investing].shift(-recalibrating_months*21) / data[ticker_investing] - 1
        except:
            print(f"{ticker_investing} not processed")
    data.reset_index(inplace=True)
    return data

def get_weights_stocks(constituents, topn, test_monthsf, train_monthsf, datesf, temp_ogf, save=True):
    inputs = []
    for date_i in range(len(datesf) - (int(train_monthsf/test_monthsf) + 1)):
        inputs.append([temp_ogf, topn,datesf,date_i,train_monthsf,test_monthsf, constituents ])
    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results = pool.map(recalibrate_weights_stocks, inputs)
    finally:  # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    # res_test2 = [pd.DataFrame(columns=["Lookback", "Low Bound", "High Bound", "AvgWinLoss",  "Sortino", "Optimization_Years"])] * len(results2)
    res_test = [pd.DataFrame(columns=["Ticker", "WtROC0.5", "WtROC1", "WtROC3", "WtROC6", "WtROC9", "WtROC12", "Accelerating Momentum"])] * (len(datesf) - (int(24/test_monthsf) + 1))
    for i in range(len(results)):
        res_test[results[i][0] + int((train_monthsf - 24) / test_monthsf)] = pd.concat(
            [res_test[results[i][0]], results[i][1].reset_index().drop(['index'], axis=1)], axis=0)

    if save == True:
        with open(f'TrainYrs_{int(train_monthsf / 12)}_Weights.pkl', 'wb') as file:
            pickle.dump(res_test, file)
    return res_test

def get_weights_stocks_live(constituents, topn, test_monthsf, train_monthsf, datesf, temp_ogf, save=True):
    inputs = []
    for date_i in range(len(datesf) - (int(train_monthsf/test_monthsf) + 1)):
        inputs.append([temp_ogf, topn,datesf,date_i,train_monthsf,test_monthsf, constituents ])

    results = recalibrate_weights_stocks(inputs[-1])

    # res_test2 = [pd.DataFrame(columns=["Lookback", "Low Bound", "High Bound", "AvgWinLoss",  "Sortino", "Optimization_Years"])] * len(results2)
    res_test = [pd.DataFrame(columns=["Ticker", "WtROC0.5", "WtROC1", "WtROC3", "WtROC6", "WtROC9", "WtROC12", "Accelerating Momentum"])]
    res_test[0] = pd.concat(
        [res_test[0], results[1].reset_index().drop(['index'], axis=1)], axis=0)

    return res_test

def recalibrate_weights_stocks(inp):

    def alpha(*args):
        weights = pd.DataFrame(args)
        weights = weights / weights.sum()
        for ticker in tickers:
            df = data[[f"{ticker}ROC0.5", f"{ticker}ROC1", f"{ticker}ROC3", f"{ticker}ROC6", f"{ticker}ROC9",
                       f"{ticker}ROC12", f"{ticker}SD12", f"{ticker}FReturn"]]
            df[f"{ticker}AM"] = np.dot(df[[f"{ticker}ROC0.5", f"{ticker}ROC1", f"{ticker}ROC3", f"{ticker}ROC6",
                                           f"{ticker}ROC9", f"{ticker}ROC12"]], weights)
            data[f"{ticker}AM"] = df[f"{ticker}AM"] / df[f"{ticker}SD12"]
        return data[[f"{ticker}AM" for ticker in tickers]].to_numpy().ravel()

    def prior(params):
        if (params[0] < 0) | (params[0] > 1):
            return 0
        if (params[1] < 0) | (params[1] > 1):
            return 0
        if (params[2] < 0) | (params[2] > 1):
            return 0
        if (params[3] < 0) | (params[3] > 1):
            return 0
        if (params[4] < 0) | (params[4] > 1):
            return 0
        if (params[5] < 0) | (params[5] > 1):
            return 0
        return 1

    # Optimizing weights for entire portfolio
    temp_ogf = inp[0]
    top_n = inp[1]
    datesf = inp[2]
    date_i = inp[3]
    train_monthsf = inp[4]
    test_monthsf = inp[5]
    constituents = inp[6]

    # data_og = temp_ogf.loc[(temp_ogf["Date"] > str(datesf[date_i])) & (temp_ogf["Date"] < str(datesf[date_i + int(train_monthsf/test_monthsf) + int(test_monthsf/test_monthsf)]))].reset_index().drop(['index'], axis=1)

    # Adjustment made for forward returns
    data_og = temp_ogf.loc[(temp_ogf["Date"] > str(datesf[date_i])) & (
                temp_ogf["Date"] < str(datesf[date_i + int(train_monthsf / test_monthsf) - 1]))].reset_index().drop(
        ['index'], axis=1)

    tickers_in_index = constituents.loc[datesf[date_i]][0]

    #data = data_og.dropna(axis=1, how='all')
    #data = data.dropna()
    data = data_og.dropna(axis=1, how='any')

    tickers1 = []
    for column in data.columns[1:]:
        if column.endswith("Return") | column.endswith("FReturn") | column.endswith("ROC0.5") | column.endswith(
                "ROC1") | column.endswith("ROC3") | \
                column.endswith("ROC6") | column.endswith("ROC9") | column.endswith("ROC12") | column.endswith("SD12"):
            continue
        else:
            tickers1.append(column)

    tickers = []
    for ticker in tickers1:
        if ((f"{ticker}" in data.columns[1:]) & (f"{ticker}ROC0.5" in data.columns[1:]) & (
                f"{ticker}ROC1" in data.columns[1:]) & (
                f"{ticker}ROC3" in data.columns[1:]) & (f"{ticker}ROC6" in data.columns[1:]) &
                (f"{ticker}ROC9" in data.columns[1:]) & (f"{ticker}ROC12" in data.columns[1:]) & (
                        f"{ticker}SD12" in data.columns[1:]) & (f"{ticker}FReturn" in data.columns[1:])):
            #print(ticker)
            tickers.append(ticker)

    for ticker in tickers:
        if not ticker in tickers_in_index:
            tickers.remove(ticker)

    random_starts = 20
    iterations = 200
    guess_list = [np.random.dirichlet(np.ones(6), size=1).tolist()[0] for i in range(random_starts)]
    res = pd.DataFrame(columns=["WtROC0.5", "WtROC1", "WtROC3", "WtROC6", "WtROC9", "WtROC12", "NMIS"])

    try:
        for guess in guess_list:
            mc = MCMC(alpha_fn=alpha, alpha_fn_params_0=guess,
                      target=data[[f"{ticker}FReturn" for ticker in tickers]].to_numpy().ravel(), num_iters=iterations,
                      prior=prior, optimize_fn=None, lower_limit=0, upper_limit=1)
            rs = mc.optimize()
            res_iter = [{"WtROC0.5": mc.analyse_results(rs, top_n=iterations)[0][i][0],
                         "WtROC1": mc.analyse_results(rs, top_n=iterations)[0][i][1],
                         "WtROC3": mc.analyse_results(rs, top_n=iterations)[0][i][2],
                         "WtROC6": mc.analyse_results(rs, top_n=iterations)[0][i][3],
                         "WtROC9": mc.analyse_results(rs, top_n=iterations)[0][i][4],
                         "WtROC12": mc.analyse_results(rs, top_n=iterations)[0][i][5],
                         "NMIS": mc.analyse_results(rs, top_n=iterations)[1][i]} \
                        for i in range(iterations)]
            res_iter = pd.DataFrame(res_iter)
            res = pd.concat([res, res_iter], axis=0)
        res = res.sort_values("NMIS", axis=0, ascending=False).reset_index(drop=True)
        chosen_weights = pd.DataFrame(
            [res.iloc[0]["WtROC0.5"], res.iloc[0]["WtROC1"], res.iloc[0]["WtROC3"], res.iloc[0]["WtROC6"],
             res.iloc[0]["WtROC9"], res.iloc[0]["WtROC12"]])
        chosen_weights = chosen_weights / chosen_weights.sum()
        am = []
        for ticker in tickers:
            am.append({"Ticker": ticker, "WtROC0.5": float(chosen_weights.iloc[0]), "WtROC1": float(chosen_weights.iloc[1]),
                       "WtROC3": float(chosen_weights.iloc[2]), "WtROC6": float(chosen_weights.iloc[3]),
                       "WtROC9": float(chosen_weights.iloc[4]), "WtROC12": float(chosen_weights.iloc[5]),
                       "Accelerating Momentum": np.dot(
                           data_og.iloc[-1][[f"{ticker}ROC0.5", f"{ticker}ROC1", f"{ticker}ROC3",
                                             f"{ticker}ROC6", f"{ticker}ROC9", f"{ticker}ROC12"]],
                           chosen_weights)[0] / data_og.iloc[-1][f"{ticker}SD12"]})
        am = pd.DataFrame(am)
        am = am.sort_values("Accelerating Momentum", axis=0, ascending=False).reset_index(drop=True)
        am = am.iloc[:top_n]
    except:
        am = pd.DataFrame(columns = ["Ticker","WtROC0.5","WtROC1","WtROC3","WtROC6","WtROC9","WtROC12","Accelerating Momentum"])
    return date_i, am

def backtest_AM_daily_rebalance(input):

    date_i = input[0]
    dates_rebalancing = input[1]
    data_inp = input[2]
    assetsb = input[3]

    try:
        test = data_inp.loc[
            (data_inp["Date"] >= str(dates_rebalancing[date_i])) & (
                        data_inp["Date"] < str(dates_rebalancing[date_i + 1]))].reset_index().drop(['index'], axis=1)
        test.set_index(test["Date"], inplace=True)
    except:
        test = data_inp.loc[(data_inp["Date"] > str(dates_rebalancing[date_i]))]
        test.set_index(test["Date"], inplace=True)
    tickers = assetsb[date_i]["Ticker"].to_list()
    returns = pd.DataFrame()
    returns["Return"] = test[[f"{ticker}Return" for ticker in tickers]].mean(axis=1)

    return returns["Return"]

def backtest_Alpha_AM_daily_rebalance(input):

    date_i = input[0]
    dates_rebalancing = input[1]
    data_inp = input[2]
    assetsb = input[3]

    try:
        test = data_inp.loc[
            (data_inp["Date"] > str(dates_rebalancing[date_i])) & (
                        data_inp["Date"] < str(dates_rebalancing[date_i + 1]))].reset_index().drop(['index'], axis=1)
        test.set_index(test["Date"], inplace=True)
    except:
        test = data_inp.loc[(data_inp["Date"] > str(dates_rebalancing[date_i]))]
        test.set_index(test["Date"], inplace=True)
    tickers = assetsb[date_i]["Ticker"].to_list()
    returns = pd.DataFrame()
    returns["Return_nifty"] = test[[f"{ticker}Return" for ticker in tickers]].mean(axis=1)*test["signal_nifty"].shift(1)#+(6/ 25200) * (1 - test["signal"].shift(1))
    returns["signal_nifty"] = test["signal_nifty"].shift(1)
    returns["Return_gold"] = np.log(test["Close_gold"]/test["Close_gold"].shift(1))*test["signal_gold"].shift(1)
    returns["signal_gold"] = test["signal_gold"].shift(1)
    returns["Return"] = 0

    for i in range(len(returns)):

        if returns["signal_nifty"].iloc[i]==1:
            returns["Return"].iloc[i] = returns["Return_nifty"].iloc[i]
            continue
        if returns["signal_gold"].iloc[i]==1:
            returns["Return"].iloc[i] = 0.5*returns["Return_gold"].iloc[i]
            continue
        returns["Return"].iloc[i] = (6/ 25200)


    return returns["Return"]

def backtest_Alpha_AM_Midcap(dates_rebalancing,data_inp,assetsb,current_balance):
    gold_allocation = 0
    nifty_allocation = 0
    cash_allocation = 0
    portfolio_value = pd.DataFrame()

    for date_i in range(len(dates_rebalancing) - 1):
        try:
            test = data_inp.loc[
                (data_inp["Date"] >= str(dates_rebalancing[date_i])) & (
                        data_inp["Date"] < str(dates_rebalancing[date_i + 1]))].reset_index().drop(['index'], axis=1)
            test.set_index(test["Date"], inplace=True)
        except:
            test = data_inp.loc[(data_inp["Date"] >= str(dates_rebalancing[date_i]))]
            test.set_index(test["Date"], inplace=True)
        tickers = assetsb[date_i]["Ticker"].to_list()

        percent_tracker_current_balance_ticker = {}
        percent_tracker_units_ticker = {}
        percent_ticker = {}
        for ticker in tickers:
            percent_tracker_current_balance_ticker[ticker] = current_balance / len(tickers)
            percent_tracker_units_ticker[ticker] = percent_tracker_current_balance_ticker[ticker] / test.iloc[0][ticker]

        current_balance_ticker = {}
        units_ticker = {}
        for ticker in tickers:
            current_balance_ticker[ticker] = current_balance / len(tickers)
            units_ticker[ticker] = current_balance_ticker[ticker] / test.iloc[0][ticker]

        units_gold = 0

        for i in range(len(test)):

            for ticker in tickers:
                percent_tracker_current_balance_ticker[ticker] = percent_tracker_units_ticker[ticker] *  test.iloc[i][ticker]
            for ticker in tickers:
                percent_ticker[ticker] = percent_tracker_current_balance_ticker[ticker] / sum(percent_tracker_current_balance_ticker.values())

            signal_midcap = test["signal_midcap"].iloc[i]
            signal_gold = test["signal_gold"].iloc[i]

            if signal_midcap == 1:
                midcap_allocation = current_balance
                gold_allocation = 0
                cash_allocation = 0
            if (signal_midcap == 0) & (signal_gold == 1):
                midcap_allocation = 0
                gold_allocation = current_balance / 2
                cash_allocation = current_balance / 2
            if (signal_midcap == 0) & (signal_gold == 0):
                midcap_allocation = 0
                gold_allocation = 0
                cash_allocation = current_balance

            if ((test["signal_midcap"].iloc[i] == 1) & (test["signal_midcap"].shift(1).fillna(0).iloc[i] == 0)):
                for ticker in tickers:
                    current_balance_ticker[ticker] = midcap_allocation * percent_ticker[ticker]
                    units_ticker[ticker] = current_balance_ticker[ticker] / test.iloc[i][ticker]

            if ((test["signal_midcap"].iloc[i] == 0) & (test["signal_midcap"].shift(1).fillna(1).iloc[i] == 1)):
                for ticker in tickers:
                    current_balance_ticker[ticker] = 0
                    units_ticker[ticker] = 0
                if signal_gold == 1:
                    units_gold = gold_allocation / test.iloc[i]["Close_gold"]

            if ((test["signal_gold"].iloc[i] == 1) & (test["signal_gold"].shift(1).fillna(0).iloc[i] == 0)):
                units_gold = gold_allocation / test.iloc[i]["Close_gold"]

            if ((test["signal_gold"].iloc[i] == 0) & (test["signal_gold"].shift(1).fillna(1).iloc[i] == 1)):
                units_gold = 0

            if signal_midcap == 1:
                units_gold = 0
                midcap_allocation = 0
                for ticker in tickers:
                    current_balance_ticker[ticker] = units_ticker[ticker] * test.iloc[i][ticker]
                    midcap_allocation = midcap_allocation + current_balance_ticker[ticker]
            if (signal_midcap == 0) & (signal_gold == 1):
                gold_allocation = units_gold * test.iloc[i]["Close_gold"]
            cash_allocation = cash_allocation * (1 + 6 / 25200)
            current_balance = midcap_allocation + gold_allocation + cash_allocation
            portfolio_day = {'Date': test.iloc[i]["Date"], 'signal_midcap': signal_midcap, 'signal_gold':signal_gold,'units_gold':units_gold,'midcap_allocation':midcap_allocation,'gold_allocation':gold_allocation,'cash_allocation':cash_allocation,'Pvalue': current_balance}
            portfolio_day[f"Gold_close"] = test.iloc[i]["Close_gold"]
            for ticker in tickers:
                portfolio_day[f"{ticker}_close"] = test.iloc[i][ticker]
                portfolio_day[f"{ticker}_percent"] = percent_ticker[ticker]
                portfolio_day[f"{ticker}_units"] = units_ticker[ticker]
                portfolio_day[f"{ticker}_current_balance"] = current_balance_ticker[ticker]
            portfolio_day = pd.DataFrame([portfolio_day])
            portfolio_day = portfolio_day.set_index("Date")
            portfolio_value = pd.concat([portfolio_value, portfolio_day], axis=0, join="outer")


    #print(portfolio_value)
    #returns = pd.DataFrame(np.log(portfolio_value / portfolio_value.shift(1))).rename(columns={"Pvalue": "Return"})
    return portfolio_value,units_ticker,units_gold

def backtest_Alpha_AM(dates_rebalancing,data_inp,assetsb):
    current_balance = 7930747.142346616
    gold_allocation = 0
    nifty_allocation = 0
    cash_allocation = 0
    portfolio_value = pd.DataFrame()

    for date_i in range(len(dates_rebalancing) - 1):
        try:
            test = data_inp.loc[
                (data_inp["Date"] >= str(dates_rebalancing[date_i])) & (
                        data_inp["Date"] < str(dates_rebalancing[date_i + 1]))].reset_index().drop(['index'], axis=1)
            test.set_index(test["Date"], inplace=True)
        except:
            test = data_inp.loc[(data_inp["Date"] >= str(dates_rebalancing[date_i]))]
            test.set_index(test["Date"], inplace=True)
        tickers = assetsb[date_i]["Ticker"].to_list()

        percent_tracker_current_balance_ticker = {}
        percent_tracker_units_ticker = {}
        percent_ticker = {}
        for ticker in tickers:
            percent_tracker_current_balance_ticker[ticker] = current_balance / len(tickers)
            percent_tracker_units_ticker[ticker] = percent_tracker_current_balance_ticker[ticker] / test.iloc[0][ticker]

        current_balance_ticker = {}
        units_ticker = {}
        for ticker in tickers:
            current_balance_ticker[ticker] = current_balance / len(tickers)
            units_ticker[ticker] = current_balance_ticker[ticker] / test.iloc[0][ticker]

        units_gold = 0

        for i in range(len(test)):

            for ticker in tickers:
                percent_tracker_current_balance_ticker[ticker] = percent_tracker_units_ticker[ticker] *  test.iloc[i][ticker]
            for ticker in tickers:
                percent_ticker[ticker] = percent_tracker_current_balance_ticker[ticker] / sum(percent_tracker_current_balance_ticker.values())

            signal_nifty = test["signal_nifty"].iloc[i]
            signal_gold = test["signal_gold"].iloc[i]

            if signal_nifty == 1:
                nifty_allocation = current_balance
                gold_allocation = 0
                cash_allocation = 0
            if (signal_nifty == 0) & (signal_gold == 1):
                nifty_allocation = 0
                gold_allocation = current_balance / 2
                cash_allocation = current_balance / 2
            if (signal_nifty == 0) & (signal_gold == 0):
                nifty_allocation = 0
                gold_allocation = 0
                cash_allocation = current_balance

            if ((test["signal_nifty"].iloc[i] == 1) & (test["signal_nifty"].shift(1).fillna(0).iloc[i] == 0)):
                for ticker in tickers:
                    current_balance_ticker[ticker] = nifty_allocation * percent_ticker[ticker]
                    units_ticker[ticker] = current_balance_ticker[ticker] / test.iloc[i][ticker]

            if ((test["signal_nifty"].iloc[i] == 0) & (test["signal_nifty"].shift(1).fillna(1).iloc[i] == 1)):
                for ticker in tickers:
                    current_balance_ticker[ticker] = 0
                    units_ticker[ticker] = 0
                if signal_gold == 1:
                    units_gold = gold_allocation / test.iloc[i]["Close_gold"]

            if ((test["signal_gold"].iloc[i] == 1) & (test["signal_gold"].shift(1).fillna(0).iloc[i] == 0)):
                units_gold = gold_allocation / test.iloc[i]["Close_gold"]

            if ((test["signal_gold"].iloc[i] == 0) & (test["signal_gold"].shift(1).fillna(1).iloc[i] == 1)):
                units_gold = 0

            if signal_nifty == 1:
                units_gold = 0
                nifty_allocation = 0
                for ticker in tickers:
                    current_balance_ticker[ticker] = units_ticker[ticker] * test.iloc[i][ticker]
                    nifty_allocation = nifty_allocation + current_balance_ticker[ticker]
            if (signal_nifty == 0) & (signal_gold == 1):
                gold_allocation = units_gold * test.iloc[i]["Close_gold"]
            cash_allocation = cash_allocation * (1 + 6 / 25200)
            current_balance = nifty_allocation + gold_allocation + cash_allocation
            portfolio_day = {'Date': test.iloc[i]["Date"], 'signal_nifty': signal_nifty, 'signal_gold':signal_gold,'units_gold':units_gold,'nifty_allocation':nifty_allocation,'gold_allocation':gold_allocation,'cash_allocation':cash_allocation,'Pvalue': current_balance}
            portfolio_day[f"Gold_close"] = test.iloc[i]["Close_gold"]
            for ticker in tickers:
                portfolio_day[f"{ticker}_close"] = test.iloc[i][ticker]
                portfolio_day[f"{ticker}_percent"] = percent_ticker[ticker]
                portfolio_day[f"{ticker}_units"] = units_ticker[ticker]
                portfolio_day[f"{ticker}_current_balance"] = current_balance_ticker[ticker]
            portfolio_day = pd.DataFrame([portfolio_day])
            portfolio_day = portfolio_day.set_index("Date")
            portfolio_value = pd.concat([portfolio_value, portfolio_day], axis=0, join="outer")

    #print(portfolio_value)
    #returns = pd.DataFrame(np.log(portfolio_value / portfolio_value.shift(1))).rename(columns={"Pvalue": "Return"})
    return portfolio_value,units_ticker,units_gold

def backtest_Alpha_AM_Nifty_Acsys(dates_rebalancing,data_inp,assetsb):
    current_balance = 8151200
    gold_allocation = 0
    nifty_allocation = 0
    cash_allocation = 0
    portfolio_value = pd.DataFrame()

    for date_i in range(len(dates_rebalancing) - 1):
        try:
            test = data_inp.loc[
                (data_inp["Date"] >= str(dates_rebalancing[date_i])) & (
                        data_inp["Date"] < str(dates_rebalancing[date_i + 1]))].reset_index().drop(['index'], axis=1)
            test.set_index(test["Date"], inplace=True)
        except:
            test = data_inp.loc[(data_inp["Date"] >= str(dates_rebalancing[date_i]))]
            test.set_index(test["Date"], inplace=True)
        tickers = assetsb[date_i]["Ticker"].to_list()

        percent_tracker_current_balance_ticker = {}
        percent_tracker_units_ticker = {}
        percent_ticker = {}
        for ticker in tickers:
            percent_tracker_current_balance_ticker[ticker] = current_balance / len(tickers)
            percent_tracker_units_ticker[ticker] = percent_tracker_current_balance_ticker[ticker] / test.iloc[0][ticker]

        current_balance_ticker = {}
        units_ticker = {}
        for ticker in tickers:
            current_balance_ticker[ticker] = current_balance / len(tickers)
            units_ticker[ticker] = current_balance_ticker[ticker] / test.iloc[0][ticker]

        units_gold = 0

        for i in range(len(test)):

            for ticker in tickers:
                percent_tracker_current_balance_ticker[ticker] = percent_tracker_units_ticker[ticker] *  test.iloc[i][ticker]
            for ticker in tickers:
                percent_ticker[ticker] = percent_tracker_current_balance_ticker[ticker] / sum(percent_tracker_current_balance_ticker.values())

            signal_nifty = test["signal_nifty"].iloc[i]
            signal_gold = test["signal_gold"].iloc[i]

            if signal_nifty == 1:
                nifty_allocation = current_balance
                gold_allocation = 0
                cash_allocation = 0
            if (signal_nifty == 0) & (signal_gold == 1):
                nifty_allocation = 0
                gold_allocation = current_balance / 2
                cash_allocation = current_balance / 2
            if (signal_nifty == 0) & (signal_gold == 0):
                nifty_allocation = 0
                gold_allocation = 0
                cash_allocation = current_balance

            if ((test["signal_nifty"].iloc[i] == 1) & (test["signal_nifty"].shift(1).fillna(0).iloc[i] == 0)):
                for ticker in tickers:
                    current_balance_ticker[ticker] = nifty_allocation * percent_ticker[ticker]
                    units_ticker[ticker] = current_balance_ticker[ticker] / test.iloc[i][ticker]

            if ((test["signal_nifty"].iloc[i] == 0) & (test["signal_nifty"].shift(1).fillna(1).iloc[i] == 1)):
                for ticker in tickers:
                    current_balance_ticker[ticker] = 0
                    units_ticker[ticker] = 0
                if signal_gold == 1:
                    units_gold = gold_allocation / test.iloc[i]["Close_gold"]

            if ((test["signal_gold"].iloc[i] == 1) & (test["signal_gold"].shift(1).fillna(0).iloc[i] == 0)):
                units_gold = gold_allocation / test.iloc[i]["Close_gold"]

            if ((test["signal_gold"].iloc[i] == 0) & (test["signal_gold"].shift(1).fillna(1).iloc[i] == 1)):
                units_gold = 0

            if signal_nifty == 1:
                units_gold = 0
                nifty_allocation = 0
                for ticker in tickers:
                    current_balance_ticker[ticker] = units_ticker[ticker] * test.iloc[i][ticker]
                    nifty_allocation = nifty_allocation + current_balance_ticker[ticker]
            if (signal_nifty == 0) & (signal_gold == 1):
                gold_allocation = units_gold * test.iloc[i]["Close_gold"]
            cash_allocation = cash_allocation * (1 + 6 / 25200)
            current_balance = nifty_allocation + gold_allocation + cash_allocation
            portfolio_day = {'Date': test.iloc[i]["Date"], 'signal_nifty': signal_nifty, 'signal_gold':signal_gold,'units_gold':units_gold,'nifty_allocation':nifty_allocation,'gold_allocation':gold_allocation,'cash_allocation':cash_allocation,'Pvalue': current_balance}
            portfolio_day[f"Gold_close"] = test.iloc[i]["Close_gold"]
            for ticker in tickers:
                portfolio_day[f"{ticker}_close"] = test.iloc[i][ticker]
                portfolio_day[f"{ticker}_percent"] = percent_ticker[ticker]
                portfolio_day[f"{ticker}_units"] = units_ticker[ticker]
                portfolio_day[f"{ticker}_current_balance"] = current_balance_ticker[ticker]
            portfolio_day = pd.DataFrame([portfolio_day])
            portfolio_day = portfolio_day.set_index("Date")
            portfolio_value = pd.concat([portfolio_value, portfolio_day], axis=0, join="outer")

    #print(portfolio_value)
    #returns = pd.DataFrame(np.log(portfolio_value / portfolio_value.shift(1))).rename(columns={"Pvalue": "Return"})
    return portfolio_value,units_ticker,units_gold


def rolling_outperformance(p, naive_p):
    # ecdf1 = p['Pvalue'].to_frame().set_index(p["Date"])
    # ecdf2 = naive_p['Pvalue'].to_frame().set_index(naive_p["Date"])
    ecdf1 = pd.DataFrame()
    ecdf2= pd.DataFrame()
    ecdf1['Portfolio'] = p
    ecdf1['Portfolio_1yr'] = ecdf1['Portfolio'].to_frame().shift(365, freq='D')
    ecdf2['Portfolio'] = 1 + naive_p
    ecdf2['Portfolio_1yr'] = ecdf2['Portfolio'].to_frame().shift(365, freq='D')
    ecdf1['Portfolio_1yr'] = ecdf1['Portfolio_1yr'].fillna(method="ffill")
    ecdf1['RCAGR_Strategy_Series'] = ecdf1['Portfolio'] / ecdf1['Portfolio_1yr'] - 1
    ecdf2['Portfolio_1yr'] = ecdf2['Portfolio_1yr'].fillna(method="ffill")
    ecdf2['RCAGR_Naive_Series'] = ecdf2['Portfolio'] / ecdf2['Portfolio_1yr'] - 1
    ROutperformance = ecdf1['RCAGR_Strategy_Series'] - ecdf2['RCAGR_Naive_Series']
    return ROutperformance


def backtest_Alpha_AM_NASDAQ(dates_rebalancing,data_inp,assetsb):
    current_balance = 39364
    tlt_allocation = 0
    nasdaq_allocation = 0
    cash_allocation = 0
    portfolio_value = pd.DataFrame()

    for date_i in range(len(dates_rebalancing) - 1):
        try:
            test = data_inp.loc[
                (data_inp["Date"] >= str(dates_rebalancing[date_i])) & (
                        data_inp["Date"] < str(dates_rebalancing[date_i + 1]))].reset_index().drop(['index'], axis=1)
            test.set_index(test["Date"], inplace=True)
        except:
            test = data_inp.loc[(data_inp["Date"] >= str(dates_rebalancing[date_i]))]
            test.set_index(test["Date"], inplace=True)
        tickers = assetsb[date_i]["Ticker"].to_list()

        if 'MXIM' in tickers:
            tickers.remove('MXIM')

        if 'MAR' in tickers:
            tickers.remove('MAR')

        if 'GOOG' in tickers:
            tickers.remove('GOOG')
            if 'GOOGL' not in tickers:
                tickers = tickers + ['GOOGL']

        percent_tracker_current_balance_ticker = {}
        percent_tracker_units_ticker = {}
        percent_ticker = {}
        for ticker in tickers:
            percent_tracker_current_balance_ticker[ticker] = current_balance / len(tickers)
            percent_tracker_units_ticker[ticker] = percent_tracker_current_balance_ticker[ticker] / test.iloc[0][ticker]

        current_balance_ticker = {}
        units_ticker = {}
        for ticker in tickers:
            current_balance_ticker[ticker] = current_balance / len(tickers)
            units_ticker[ticker] = current_balance_ticker[ticker] / test.iloc[0][ticker]

        units_tlt = 0

        for i in range(len(test)):

            for ticker in tickers:
                if math.isnan(test.iloc[i][ticker]):
                    percent_tracker_current_balance_ticker[ticker] = percent_tracker_units_ticker[ticker] * test.iloc[i-1][ticker]
                else:
                    percent_tracker_current_balance_ticker[ticker] = percent_tracker_units_ticker[ticker] *  test.iloc[i][ticker]
            for ticker in tickers:
                percent_ticker[ticker] = percent_tracker_current_balance_ticker[ticker] / sum(percent_tracker_current_balance_ticker.values())

            signal_nasdaq = test["signal_nasdaq"].iloc[i]
            signal_tlt = test["signal_tlt"].iloc[i]

            if signal_nasdaq == 1:
                nasdaq_allocation = current_balance
                tlt_allocation = 0
                cash_allocation = 0
            if (signal_nasdaq == 0) & (signal_tlt == 1):
                nasdaq_allocation = 0
                tlt_allocation = current_balance / 2
                cash_allocation = current_balance / 2
            if (signal_nasdaq == 0) & (signal_tlt == 0):
                nasdaq_allocation = 0
                tlt_allocation = 0
                cash_allocation = current_balance

            if ((test["signal_nasdaq"].iloc[i] == 1) & (test["signal_nasdaq"].shift(1).fillna(0).iloc[i] == 0)):
                for ticker in tickers:
                    current_balance_ticker[ticker] = nasdaq_allocation * percent_ticker[ticker]
                    units_ticker[ticker] = current_balance_ticker[ticker] / test.iloc[i][ticker]

            if ((test["signal_nasdaq"].iloc[i] == 0) & (test["signal_nasdaq"].shift(1).fillna(1).iloc[i] == 1)):
                for ticker in tickers:
                    current_balance_ticker[ticker] = 0
                    units_ticker[ticker] = 0
                if signal_tlt == 1:
                    units_tlt = tlt_allocation / test.iloc[i]["Close_tlt"]

            if ((test["signal_tlt"].iloc[i] == 1) & (test["signal_tlt"].shift(1).fillna(0).iloc[i] == 0)):
                units_tlt = tlt_allocation / test.iloc[i]["Close_tlt"]

            if ((test["signal_tlt"].iloc[i] == 0) & (test["signal_tlt"].shift(1).fillna(1).iloc[i] == 1)):
                units_tlt = 0

            if signal_nasdaq == 1:
                units_tlt = 0
                nasdaq_allocation = 0
                for ticker in tickers:
                    current_balance_ticker[ticker] = units_ticker[ticker] * test.iloc[i][ticker]
                    nasdaq_allocation = nasdaq_allocation + current_balance_ticker[ticker]
            if (signal_nasdaq == 0) & (signal_tlt == 1):
                tlt_allocation = units_tlt * test.iloc[i]["Close_tlt"]
            cash_allocation = cash_allocation * (1 + 6 / 25200)
            current_balance = nasdaq_allocation + tlt_allocation + cash_allocation

            portfolio_day = {'Date': test.iloc[i]["Date"], 'signal_nasdaq': signal_nasdaq, 'signal_tlt':signal_tlt,'units_tlt':units_tlt,'nasdaq_allocation':nasdaq_allocation,'tlt_allocation':tlt_allocation,'cash_allocation':cash_allocation,'Pvalue': current_balance}
            portfolio_day[f"Tlt_close"] = test.iloc[i]["Close_tlt"]
            for ticker in tickers:
                portfolio_day[f"{ticker}_close"] = test.iloc[i][ticker]
                portfolio_day[f"{ticker}_percent"] = percent_ticker[ticker]
                portfolio_day[f"{ticker}_units"] = units_ticker[ticker]
                portfolio_day[f"{ticker}_current_balance"] = current_balance_ticker[ticker]
            portfolio_day = pd.DataFrame([portfolio_day])
            portfolio_day = portfolio_day.set_index("Date")
            portfolio_value = pd.concat([portfolio_value, portfolio_day], axis=0, join="outer")

    #print(portfolio_value)
    #returns = pd.DataFrame(np.log(portfolio_value / portfolio_value.shift(1))).rename(columns={"Pvalue": "Return"})
    return portfolio_value,units_ticker,units_tlt

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

class FISHER_bounds_strategy_opt_live:
    metrics = {'Sharpe': 'Sharpe Ratio',
               'CAGR': 'Compound Average Growth Rate',
               'MDD': 'Maximum Drawdown',
               'NHR': 'Normalized Hit Ratio',
               'OTS': 'Optimal trade size'}

    def __init__(self, data, zone_low, zone_high, start=None, end=None):

        self.zl = zone_low
        self.zh = zone_high
        self.data = data  # the dataframe
        self.data['yr'] = self.data['Date'].dt.year
        # self.n_days = (self.data.Date.iloc[-1] - self.data.Date.iloc[0])
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

        sell_mask = ((self.data["fisher"] < self.data["ub"]) & (self.data["fisher_lag"] > self.data["ub"]))|(self.data["fisher"] < np.minimum(self.data["lb"], self.data["ub"]))

        bval = +1
        sval = 0 # -1 if short selling is allowed, otherwise 0

        self.data['signal_bounds'] = np.nan
        self.data.loc[buy_mask, 'signal_bounds'] = bval
        self.data.loc[sell_mask, 'signal_bounds'] = sval
        # initialize with long
        self.data["signal_bounds"][0] = 1

        self.data.signal_bounds = self.data.signal_bounds.fillna(method="ffill")
        #self.data.signal_bounds = self.data.signal_bounds.fillna(0)

        self.data["signal"] = self.data.signal_bounds

        mask = ((self.data.signal == 1) & (self.data.signal.shift(1)==0)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

        return self.data[["Date", "signal"]]

    def signal_performance(self, allocation, interest_rate):

        self.allocation = allocation
        self.int = interest_rate

        # creating returns and portfolio value series
        self.data['Return'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['S_Return'] = self.data['signal'].shift(1) * self.data['Return']
        self.data['S_Return'] = self.data['S_Return'].fillna(0)
        self.data["S_Return"] = self.data["S_Return"] + (self.int/25200)*(1-self.data['signal'].shift(1))
        self.data['Market_Return'] = self.data['Return'].expanding().sum()
        self.data['Strategy_Return'] = self.data['S_Return'].expanding().sum()
        self.data['Portfolio Value'] = ((self.data['Strategy_Return'] + 1) * self.allocation)
        self.data['Wins'] = np.where(self.data['S_Return'] > 0, 1, 0)
        self.data['Losses'] = np.where(self.data['S_Return'] < 0, 1, 0)

        return self.data[['Date', 'Close', 'signal', 'S_Return', 'trade_num', 'Return']]

class FISHER_MCMC_live:

    def __init__(self, data, signals, start=None, end=None):

        self.signals = signals
        self.data = data  # the dataframe
        self.start = start  # the beginning date of the sample period
        self.end = end  # the ending date of the sample period

    def generate_signals(self, charts=True):

        self.data["signal"] = self.signals

        mask = ((self.data.signal == 1) & (self.data.signal.shift(1)==0)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

    def signal_performance(self, allocation, interest_rate):

        self.allocation = allocation
        self.int = interest_rate
        self.data = self.data.reset_index().rename(columns={'index':'Date'})
        self.data['yr'] = self.data['Date'].dt.year
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

def optimize_weights_live(input):

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

        if len(guess)>1:
            if metric == 'rolling_sharpe':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=10000, prior=prior,
                          optimize_fn=rolling_sharpe, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'rolling_sortino':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=10000, prior=prior,
                          optimize_fn=rolling_sortino, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'rolling_cagr':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=10000, prior=prior,
                          optimize_fn=rolling_cagr, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'maxdrawup_by_maxdrawdown':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=10000, prior=prior,
                          optimize_fn=maxdrawup_by_maxdrawdown, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'outperformance':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=10000, prior=prior,
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

    else:
        weights = pd.DataFrame()

    return date_i, weights