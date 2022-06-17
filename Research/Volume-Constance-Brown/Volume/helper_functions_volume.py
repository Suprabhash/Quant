import multiprocessing
import itertools
import eikon as ek
from datetime import date,timedelta
import seaborn as sns
import psutil
import pytz
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from tqdm import tqdm
import ssl
import pickle
import os
from scipy import stats
from MCMC.MCMC import MCMC
import math
import time
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    make_scorer,
    recall_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
    matthews_corrcoef,
    classification_report,
)
from sklearn.svm import SVR

from azure.data.tables import TableServiceClient
from azure.core.credentials import AzureNamedKeyCredential

_STORAGE_ACCOUNT_NAME = 'acsysbatchstroageacc'  # Your storage account name
_STORAGE_ACCOUNT_KEY = 'A1I+gDNiz+bBbFbAh+vvRthOccMvAOQ8BkpJ8EVfp3x4yWnVe1fx4aW8BUSeovltcM4TBqb7YLEAOabjuLisdA=='


tickers_all = {
    ".NSEI": {
        "ticker_investpy": "NSEI",
        "ticker_yfinance": "^NSEI",
        "Start_Date": "2007-09-17",
        "exchange_name": "NSE",
    },
    "GBES.NS": {
        "ticker_investpy": "NA",
        "ticker_yfinance": "GOLDBEES.NS",
        "Start_Date": "2009-01-02",
        "exchange_name": "NSE",
    },
    "TLT.OQ": {
        "ticker_investpy": "TLT",
        "ticker_yfinance": "TLT",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    ".IXIC": {
        "ticker_investpy": "IXIC",
        "ticker_yfinance": "^IXIC",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    ".NIMDCP50": {
        "ticker_investpy": "NIMDCP50",
        "ticker_yfinance": "^NSEMDCP50",
        "Start_Date": "2007-09-24",
        "exchange_name": "NSE",
    },
    "ROST.OQ": {
        "ticker_investpy": "ROST",
        "ticker_yfinance": "ROST",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "MNST.OQ": {
        "ticker_investpy": "MNST",
        "ticker_yfinance": "MNST",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "CMCSA.OQ": {
        "ticker_investpy": "CMCSA",
        "ticker_yfinance": "CMCSA",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "KLAC.OQ": {
        "ticker_investpy": "KLAC",
        "ticker_yfinance": "KLAC",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "NXPI.OQ": {
        "ticker_investpy": "NXPI",
        "ticker_yfinance": "NXPI",
        "Start_Date": "2010-08-09",
        "exchange_name": "NASDAQ",
    },
    "XLNX.OQ": {
        "ticker_investpy": "XLNX",
        "ticker_yfinance": "XLNX",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "ALGN.OQ": {
        "ticker_investpy": "ALGN",
        "ticker_yfinance": "ALGN",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "MRVL.OQ": {
        "ticker_investpy": "MRVL",
        "ticker_yfinance": "MRVL",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "ISRG.OQ": {
        "ticker_investpy": "ISRG",
        "ticker_yfinance": "ISRG",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "MAT.OQ": {
        "ticker_investpy": "MAT",
        "ticker_yfinance": "MAT",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "OKTA.OQ": {
        "ticker_investpy": "OKTA",
        "ticker_yfinance": "OKTA",
        "Start_Date": "2017-04-10",
        "exchange_name": "NASDAQ",
    },
    "AVGO.OQ": {
        "ticker_investpy": "AVGO",
        "ticker_yfinance": "AVGO",
        "Start_Date": "2009-08-06",
        "exchange_name": "NASDAQ",
    },
    "DXCM.OQ": {
        "ticker_investpy": "DXCM",
        "ticker_yfinance": "DXCM",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "AMD.OQ": {
        "ticker_investpy": "AMD",
        "ticker_yfinance": "AMD",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "DOCU.OQ": {
        "ticker_investpy": "DOCU",
        "ticker_yfinance": "DOCU",
        "Start_Date": "2018-04-30",
        "exchange_name": "NASDAQ",
    },
    "INTC.OQ": {
        "ticker_investpy": "INTC",
        "ticker_yfinance": "INTC",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "UAL.OQ": {
        "ticker_investpy": "UAL",
        "ticker_yfinance": "UAL",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "KDP.OQ": {
        "ticker_investpy": "KDP",
        "ticker_yfinance": "KDP",
        "Start_Date": "2008-04-28",
        "exchange_name": "NASDAQ",
    },
    "WBA.OQ": {
        "ticker_investpy": "WBA",
        "ticker_yfinance": "WBA",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "CSCO.OQ": {
        "ticker_investpy": "CSCO",
        "ticker_yfinance": "CSCO",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "SIRI.OQ": {
        "ticker_investpy": "SIRI",
        "ticker_yfinance": "SIRI",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "LRCX.OQ": {
        "ticker_investpy": "LRCX",
        "ticker_yfinance": "LRCX",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "GILD.OQ": {
        "ticker_investpy": "GILD",
        "ticker_yfinance": "GILD",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "ADP.OQ": {
        "ticker_investpy": "ADP",
        "ticker_yfinance": "ADP",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "NLOK.OQ": {
        "ticker_investpy": "NLOK",
        "ticker_yfinance": "NLOK",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "ADSK.OQ": {
        "ticker_investpy": "ADSK",
        "ticker_yfinance": "ADSK",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "AMZN.OQ": {
        "ticker_investpy": "AMZN",
        "ticker_yfinance": "AMZN",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "QRTEA.OQ": {
        "ticker_investpy": "QRTEA",
        "ticker_yfinance": "QRTEA",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "REGN.OQ": {
        "ticker_investpy": "REGN",
        "ticker_yfinance": "REGN",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "LBTYA.OQ": {
        "ticker_investpy": "LBTYA",
        "ticker_yfinance": "LBTYA",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "TMUS.OQ": {
        "ticker_investpy": "TMUS",
        "ticker_yfinance": "TMUS",
        "Start_Date": "2007-04-20",
        "exchange_name": "NASDAQ",
    },
    "LULU.OQ": {
        "ticker_investpy": "LULU",
        "ticker_yfinance": "LULU",
        "Start_Date": "2007-07-30",
        "exchange_name": "NASDAQ",
    },
    "SGEN.OQ": {
        "ticker_investpy": "SGEN",
        "ticker_yfinance": "SGEN",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "MDLZ.OQ": {
        "ticker_investpy": "MDLZ",
        "ticker_yfinance": "MDLZ",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "INCY.OQ": {
        "ticker_investpy": "INCY",
        "ticker_yfinance": "INCY",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "TCOM.OQ": {
        "ticker_investpy": "TCOM",
        "ticker_yfinance": "TCOM",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "STX.OQ": {
        "ticker_investpy": "STX",
        "ticker_yfinance": "STX",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "CDNS.OQ": {
        "ticker_investpy": "CDNS",
        "ticker_yfinance": "CDNS",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "NTAP.OQ": {
        "ticker_investpy": "NTAP",
        "ticker_yfinance": "NTAP",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "HAS.OQ": {
        "ticker_investpy": "HAS",
        "ticker_yfinance": "HAS",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "CHTR.OQ": {
        "ticker_investpy": "CHTR",
        "ticker_yfinance": "CHTR",
        "Start_Date": "2009-12-02",
        "exchange_name": "NASDAQ",
    },
    "ILMN.OQ": {
        "ticker_investpy": "ILMN",
        "ticker_yfinance": "ILMN",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "SBUX.OQ": {
        "ticker_investpy": "SBUX",
        "ticker_yfinance": "SBUX",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "PYPL.OQ": {
        "ticker_investpy": "PYPL",
        "ticker_yfinance": "PYPL",
        "Start_Date": "2015-07-07",
        "exchange_name": "NASDAQ",
    },
    "EBAY.OQ": {
        "ticker_investpy": "EBAY",
        "ticker_yfinance": "EBAY",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "AMGN.OQ": {
        "ticker_investpy": "AMGN",
        "ticker_yfinance": "AMGN",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "TEAM.OQ": {
        "ticker_investpy": "TEAM",
        "ticker_yfinance": "TEAM",
        "Start_Date": "2015-12-11",
        "exchange_name": "NASDAQ",
    },
    "MCHP.OQ": {
        "ticker_investpy": "MCHP",
        "ticker_yfinance": "MCHP",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "BIDU.OQ": {
        "ticker_investpy": "BIDU",
        "ticker_yfinance": "BIDU",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
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
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "XEL.OQ": {
        "ticker_investpy": "XEL",
        "ticker_yfinance": "XEL",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "CERN.OQ": {
        "ticker_investpy": "CERN",
        "ticker_yfinance": "CERN",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "CDW.OQ": {
        "ticker_investpy": "CDW",
        "ticker_yfinance": "CDW",
        "Start_Date": "2013-06-27",
        "exchange_name": "NASDAQ",
    },
    "AMAT.OQ": {
        "ticker_investpy": "AMAT",
        "ticker_yfinance": "AMAT",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "CPRT.OQ": {
        "ticker_investpy": "CPRT",
        "ticker_yfinance": "CPRT",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "BKNG.OQ": {
        "ticker_investpy": "BKNG",
        "ticker_yfinance": "BKNG",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "CTSH.OQ": {
        "ticker_investpy": "CTSH",
        "ticker_yfinance": "CTSH",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "AEP.OQ": {
        "ticker_investpy": "AEP",
        "ticker_yfinance": "AEP",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "CHKP.OQ": {
        "ticker_investpy": "CHKP",
        "ticker_yfinance": "CHKP",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "PEP.OQ": {
        "ticker_investpy": "PEP",
        "ticker_yfinance": "PEP",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "FB.OQ": {
        "ticker_investpy": "FB",
        "ticker_yfinance": "FB",
        "Start_Date": "2012-05-18",
        "exchange_name": "NASDAQ",
    },
    "JD.OQ": {
        "ticker_investpy": "JD",
        "ticker_yfinance": "JD",
        "Start_Date": "2014-05-23",
        "exchange_name": "NASDAQ",
    },
    "ANSS.OQ": {
        "ticker_investpy": "ANSS",
        "ticker_yfinance": "ANSS",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "VTRS.OQ": {
        "ticker_investpy": "VTRS",
        "ticker_yfinance": "VTRS",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "INTU.OQ": {
        "ticker_investpy": "INTU",
        "ticker_yfinance": "INTU",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "LILA.OQ": {
        "ticker_investpy": "LILA",
        "ticker_yfinance": "LILA",
        "Start_Date": "2015-07-01",
        "exchange_name": "NASDAQ",
    },
    "CSGP.OQ": {
        "ticker_investpy": "CSGP",
        "ticker_yfinance": "CSGP",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "NVDA.OQ": {
        "ticker_investpy": "NVDA",
        "ticker_yfinance": "NVDA",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "GOOGL.OQ": {
        "ticker_investpy": "GOOGL",
        "ticker_yfinance": "GOOGL",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "VOD.OQ": {
        "ticker_investpy": "VOD",
        "ticker_yfinance": "VOD",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "NFLX.OQ": {
        "ticker_investpy": "NFLX",
        "ticker_yfinance": "NFLX",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "JBHT.OQ": {
        "ticker_investpy": "JBHT",
        "ticker_yfinance": "JBHT",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "XRAY.OQ": {
        "ticker_investpy": "XRAY",
        "ticker_yfinance": "XRAY",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "DLTR.OQ": {
        "ticker_investpy": "DLTR",
        "ticker_yfinance": "DLTR",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "VRTX.OQ": {
        "ticker_investpy": "VRTX",
        "ticker_yfinance": "VRTX",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "COST.OQ": {
        "ticker_investpy": "COST",
        "ticker_yfinance": "COST",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "IDXX.OQ": {
        "ticker_investpy": "IDXX",
        "ticker_yfinance": "IDXX",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "TTWO.OQ": {
        "ticker_investpy": "TTWO",
        "ticker_yfinance": "TTWO",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "FISV.OQ": {
        "ticker_investpy": "FISV",
        "ticker_yfinance": "FISV",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "AKAM.OQ": {
        "ticker_investpy": "AKAM",
        "ticker_yfinance": "AKAM",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "ADBE.OQ": {
        "ticker_investpy": "ADBE",
        "ticker_yfinance": "ADBE",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "NTES.OQ": {
        "ticker_investpy": "NTES",
        "ticker_yfinance": "NTES",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "BIIB.OQ": {
        "ticker_investpy": "BIIB",
        "ticker_yfinance": "BIIB",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "SWKS.OQ": {
        "ticker_investpy": "SWKS",
        "ticker_yfinance": "SWKS",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "SNPS.OQ": {
        "ticker_investpy": "SNPS",
        "ticker_yfinance": "SNPS",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "AAL.OQ": {
        "ticker_investpy": "AAL",
        "ticker_yfinance": "AAL",
        "Start_Date": "2013-12-09",
        "exchange_name": "NASDAQ",
    },
    "EXC.OQ": {
        "ticker_investpy": "EXC",
        "ticker_yfinance": "EXC",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "DISH.OQ": {
        "ticker_investpy": "DISH",
        "ticker_yfinance": "DISH",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "MU.OQ": {
        "ticker_investpy": "MU",
        "ticker_yfinance": "MU",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "VRSN.OQ": {
        "ticker_investpy": "VRSN",
        "ticker_yfinance": "VRSN",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "QCOM.OQ": {
        "ticker_investpy": "QCOM",
        "ticker_yfinance": "QCOM",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "TSCO.OQ": {
        "ticker_investpy": "TSCO",
        "ticker_yfinance": "TSCO",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "MELI.OQ": {
        "ticker_investpy": "MELI",
        "ticker_yfinance": "MELI",
        "Start_Date": "2007-08-13",
        "exchange_name": "NASDAQ",
    },
    "HOLX.OQ": {
        "ticker_investpy": "HOLX",
        "ticker_yfinance": "HOLX",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "WYNN.OQ": {
        "ticker_investpy": "WYNN",
        "ticker_yfinance": "WYNN",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "EXPE.OQ": {
        "ticker_investpy": "EXPE",
        "ticker_yfinance": "EXPE",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "BMRN.OQ": {
        "ticker_investpy": "BMRN",
        "ticker_yfinance": "BMRN",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "FAST.OQ": {
        "ticker_investpy": "FAST",
        "ticker_yfinance": "FAST",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "ASML.OQ": {
        "ticker_investpy": "ASML",
        "ticker_yfinance": "ASML",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "TSLA.OQ": {
        "ticker_investpy": "TSLA",
        "ticker_yfinance": "TSLA",
        "Start_Date": "2010-06-30",
        "exchange_name": "NASDAQ",
    },
    "KHC.OQ": {
        "ticker_investpy": "KHC",
        "ticker_yfinance": "KHC",
        "Start_Date": "2009-01-23",
        "exchange_name": "NASDAQ",
    },
    "MSFT.OQ": {
        "ticker_investpy": "MSFT",
        "ticker_yfinance": "MSFT",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "ORLY.OQ": {
        "ticker_investpy": "ORLY",
        "ticker_yfinance": "ORLY",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "PAYX.OQ": {
        "ticker_investpy": "PAYX",
        "ticker_yfinance": "PAYX",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "CTXS.OQ": {
        "ticker_investpy": "CTXS",
        "ticker_yfinance": "CTXS",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "PCAR.OQ": {
        "ticker_investpy": "PCAR",
        "ticker_yfinance": "PCAR",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "ULTA.OQ": {
        "ticker_investpy": "ULTA",
        "ticker_yfinance": "ULTA",
        "Start_Date": "2007-10-26",
        "exchange_name": "NASDAQ",
    },
    "CSX.OQ": {
        "ticker_investpy": "CSX",
        "ticker_yfinance": "CSX",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "DISCA.OQ": {
        "ticker_investpy": "DISCA",
        "ticker_yfinance": "DISCA",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "WDC.OQ": {
        "ticker_investpy": "WDC",
        "ticker_yfinance": "WDC",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "ADI.OQ": {
        "ticker_investpy": "ADI",
        "ticker_yfinance": "ADI",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "HSIC.OQ": {
        "ticker_investpy": "HSIC",
        "ticker_yfinance": "HSIC",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "TXN.OQ": {
        "ticker_investpy": "TXN",
        "ticker_yfinance": "TXN",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "ATVI.OQ": {
        "ticker_investpy": "ATVI",
        "ticker_yfinance": "ATVI",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "AAPL.OQ": {
        "ticker_investpy": "AAPL",
        "ticker_yfinance": "AAPL",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "CTAS.OQ": {
        "ticker_investpy": "CTAS",
        "ticker_yfinance": "CTAS",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "SPLK.OQ": {
        "ticker_investpy": "SPLK",
        "ticker_yfinance": "SPLK",
        "Start_Date": "2012-04-20",
        "exchange_name": "NASDAQ",
    },
    "HON.OQ": {
        "ticker_investpy": "HON",
        "ticker_yfinance": "HON",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "VRSK.OQ": {
        "ticker_investpy": "VRSK",
        "ticker_yfinance": "VRSK",
        "Start_Date": "2009-10-08",
        "exchange_name": "NASDAQ",
    },
    "PDD.OQ": {
        "ticker_investpy": "PDD",
        "ticker_yfinance": "PDD",
        "Start_Date": "2018-07-27",
        "exchange_name": "NASDAQ",
    },
    "TRIP.OQ": {
        "ticker_investpy": "TRIP",
        "ticker_yfinance": "TRIP",
        "Start_Date": "2011-12-08",
        "exchange_name": "NASDAQ",
    },
    "MTCH.OQ": {
        "ticker_investpy": "MTCH",
        "ticker_yfinance": "MTCH",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "WDAY.OQ": {
        "ticker_investpy": "WDAY",
        "ticker_yfinance": "WDAY",
        "Start_Date": "2012-10-15",
        "exchange_name": "NASDAQ",
    },
    "SBAC.OQ": {
        "ticker_investpy": "SBAC",
        "ticker_yfinance": "SBAC",
        "Start_Date": "2007-01-03",
        "exchange_name": "NASDAQ",
    },
    "TAMO.NS": {
        "ticker_investpy": "TAMO",
        "ticker_yfinance": "TATAMOTORS.NS",
        "Start_Date": "2007-01-01",
        "exchange_name": "NSE",
    },
    "SBI.NS": {
        "ticker_investpy": "SBI",
        "ticker_yfinance": "SBI",
        "Start_Date": "2007-01-03",
        "exchange_name": "NSE",
    },
    "NEST.NS": {
        "ticker_investpy": "NEST",
        "ticker_yfinance": "NEST",
        "Start_Date": "2010-01-11",
        "exchange_name": "NSE",
    },
    "INFY.NS": {
        "ticker_investpy": "INFY",
        "ticker_yfinance": "INFY",
        "Start_Date": "2007-01-03",
        "exchange_name": "NSE",
    },
    "TCS.NS": {
        "ticker_investpy": "TCS",
        "ticker_yfinance": "TCS",
        "Start_Date": "2013-11-01",
        "exchange_name": "NSE",
    },
    "COAL.NS": {
        "ticker_investpy": "COAL",
        "ticker_yfinance": "COAL",
        "Start_Date": "2010-11-05",
        "exchange_name": "NSE",
    },
    "HCLT.NS": {
        "ticker_investpy": "HCLT",
        "ticker_yfinance": "HCLT",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "NTPC.NS": {
        "ticker_investpy": "NTPC",
        "ticker_yfinance": "NTPC",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "ICBK.NS": {
        "ticker_investpy": "ICBK",
        "ticker_yfinance": "ICBK",
        "Start_Date": "2015-01-20",
        "exchange_name": "NSE",
    },
    "LART.NS": {
        "ticker_investpy": "LART",
        "ticker_yfinance": "LART",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "HDBK.NS": {
        "ticker_investpy": "HDBK",
        "ticker_yfinance": "HDBK",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "TISC.NS": {
        "ticker_investpy": "TISC",
        "ticker_yfinance": "TISC",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "BAJA.NS": {
        "ticker_investpy": "BAJA",
        "ticker_yfinance": "BAJA",
        "Start_Date": "2008-05-27",
        "exchange_name": "NSE",
    },
    "ASPN.NS": {
        "ticker_investpy": "ASPN",
        "ticker_yfinance": "ASPN",
        "Start_Date": "2014-06-16",
        "exchange_name": "NSE",
    },
    "REDY.NS": {
        "ticker_investpy": "REDY",
        "ticker_yfinance": "REDY",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "TEML.NS": {
        "ticker_investpy": "TEML",
        "ticker_yfinance": "TEML",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "CIPL.NS": {
        "ticker_investpy": "CIPL",
        "ticker_yfinance": "CIPL",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "ULTC.NS": {
        "ticker_investpy": "ULTC",
        "ticker_yfinance": "ULTC",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "BJFS.NS": {
        "ticker_investpy": "BJFS",
        "ticker_yfinance": "BJFS",
        "Start_Date": "2008-05-27",
        "exchange_name": "NSE",
    },
    "HDFC.NS": {
        "ticker_investpy": "HDFC",
        "ticker_yfinance": "HDFC",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "SUN.NS": {
        "ticker_investpy": "SUN",
        "ticker_yfinance": "SUN",
        "Start_Date": "2010-05-21",
        "exchange_name": "NSE",
    },
    "ITC.NS": {
        "ticker_investpy": "ITC",
        "ticker_yfinance": "ITC",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "WIPR.NS": {
        "ticker_investpy": "WIPR",
        "ticker_yfinance": "WIPR",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "GAIL.NS": {
        "ticker_investpy": "GAIL",
        "ticker_yfinance": "GAIL",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "VDAN.NS": {
        "ticker_investpy": "VDAN",
        "ticker_yfinance": "VDAN",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "PGRD.NS": {
        "ticker_investpy": "PGRD",
        "ticker_yfinance": "PGRD",
        "Start_Date": "2007-10-08",
        "exchange_name": "NSE",
    },
    "HROM.NS": {
        "ticker_investpy": "HROM",
        "ticker_yfinance": "HROM",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "AXBK.NS": {
        "ticker_investpy": "AXBK",
        "ticker_yfinance": "AXBK",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "YESB.NS": {
        "ticker_investpy": "YESB",
        "ticker_yfinance": "YESB",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "ONGC.NS": {
        "ticker_investpy": "ONGC",
        "ticker_yfinance": "ONGC",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "HLL.NS": {
        "ticker_investpy": "HLL",
        "ticker_yfinance": "HLL",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "APSE.NS": {
        "ticker_investpy": "APSE",
        "ticker_yfinance": "APSE",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "BRTI.NS": {
        "ticker_investpy": "BRTI",
        "ticker_yfinance": "BRTI",
        "Start_Date": "2017-07-25",
        "exchange_name": "NSE",
    },
    "VODA.NS": {
        "ticker_investpy": "VODA",
        "ticker_yfinance": "VODA",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "BFRG.NS": {
        "ticker_investpy": "BFRG",
        "ticker_yfinance": "BFRG",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "CUMM.NS": {
        "ticker_investpy": "CUMM",
        "ticker_yfinance": "CUMM",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "CAST.NS": {
        "ticker_investpy": "CAST",
        "ticker_yfinance": "CAST",
        "Start_Date": "2007-08-16",
        "exchange_name": "NSE",
    },
    "ASOK.NS": {
        "ticker_investpy": "ASOK",
        "ticker_yfinance": "ASOK",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "AUFI.NS": {
        "ticker_investpy": "AUFI",
        "ticker_yfinance": "AUFI",
        "Start_Date": "2017-07-10",
        "exchange_name": "NSE",
    },
    "SRTR.NS": {
        "ticker_investpy": "SRTR",
        "ticker_yfinance": "SRTR",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "MAXI.NS": {
        "ticker_investpy": "MAXI",
        "ticker_yfinance": "MAXI",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "BATA.NS": {
        "ticker_investpy": "BATA",
        "ticker_yfinance": "BATA",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "MINT.NS": {
        "ticker_investpy": "MINT",
        "ticker_yfinance": "MINT",
        "Start_Date": "2009-11-18",
        "exchange_name": "NSE",
    },
    "COFO.NS": {
        "ticker_investpy": "COFO",
        "ticker_yfinance": "COFO",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "TVSM.NS": {
        "ticker_investpy": "TVSM",
        "ticker_yfinance": "TVSM",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "PAGE.NS": {
        "ticker_investpy": "PAGE",
        "ticker_yfinance": "PAGE",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "CCRI.NS": {
        "ticker_investpy": "CCRI",
        "ticker_yfinance": "CCRI",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "ESCO.NS": {
        "ticker_investpy": "ESCO",
        "ticker_yfinance": "ESCO",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "SRFL.NS": {
        "ticker_investpy": "SRFL",
        "ticker_yfinance": "SRFL",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "CNBK.NS": {
        "ticker_investpy": "CNBK",
        "ticker_yfinance": "CNBK",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "TTPW.NS": {
        "ticker_investpy": "TTPW",
        "ticker_yfinance": "TTPW",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "ZEE.NS": {
        "ticker_investpy": "ZEE",
        "ticker_yfinance": "ZEE",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "MNFL.NS": {
        "ticker_investpy": "MNFL",
        "ticker_yfinance": "MNFL",
        "Start_Date": "2010-07-01",
        "exchange_name": "NSE",
    },
    "FED.NS": {
        "ticker_investpy": "FED",
        "ticker_yfinance": "FED",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "GLEN.NS": {
        "ticker_investpy": "GLEN",
        "ticker_yfinance": "GLEN",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "CHLA.NS": {
        "ticker_investpy": "CHLA",
        "ticker_yfinance": "CHLA",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "AMAR.NS": {
        "ticker_investpy": "AMAR",
        "ticker_yfinance": "AMAR",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "APLO.NS": {
        "ticker_investpy": "APLO",
        "ticker_yfinance": "APLO",
        "Start_Date": "2007-01-03",
        "exchange_name": "NSE",
    },
    "BAJE.NS": {
        "ticker_investpy": "BAJE",
        "ticker_yfinance": "BAJE",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "SAIL.NS": {
        "ticker_investpy": "SAIL",
        "ticker_yfinance": "SAIL",
        "Start_Date": "2017-11-17",
        "exchange_name": "NSE",
    },
    "MMFS.NS": {
        "ticker_investpy": "MMFS",
        "ticker_yfinance": "MMFS",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "BLKI.NS": {
        "ticker_investpy": "BLKI",
        "ticker_yfinance": "BLKI",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "PWFC.NS": {
        "ticker_investpy": "PWFC",
        "ticker_yfinance": "PWFC",
        "Start_Date": "2007-02-26",
        "exchange_name": "NSE",
    },
    "TOPO.NS": {
        "ticker_investpy": "TOPO",
        "ticker_yfinance": "TOPO",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "BOB.NS": {
        "ticker_investpy": "BOB",
        "ticker_yfinance": "BOB",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "GODR.NS": {
        "ticker_investpy": "GODR",
        "ticker_yfinance": "GODR",
        "Start_Date": "2010-01-06",
        "exchange_name": "NSE",
    },
    "LTFH.NS": {
        "ticker_investpy": "LTFH",
        "ticker_yfinance": "LTFH",
        "Start_Date": "2011-08-16",
        "exchange_name": "NSE",
    },
    "INBF.NS": {
        "ticker_investpy": "INBF",
        "ticker_yfinance": "INBF",
        "Start_Date": "2013-07-24",
        "exchange_name": "NSE",
    },
    "BOI.NS": {
        "ticker_investpy": "BOI",
        "ticker_yfinance": "BOI",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "JNSP.NS": {
        "ticker_investpy": "JNSP",
        "ticker_yfinance": "JNSP",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "IDFB.NS": {
        "ticker_investpy": "IDFB",
        "ticker_yfinance": "IDFB",
        "Start_Date": "2015-11-09",
        "exchange_name": "NSE",
    },
    "SUTV.NS": {
        "ticker_investpy": "SUTV",
        "ticker_yfinance": "SUTV",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "VOLT.NS": {
        "ticker_investpy": "VOLT",
        "ticker_yfinance": "VOLT",
        "Start_Date": "2007-01-03",
        "exchange_name": "NSE",
    },
    "MGAS.NS": {
        "ticker_investpy": "MGAS",
        "ticker_yfinance": "MGAS",
        "Start_Date": "2016-07-04",
        "exchange_name": "NSE",
    },
    "RECM.NS": {
        "ticker_investpy": "RECM",
        "ticker_yfinance": "RECM",
        "Start_Date": "2008-03-13",
        "exchange_name": "NSE",
    },
    "GMRI.NS": {
        "ticker_investpy": "GMRI",
        "ticker_yfinance": "GMRI",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "BHEL.NS": {
        "ticker_investpy": "BHEL",
        "ticker_yfinance": "BHEL",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "LICH.NS": {
        "ticker_investpy": "LICH",
        "ticker_yfinance": "LICH",
        "Start_Date": "2007-01-03",
        "exchange_name": "NSE",
    },
    "EXID.NS": {
        "ticker_investpy": "EXID",
        "ticker_yfinance": "EXID",
        "Start_Date": "2007-01-02",
        "exchange_name": "NSE",
    },
    "TRCE.NS": {
        "ticker_investpy": "TRCE",
        "ticker_yfinance": "TRCE",
        "Start_Date": "2008-01-02",
        "exchange_name": "NSE",
    },
    "UNBK.NS": {
            "ticker_investpy": "UNBK",
            "ticker_yfinance": "UNIONBANK.NS",
            "Start_Date": "2008-01-02",
            "exchange_name": "NSE",
    },
    "MRTI.NS": {
                "ticker_investpy": "MRTI",
                "ticker_yfinance": "MARUTI.NS",
                "Start_Date": "2008-01-02",
                "exchange_name": "NSE",
        },
    "TITN.NS": {
                "ticker_investpy": "TITN",
                "ticker_yfinance": "TITAN.NS",
                "Start_Date": "2008-01-02",
                "exchange_name": "NSE",
        },
    "MAHM.NS": {
                "ticker_investpy": "MAHM",
                "ticker_yfinance": "M&M.NS",
                "Start_Date": "2007-01-02",
                "exchange_name": "NSE",
        },
    "RELI.NS": {
                    "ticker_investpy": "RELI",
                    "ticker_yfinance": "RELIANCE.NS",
                    "Start_Date": "2007-01-02",
                    "exchange_name": "NSE",
                },
    "LUPN.NS": {
                    "ticker_investpy": "LUPN",
                    "ticker_yfinance": "LUPIN.NS",
                    "Start_Date": "2007-01-02",
                    "exchange_name": "NSE",
                },
    "INBK.NS": {
                        "ticker_investpy": "INBK",
                        "ticker_yfinance": "INDUSINDBK.NS",
                        "Start_Date": "2007-01-02",
                        "exchange_name": "NSE",
                },
    "KTKM.NS": {
                        "ticker_investpy": "KTKM",
                        "ticker_yfinance": "KOTAKBANK.NS",
                        "Start_Date": "2007-01-02",
                        "exchange_name": "NSE",
                },
    "BJFN.NS": {
                        "ticker_investpy": "BJFN",
                        "ticker_yfinance": "BAJFINANCE.NS",
                        "Start_Date": "2007-01-02",
                        "exchange_name": "NSE",
                },
    "INIR.NS": {
        "ticker_investpy": "INIR",
        "ticker_yfinance": "IRCTC.NS",
        "Start_Date": "2019-10-15",
        "exchange_name": "NSE",
    },
    "TAMdv.NS": {
            "ticker_investpy": "TAMdv",
            "ticker_yfinance": "TATAMTRDVR.NS",
            "Start_Date": "2013-01-14",
            "exchange_name": "NSE",
        },
    "RATB.NS": {
        "ticker_investpy": "RATB",
        "ticker_yfinance": "RBLBANK.NS",
        "Start_Date": "2016-09-01",
        "exchange_name": "NSE",
    },
    "ETH=": {
        "ticker_investpy": "ethereum",
        "ticker_yfinance": "ETH-USD",
        "Start_Date": "2018-11-01",
        "exchange_name": "Crypto",
    },
    "BTC=": {
        "ticker_investpy": "bitcoin",
        "ticker_yfinance": "BTC-USD",
        "Start_Date": "2014-07-17",
        "exchange_name": "Crypto",
    }
}

RANDOM_STATE = 835


def unscale_percentiles(df_inp, stats):
    df = df_inp.copy()
    for column in df.columns:
        df[column] = (df[column]*stats[column]["std"]+stats[column]["mean"])
    return df

def unscale_datasets_percentiles(train, val, test):
    stats = train.describe()
    return unscale_percentiles(train, stats), unscale_percentiles(val, stats), unscale_percentiles(test, stats)


def scale_percentiles(df_inp, stats):
    df = df_inp.copy()
    for column in df.columns:
        df[column] = (df[column]-stats[column]["mean"])/stats[column]["std"]   ##Use percentiles instead
    return df

def scale_datasets_percentiles(train, val, test):
    return scale_percentiles(train, stats), scale_percentiles(val, stats), scale_percentiles(test, stats)


def unscale_zscores(df_inp, stats):
    df = df_inp.copy()
    for column in df.columns:
        df[column] = (df[column]*stats[column]["std"]+stats[column]["mean"])
    return df

def unscale_datasets_zscores(train, val, test):
    stats = train.describe()
    return unscale_zscores(train, stats), unscale_zscores(val, stats), unscale_zscores(test, stats)


def scale_zscores(df_inp, stats):
    df = df_inp.copy()
    for column in df.columns:
        df[column] = (df[column]-stats[column]["mean"])/stats[column]["std"]   ##Use percentiles instead
    return df

def scale_datasets_zscores(train, val, test):
    stats = train.describe()
    return scale_zscores(train, stats), scale_zscores(val, stats), scale_zscores(test, stats)

def calc_NMI(inp, out):  #inp and out are pandas series
    def alpha(*args):
        return inp.to_numpy()

    def prior(params):
        return 1

    guess = []
    mc = MCMC(alpha_fn=alpha, alpha_fn_params_0=guess,
              target=out.to_numpy(), num_iters=1,
              prior=prior, optimize_fn=None, lower_limit=0, upper_limit=1)
    rs = mc.optimize()
    nmi = mc.analyse_results(rs, top_n=1)[1][0]
    return nmi

def add_fisher(temp, f_look):
    # for f_look in range(50, 400, 20):
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

def resample_data(temp_og, minutes):
    return temp_og.set_index("Datetime").groupby(pd.Grouper(freq=f'{minutes}Min')).agg({"Open": "first",
                                                 "Close": "last",
                                                 "Low": "min",
                                                 "High": "max",
                                                "Volume": "sum"}).reset_index()

def get_data_ETH_minute():
    filepaths = ["https://www.cryptodatadownload.com/cdd/Bitstamp_ETHUSD_2017_minute.csv",
                 "https://www.cryptodatadownload.com/cdd/Bitstamp_ETHUSD_2018_minute.csv",
                 "https://www.cryptodatadownload.com/cdd/Bitstamp_ETHUSD_2019_minute.csv",
                 "https://www.cryptodatadownload.com/cdd/Bitstamp_ETHUSD_2020_minute.csv",
                 "https://www.cryptodatadownload.com/cdd/Bitstamp_ETHUSD_2021_minute.csv"]
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
        temp_og.sort_values("Datetime", ascending=True, inplace=True)
        temp_og = temp_og[temp_og["Datetime"] > "2017-08-18"]
        temp_og.reset_index(drop=True, inplace=True)
        data = pd.concat([data, temp_og])

    if os.path.isdir('ETH_M.pkl'):
        with open(f'ETH_M.pkl', 'rb') as file:
            temp_og_imp = pickle.load(file)
        temp_og = pd.concat([temp_og_imp, data], axis=0)
        temp_og.drop_duplicates(keep="first", inplace=True)
        temp_og.reset_index(drop=True, inplace=True)
        with open(f'ETH_M.pkl', 'wb') as file:
            pickle.dump(pd.DataFrame(temp_og), file)
    else:
        with open(f'ETH_M.pkl', 'wb') as file:
            pickle.dump(pd.DataFrame(data), file)

    return data

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
            {"Name": strategy,
             "k": res.iloc[int(strategy[8:])-1]["k"],
             "x": res.iloc[int(strategy[8:])-1]["x"],
             "level": res.iloc[int(strategy[8:])-1]["level"],
             "lookback": res.iloc[int(strategy[8:])-1]["lookback"],
             #"Sortino": res.iloc[int(strategy[8:])-1]["Sortino"],
             "avg_sortino_of_trades": res.iloc[int(strategy[8:])-1]["avg_sortino_of_trades"],
             "Optimization_Years": res.iloc[int(strategy[8:])-1]["Optimization_Years"]})
    selected_params = pd.DataFrame(selected_params)
    return selected_params

def top_n_strat_params_rolling_momentum(temp, res, to_train, num_of_strat, split_date):
    if len(res)>0:
        for i in range(num_of_strat):
            k = res.iloc[i, 0]
            x = res.iloc[i, 1]
            level = res.iloc[i, 2]
            lookback = res.iloc[i, 3]
            if to_train:
                train = temp.loc[(temp["Date"] <= split_date)].reset_index().drop(['index'], axis=1)
            else:
                train = temp.loc[temp["Date"] > split_date].reset_index().drop(['index'], axis=1)
            test_strategy = vol_mom_strategy(data=train,k=k,x=x, level=level, lookback=lookback)
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

def corr_filter_momentum(temp, res, dates, date_i, num_strategies, train_monthsf):
    res.sort_values("avg_sortino_of_trades", axis=0, ascending=False, inplace=True)
    res.reset_index().drop(['index'], axis=1)
    res = res[res["avg_sortino_of_trades"]>0]
    returns, _ = top_n_strat_params_rolling_momentum(temp, res, to_train=True, num_of_strat=len(res), split_date =str(dates[date_i+int(train_monthsf/3)]))
    if returns.empty:
        return [], pd.DataFrame( columns=["k", "x", "level", "avg_sortino_of_trades", "Optimization_Years"])
    corr_mat = returns.corr()
    first_selected_strategy = 'Strategy1'
    selected_strategies = strategy_selection(returns, corr_mat, num_strategies, first_selected_strategy)
    params = selected_params(selected_strategies, res)
    res = params.drop(["Name"], axis=1)
    return (selected_strategies, res)

def corr_sortino_filter_momentum(inp):
    date_i = inp[0]
    dates = inp[1]
    temp_og = inp[2]
    res_total = inp[3]
    num_strategies = inp[4]
    train_monthsf = inp[5]

    temp = temp_og.loc[(temp_og["Date"] > str(dates[date_i])) & (temp_og["Date"] < str(dates[date_i + (int(train_monthsf/3)+1)]))].reset_index().drop(['index'], axis=1)
    res = res_total[date_i]
    x, y = corr_filter_momentum(temp, res, dates, date_i, num_strategies, train_monthsf)
    return date_i,x,y

def select_strategies_from_corr_filter_momentum(res_testf2,res_testf4,res_testf8, datesf, temp_ogf, num_opt_periodsf,num_strategiesf, ticker, save=True):
    train_monthsf = 24  #minimum optimization lookback
    for r2 in res_testf2:
        r2["Optimization_Years"] = 2
    for r4 in res_testf4:
        r4["Optimization_Years"] = 4
    for r8 in res_testf8:
        r8["Optimization_Years"] = 8
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
        results_filtered = pool.map(corr_sortino_filter_momentum, inputs)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    for i in range(len(datesf)-(int(train_monthsf/3)+1)):
        ss_test[results_filtered[i][0]] = results_filtered[i][1]
        res_test[results_filtered[i][0]] = results_filtered[i][2]

    if save==True:
        with open(f'{ticker}_Momentum/SelectedStrategies/{ticker}_OptPeriods_{num_opt_periodsf}_Selected_Strategies_ss.pkl', 'wb') as file:
            pickle.dump(ss_test, file)
        with open(f'{ticker}_Momentum/SelectedStrategies/{ticker}_OptPeriods_{num_opt_periodsf}_Selected_Strategies_res.pkl', 'wb') as file:
            pickle.dump(res_test, file)

    return ss_test, res_test

def backtest_mr_stdev_for_avg_sortino(input):

    strat = vol_mean_reverting_stdev_strategy(input[0], levelb=input[1], lookbackb=input[2], band_buy=input[3], xtp=input[4], ktp = input[5],leveltp=input[6],lookbacktp=input[7], xsl=input[8], ksl = input[9],levelsl=input[10],lookbacksl=input[11])
    strat.generate_signals()
    strat.signal_performance(10000, 6)
    return {"levelb":input[1], "lookbackb":input[2], "band_buy":input[3], "xtp":input[4], "ktp" : input[5],
            "leveltp":input[6],"lookbacktp":input[7], "xsl":input[8], "ksl" : input[9],"levelsl":input[10],"lookbacksl":input[11], "avg_sortino_of_trades":strat.daywise_performance['avg_sortino_of_trades']}

def BFO_vol_mr_stdev_strategy(temp_og, kmin = 21, kmax = 250, xtp_min = 1.5, xtp_max = 5, xsl_min = -5, xsl_max = -1):

    data = [temp_og]
    # band_list = [0.01,0.02]
    # k_list = [i for i in range(kmin, kmax+1)]
    # xtp_list = [i/10 for i in range(int(xtp_min*10), int(xtp_max*10)+5,5)]
    # xsl_list = [i/10 for i in range(int(xsl_min*10),int(xsl_max*10+5),5)]
    # levels = ["vah", "val", "poc"]
    # lookbacks = [2, 5, 10, 21, 42, 63, 126, 252, 504]

    band_list = [0.005,0.01]
    k_list = [21, 42, 63, 126, 252, 504]
    xtp_list = [1.5,2.5,3.5,4.5]
    xsl_list = [-5,-3,-1]
    levels = ["vah", "val", "poc"]
    lookbacks = [2, 5, 10, 21, 42, 63, 126, 252, 504]

    inputs = list(itertools.product(data, levels, lookbacks, band_list, xtp_list, k_list, levels, lookbacks, xsl_list, k_list, levels, lookbacks))

    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results = pool.map(backtest_mr_stdev_for_avg_sortino, inputs)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    results = pd.DataFrame(results)
    results.sort_values("avg_sortino_of_trades", ascending=False, inplace=True)
    results.reset_index(drop=True,inplace=True)
    return results

def add_volume_stdevs(data):
    for k in [21, 42, 63, 126, 252, 504]:
        for n in [2, 5, 10, 21, 42, 63, 126, 252, 504]:
            for level in ["vah", "val", "poc"]:
                data[f"ZS_{k}_{n}_{level}"] = zscore(data[f"dev_from_{level}_{n}"], k)
    return data

def add_volume_percentile_column(input):

    def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
        nrows = ((a.size - L) // S) + 1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))

    k = input[0]
    n = input[1]
    level = input[2]
    data = input[3]

    a = data[f"dev_from_{level}_{n}"].to_numpy()
    W = k + 1  # window length
    tt = pd.DataFrame(strided_app(a, W, 1))
    tt.dropna(inplace=True)
    tt.reset_index(drop=True, inplace=True)
    tt["Percentile"] = np.nan
    for i in range(len(tt)):
        tt.loc[i, "Percentile"] = stats.percentileofscore(tt.loc[i, 0:k], tt.loc[i, k])
    data[f"Percentile_{k}_{n}_{level}"] = np.pad(tt["Percentile"].to_numpy(), len(data) - len(tt),
                                                 'constant', constant_values=(np.nan))[
                                          :-(len(data) - len(tt))]
    return data

def add_volume_percentiles(data):

    data_list = [data.copy()]
    k_list = [21, 42, 63, 126, 252, 504]
    n_list = [2, 5, 10, 21, 42, 63, 126, 252, 504]
    levels = ["vah", "val", "poc"]
    inputs = list(itertools.product(k_list,n_list,levels,data_list))

    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results = pool.map(add_volume_percentile_column, inputs)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    data_return  = data.copy()
    for result in results:
        data_return = pd.concat([data_return, result],axis=1)
        data_return = data_return.loc[:, ~data_return.columns.duplicated()]
    return data_return


def zscore(x, window):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z

def select_all_strategies_mr_stdev(train_monthsf, datesf, temp_ogf, ticker, save=True):
    res_test = []
    for date_i in range(len(datesf) - (int(train_monthsf / 3) + 1)):
        temp = temp_ogf.loc[(temp_ogf["Date"] > str(datesf[date_i])) & (temp_ogf["Date"] < str(
            datesf[date_i + int(train_monthsf / 3)]))].reset_index().drop(['index'], axis=1)
        res_test.append(BFO_vol_mr_stdev_strategy(temp))

    if save == True:
        with open(f'{ticker}_MRStdev/SelectedStrategies/{ticker}_TrainYrs_{int(train_monthsf / 12)}_All_Strategies.pkl',
                  'wb') as file:
            pickle.dump(res_test, file)

    return res_test

def select_all_strategies_momentum(train_monthsf, datesf, temp_ogf, ticker, save=True):
    res_test = []
    for date_i in range(len(datesf) - (int(train_monthsf / 3) + 1)):
        temp = temp_ogf.loc[(temp_ogf["Date"] > str(datesf[date_i])) & (temp_ogf["Date"] < str(
            datesf[date_i + int(train_monthsf / 3)]))].reset_index().drop(['index'], axis=1)
        res_test.append(BFO_vol_mom_strategy(temp))

    if save == True:
        with open(f'{ticker}_Momentum/SelectedStrategies/{ticker}_TrainYrs_{int(train_monthsf / 12)}_All_Strategies.pkl',
                  'wb') as file:
            pickle.dump(res_test, file)

    return res_test


def backtest_mom_for_avg_sortino(input):

    strat = vol_mom_strategy(data = input[0], kb = input[1], xb = input[2], levelb = input[3], lookbackb = input[4], ks = input[5], xs = input[6], levels = input[7], lookbacks = input[8])
    strat.generate_signals()
    strat.signal_performance(10000, 6)
    return {"kb" : input[1], "xb" : input[2], "levelb" : input[3], "lookbackb" : input[4], "ks" : input[5], "xs" : input[6], "levels" : input[7], "lookbacks" : input[8],
            "avg_sortino_of_trades": strat.daywise_performance['avg_sortino_of_trades']}

def BFO_vol_mom_strategy(temp_og, kmax = 9, xmax = 5):

    data = [temp_og]
    k_list = [i for i in range(kmax+1)]
    x_list = [i/100 for i in range(xmax+1)]
    levels = ["vah", "val", "poc"]
    lookbacks = [2, 5, 10, 21, 42, 63, 126, 252, 504]
    inputs = list(itertools.product(data,k_list,x_list,levels,lookbacks,k_list,x_list,levels,lookbacks))

    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results = pool.map(backtest_mom_for_avg_sortino, inputs)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    results = pd.DataFrame(results)
    results.sort_values("avg_sortino_of_trades", ascending=False, inplace=True)
    results.reset_index(drop=True,inplace=True)
    return results

class vol_mean_reverting_stdev_strategy:
    metrics = {'Sharpe': 'Sharpe Ratio',
               'CAGR': 'Compound Average Growth Rate',
               'MDD': 'Maximum Drawdown',
               'NHR': 'Normalized Hit Ratio',
               'OTS': 'Optimal trade size'}

    def __init__(self, data, levelb, lookbackb, band_buy, xtp, ktp,leveltp,lookbacktp, xsl, ksl,levelsl,lookbacksl, start=None, end=None):



        self.bbh = band_buy
        self.bbl = -band_buy
        self.ktp = ktp
        self.ksl = ksl
        self.xtp = xtp
        self.xsl = xsl
        self.levelb = levelb
        self.levelsl = levelsl
        self.leveltp = leveltp
        self.nb = lookbackb
        self.ntp = lookbacktp
        self.nsl = lookbacksl
        self.data = data[["Date", "Close", f"dev_from_{levelb}_{lookbackb}", f"ZS_{ksl}_{lookbacksl}_{levelsl}", f"ZS_{ktp}_{lookbacktp}_{leveltp}"]] # the dataframe
        self.data['yr'] = self.data['Date'].dt.year
        self.n_days = (self.data.Date.iloc[-1] - self.data.Date.iloc[0])
        self.start = start  # the beginning date of the sample period
        self.end = end  # the ending date of the sample period

    def generate_signals(self, charts=False):

        self.data = self.data.loc[:, ~self.data.columns.duplicated()]
        self.data[f"Zscore_SL"] = self.data[f"ZS_{self.ksl}_{self.nsl}_{self.levelsl}"]
        self.data[f"Zscore_TP"] = self.data[f"ZS_{self.ktp}_{self.ntp}_{self.leveltp}"]

        self.data["lb"] = self.bbl
        self.data["ub"] = self.bbh
        self.data.dropna()
        self.data.reset_index(inplace=True)
        self.data = self.data.drop(['index'], axis=1)

        ## creating signal
        ##Upper and lower bounds
        buy_mask = (self.data[f"dev_from_{self.levelb}_{self.nb}"]>=self.data["lb"])&(self.data[f"dev_from_{self.levelb}_{self.nb}"]<=self.data["ub"])
        sell_mask = (self.data[f"Zscore_TP"]>=self.xtp)|(self.data[f"Zscore_SL"]<=self.xsl)

        #buy_mask = ((self.data["fisher"] >= self.data["lb"]) & (self.data["fisher"]>=self.data["ub"]))
        #sell_mask = ((self.data["fisher"] < self.data["lb"])|(self.data["fisher"]<self.data["ub"])&(self.data["fisher"]>=self.data["lb"]))

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

        # Closing positions at end of time period
        self.data["signal"][-1:] = 0

        buy_plot_mask = ((self.data.signal.shift(-1) == bval) & (self.data.signal == sval))
        sell_plot_mask = ((self.data.signal.shift(-1) == sval) & (self.data.signal == bval))

        #initialize with a buy position
        buy_plot_mask[0] = True

        mask = ((self.data.signal == 1) & (self.data.signal.shift(1)==0)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

        #self.data.to_csv("Test.csv")

        ## display chart
        if charts:
            plt.plot(self.data['Date'].to_numpy(), self.data['Close'].to_numpy(), color='black', label='Price')
            plt.plot(self.data.loc[buy_plot_mask, 'Date'].to_numpy(), self.data.loc[buy_plot_mask, 'Close'].to_numpy(), r'^', ms=15,
                     label="Entry Signal", color='green', markeredgecolor='k', markeredgewidth=1)
            plt.plot(self.data.loc[sell_plot_mask, 'Date'].to_numpy(), self.data.loc[sell_plot_mask, 'Close'].to_numpy(), r'^', ms=15,
                     label="Exit Signal", color='red', markeredgecolor='k', markeredgewidth=1)
            plt.title('Strategy Backtest')
            plt.legend(loc=0)
            d_color = {}
            d_color[1] = '#90ee90'  ## light green
            d_color[-1] = "#ffcccb"  ## light red
            d_color[0] = '#ffffff'

            j = 0
            for i in range(1, self.data.shape[0]):
                if np.isnan(self.data.signal[i - 1]):
                    j = i
                elif (self.data.signal[i - 1] == self.data.signal[i]) and (i < (self.data.shape[0] - 1)):
                    continue
                else:
                    plt.axvspan(self.data['Date'][j], self.data['Date'][i],
                                alpha=0.5, color=d_color[self.data.signal[i - 1]], label="interval")
                    j = i
            plt.show()


            # plt.plot(self.data['Date'], self.data[f"dev_from_{self.level}_{self.n}"], color='black', label=f"dev_from_{self.level}_{self.n}")
            # plt.plot(self.data['Date'], 0.2*self.data["signal"],color='purple', label=f"Signal")
            # plt.plot(self.data['Date'], self.data['lb'], color='green', label='Lower Bound for Buy')
            # plt.plot(self.data['Date'], self.data['ub'], color='red', label='Upper Bound for Buy')
            #
            # # plt.plot(self.data.loc[buy_plot_mask]['Date'], self.data.loc[buy_plot_mask][f"dev_from_{self.level}_{self.n}"],
            # #          r'^', ms=15,
            # #          label="Entry Signal", color='green', markeredgecolor='k', markeredgewidth=1)
            # # plt.plot(self.data.loc[sell_plot_mask]['Date'], self.data.loc[sell_plot_mask][f"dev_from_{self.level}_{self.n}"],
            # #          r'^', ms=15,
            # #          label="Exit Signal", color='red', markeredgecolor='k', markeredgewidth=1)
            # plt.title('Strategy Backtest')
            # plt.legend(loc=0)
            # d_color = {}
            # d_color[1] = '#90ee90'  ## light green
            # d_color[-1] = "#ffcccb"  ## light red
            # d_color[0] = '#ffffff'
            #
            # j = 0
            # for i in range(1, self.data.shape[0]):
            #     if np.isnan(self.data.signal[i - 1]):
            #         j = i
            #     elif (self.data.signal[i - 1] == self.data.signal[i]) and (i < (self.data.shape[0] - 1)):
            #         continue
            #     else:
            #         plt.axvspan(self.data['Date'][j], self.data['Date'][i],
            #                     alpha=0.5, color=d_color[self.data.signal[i - 1]], label="interval")
            #         j = i
            # plt.show()
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
        self.data["S_Return"] = self.data["S_Return"] + (self.int/25200)*(1-self.data['signal'].shift(1))
        self.data.dropna(inplace=True)
        self.data['Market_Return'] = self.data['Return'].expanding().sum()
        self.data['Strategy_Return'] = self.data['S_Return'].expanding().sum()
        self.data['Portfolio Value'] = ((self.data['Strategy_Return'] + 1) * self.allocation)
        # self.data['Wins'] = np.where(self.data['S_Return'] > 0, 1, 0)
        # self.data['Losses'] = np.where(self.data['S_Return'] < 0, 1, 0)

        ## Daywise Performance
        d_perform = {}
        num_trades = self.data.iloc[-1]["trade_num"]
        sortino_of_trades = []
        for i in range(1, num_trades + 1):
            try:
                if self.data.loc[(self.data["trade_num"] == i) & (self.data["signal"] == 1) & (self.data["S_Return"] < 0), "S_Return"].std() != 0:
                    sortino_of_trades.append(
                        (self.data.loc[self.data["trade_num"] == i, "S_Return"].mean() - 0.06 / 252) / self.data.loc[
                            (self.data["trade_num"] == i) & (self.data["signal"] == 1) & (self.data["S_Return"] < 0), "S_Return"].std() * (
                                    252 ** .5))
                else:
                    sortino_of_trades.append(5)
            except:
                sortino_of_trades.append(0)
        if len(sortino_of_trades) > 0:
            d_perform['avg_sortino_of_trades'] = sum(sortino_of_trades) / len(sortino_of_trades)
        else:
            d_perform['avg_sortino_of_trades'] = 0
        # d_perform['TotalWins'] = self.data['Wins'].sum()
        # d_perform['TotalLosses'] = self.data['Losses'].sum()
        # d_perform['TotalTrades'] = d_perform['TotalWins'] + d_perform['TotalLosses']
        # if d_perform['TotalTrades']==0:
        #     d_perform['HitRatio'] = 0
        # else:
        #     d_perform['HitRatio'] = round(d_perform['TotalWins'] / d_perform['TotalTrades'], 2)
        # d_perform['SharpeRatio'] = (self.data["S_Return"].mean() -0.06/252)/ self.data["S_Return"].std() * (252 ** .5)
        # d_perform['StDev Annualized Downside Return'] = self.data.loc[self.data["S_Return"]<0, "S_Return"].std() * (252 ** .5)
        # #print(self.data["S_Return"])#.isnull().sum().sum())
        # if math.isnan(d_perform['StDev Annualized Downside Return']):
        #     d_perform['StDev Annualized Downside Return'] = 0.0
        # #print(d_perform['StDev Annualized Downside Return'])
        # if d_perform['StDev Annualized Downside Return'] != 0.0:
        #     d_perform['SortinoRatio'] = (self.data["S_Return"].mean()-0.06/252)*252/ d_perform['StDev Annualized Downside Return']
        # else:
        #     d_perform['SortinoRatio'] = 0
        # if len(self.data['Strategy_Return'])!=0:
        #     d_perform['CAGR'] = (1 + self.data['Strategy_Return']).iloc[-1] ** (365.25 / self.n_days.days) - 1
        # else:
        #     d_perform['CAGR'] = 0
        # d_perform['MaxDrawdown'] = (1.0 - self.data['Portfolio Value'] / self.data['Portfolio Value'].cummax()).max()

        self.daywise_performance = pd.Series(d_perform)
        #
        # ## Tradewise performance
        # ecdf = self.data[self.data["signal"] == 1]
        # trade_wise_results = []
        # if len(ecdf) > 0:
        #     for i in range(max(ecdf['trade_num'])):
        #         trade_num = i + 1
        #         entry = ecdf[ecdf["trade_num"] == trade_num].iloc[0]["Close"]
        #         exit = ecdf[ecdf["trade_num"] == trade_num].iloc[-1]["Close"]
        #         trade_wise_results.append({'Trade Number': trade_num, 'Entry': entry, 'Exit': exit})
        # trade_wise_results = pd.DataFrame(trade_wise_results)
        # d_tp = {}
        # if len(trade_wise_results) > 0:
        #     trade_wise_results["Win/Loss"] = np.where(trade_wise_results["Exit"] > trade_wise_results["Entry"], "Win",
        #                                               "Loss")
        #     trade_wise_results["Return on Trade"] = trade_wise_results["Exit"] / trade_wise_results["Entry"] - 1
        #     d_tp["TotalWins"] = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Win"])
        #     d_tp["TotalLosses"] = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"])
        #     d_tp['TotalTrades'] = d_tp["TotalWins"] + d_tp["TotalLosses"]
        #     if d_tp['TotalTrades'] == 0:
        #         d_tp['HitRatio'] = 0
        #     else:
        #         d_tp['HitRatio'] = round(d_tp['TotalWins'] / d_tp['TotalTrades'], 4)
        #     d_tp['AvgWinRet'] = np.round(
        #         trade_wise_results[trade_wise_results["Win/Loss"] == "Win"]["Return on Trade"].mean(), 4)
        #     if math.isnan(d_tp['AvgWinRet']):
        #         d_tp['AvgWinRet'] = 0.0
        #     d_tp['AvgLossRet'] = np.round(
        #         trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"]["Return on Trade"].mean(), 4)
        #     if math.isnan(d_tp['AvgLossRet']):
        #         d_tp['AvgLossRet'] = 0.0
        #     if d_tp['AvgLossRet'] != 0:
        #         d_tp['WinByLossRet'] = np.round(abs(d_tp['AvgWinRet'] / d_tp['AvgLossRet']), 2)
        #     else:
        #         d_tp['WinByLossRet'] = 0
        #     if math.isnan(d_tp['WinByLossRet']):
        #         d_tp['WinByLossRet'] = 0.0
        #     if math.isinf(d_tp['WinByLossRet']):
        #         d_tp['WinByLossRet'] = 0.0
        # else:
        #     d_tp["TotalWins"] = 0
        #     d_tp["TotalLosses"] = 0
        #     d_tp['TotalTrades'] = 0
        #     d_tp['HitRatio'] = 0
        #     d_tp['AvgWinRet'] = 0
        #     d_tp['AvgLossRet'] = 0
        #     d_tp['WinByLossRet'] = 0
        # self.tradewise_performance = pd.Series(d_tp)

        return self.data[['Date', 'Close', 'signal', 'Return', 'S_Return', 'trade_num']]

    # @staticmethod
    # def kelly(p, b):
    #     """
    #     Static method: No object or class related arguments
    #     p: win prob, b: net odds received on wager, output(f*) = p - (1-p)/b
    #
    #     Spreadsheet example
    #         from sympy import symbols, solve, diff
    #         x = symbols('x')
    #         y = (1+3.3*x)**37 *(1-x)**63
    #         solve(diff(y, x), x)[1]
    #     Shortcut
    #         .37 - 0.63/3.3
    #     """
    #     return np.round(p - (1 - p) / b, 4)

    def plot_performance(self, allocation=1, interest_rate = 6):
        # intializing a variable for initial allocation
        # to be used to create equity curve
        self.signal_performance(allocation, interest_rate)

        # yearly performance
        # self.yearly_performance()

        # Plotting the Performance of the strategy
        plt.plot(self.data['Date'], self.data['Market_Return'], color='black', label='Market Returns')
        plt.plot(self.data['Date'], self.data['Strategy_Return'], color='blue', label='Strategy Returns')
        # plt.plot(self.data['Date'],self.data['fisher_rate'],color='red', label= 'Fisher Rate')
        plt.title('Strategy Backtest')
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()

        plt.plot(self.data['Date'], self.data['Portfolio Value'], color='blue')
        plt.title('Portfolio Value')
        plt.show()
        # return(self.data)

    # def yearly_performance(self):
    #     """
    #     Instance method
    #     Adds an instance attribute: yearly_df
    #     """
    #     _yearly_df = self.data.groupby(['yr', 'signal']).S_Return.sum().unstack()
    #     _yearly_df.rename(columns={-1.0: 'Sell', 1.0: 'Buy'}, inplace=True)
    #     _yearly_df['Return'] = _yearly_df.sum(1)
    #
    #     # yearly_df
    #     self.yearly_df = _yearly_df.style.bar(color=["#ffcccb", '#90ee90'], align='zero').format({
    #         'Sell': '{:,.2%}'.format, 'Buy': '{:,.2%}'.format, 'Return': '{:,.2%}'.format})

    # def update_metrics(self):
    #     """
    #     Called from the opt_matrix class method
    #     """
    #     d_field = {}
    #
    #     d_field['PortfolioValue'] = self.data['Portfolio Value']
    #     d_field['Sharpe'] = self.daywise_performance.SharpeRatio
    #     d_field['Sortino'] = self.daywise_performance.SortinoRatio
    #     d_field['CAGR'] = self.daywise_performance.CAGR
    #     d_field['MDD'] = self.daywise_performance.MaxDrawdown
    #     d_field['NHR'] = self.tradewise_performance.NormHitRatio
    #     #d_field['OTS'] = self.tradewise_performance.OptimalTradeSize
    #     d_field['AvgWinLoss'] = self.tradewise_performance.WinByLossRet
    #
    #     return d_field

    # @classmethod
    # def opt_matrix(cls, data, buy_fish, sell_fish, metrics, optimal_sol=True):
    #     """
    #
    #     """
    #     c_green = sns.light_palette("green", as_cmap=True)
    #     c_red = sns.light_palette("red", as_cmap=True)
    #
    #     d_mats = {m: [] for m in metrics}
    #
    #
    #     for lows in buy_fish:
    #         d_row = {m: [] for m in metrics}
    #         for highs in sell_fish:
    #             # if highs>=lows:
    #             obj = cls(data, zone_high=highs, zone_low=lows)  ## object being created from the class
    #             obj.generate_signals(charts=False)
    #             obj.signal_performance(10000, 6)
    #             d_field = obj.update_metrics()
    #             for m in metrics: d_row[m].append(d_field.get(m, np.nan))
    #             # else:
    #             #     for m in metrics: d_row[m].append(0)
    #         for m in metrics: d_mats[m].append(d_row[m])
    #
    #     d_df = {m: pd.DataFrame(d_mats[m], index=buy_fish, columns=sell_fish) for m in metrics}
    #
    #     def optimal(_df):
    #
    #         _df = _df.stack().rank()
    #         _df = (_df - _df.mean()) / _df.std()
    #         return _df.unstack()
    #
    #     if optimal_sol:
    #         # d_df['Metric'] = 0
    #         # if 'Sortino' in metrics: d_df['Metric'] += optimal(d_df['Sortino'])
    #         # if 'PVal' in metrics: d_df['Metric'] += optimal(d_df['PortfolioValue'])
    #         # if 'Sharpe' in metrics: d_df['Metric'] += 2 * optimal(d_df['Sharpe'])
    #         # if 'NHR' in metrics: d_df['Metric'] += optimal(d_df['NHR'])
    #         # if 'CAGR' in metrics: d_df['Metric'] += optimal(d_df['CAGR'])
    #         # if 'MDD' in metrics: d_df['Metric'] -= 2 * optimal(d_df['MDD'])
    #         # d1 = pd.DataFrame(d_df['Metric'])
    #         # val = np.amax(d1.to_numpy())
    #         # bf = d1.index[np.where(d1 == val)[0][0]]
    #         # sf = d1.columns[np.where(d1 == val)[1][0]]
    #
    #         #******
    #         d2 = pd.DataFrame(d_df[metrics[0]])
    #         val = np.amax(d2.to_numpy())
    #         bf = d2.index[np.where(d2 == val)[0][0]]
    #         sf = d2.columns[np.where(d2 == val)[1][0]]
    #         #******
    #         #return d_df
    #
    #         #print(f"Most optimal pair is Lower Bound:{bf}, Upper Bound:{sf}, with max {metrics[0]}:{val}")
    #         #metrics.insert(0, 'Signal')
    #
    #     # for m in metrics:
    #     #     display(HTML(d_df[m].style.background_gradient(axis=None, cmap=
    #     #     c_red if m == "MDD" else c_green).format(
    #     #         ("{:,.2}" if m in ["Sharpe", "Signal"] else "{:.2%}")).set_caption(m).render()))
    #
    #
    #     return (bf, sf, val)


class vol_mean_reverting_percentile_strategy:
    metrics = {'Sharpe': 'Sharpe Ratio',
               'CAGR': 'Compound Average Growth Rate',
               'MDD': 'Maximum Drawdown',
               'NHR': 'Normalized Hit Ratio',
               'OTS': 'Optimal trade size'}

    def __init__(self, data, levelb, lookbackb, band_buy, xtp, ktp,leveltp,lookbacktp, xsl, ksl,levelsl,lookbacksl, start=None, end=None):

        self.bbh = band_buy
        self.bbl = -band_buy
        self.ktp = ktp
        self.ksl = ksl
        self.xtp = xtp
        self.xsl = xsl
        self.levelb = levelb
        self.levelsl = levelsl
        self.leveltp = leveltp
        self.nb = lookbackb
        self.ntp = lookbacktp
        self.nsl = lookbacksl
        self.data = data[["Date", "Close", f"dev_from_{levelb}_{lookbackb}", f"Percentile_{ksl}_{lookbacksl}_{levelsl}", f"Percentile_{ktp}_{lookbacktp}_{leveltp}"]] # the dataframe
        self.data['yr'] = self.data['Date'].dt.year
        self.n_days = (self.data.Date.iloc[-1] - self.data.Date.iloc[0])
        self.start = start  # the beginning date of the sample period
        self.end = end  # the ending date of the sample period

    def generate_signals(self, charts=False):
        self.data = self.data.loc[:, ~self.data.columns.duplicated()]
        self.data[f"Percentile_SL"] = self.data[f"Percentile_{self.ksl}_{self.nsl}_{self.levelsl}"]

        self.data[f"Percentile_TP"] = self.data[f"Percentile_{self.ktp}_{self.ntp}_{self.leveltp}"]

        self.data["lb"] = self.bbl
        self.data["ub"] = self.bbh
        self.data.dropna()
        self.data.reset_index(inplace=True)
        self.data = self.data.drop(['index'], axis=1)

        ## creating signal
        ##Upper and lower bounds
        buy_mask = (self.data[f"dev_from_{self.levelb}_{self.nb}"]>=self.data["lb"])&(self.data[f"dev_from_{self.levelb}_{self.nb}"]<=self.data["ub"])
        sell_mask = (self.data[f"Percentile_TP"]>=self.xtp)|(self.data[f"Percentile_SL"]<=self.xsl)

        #buy_mask = ((self.data["fisher"] >= self.data["lb"]) & (self.data["fisher"]>=self.data["ub"]))
        #sell_mask = ((self.data["fisher"] < self.data["lb"])|(self.data["fisher"]<self.data["ub"])&(self.data["fisher"]>=self.data["lb"]))

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

        # Closing positions at end of time period
        self.data["signal"][-1:] = 0

        buy_plot_mask = ((self.data.signal.shift(-1) == bval) & (self.data.signal == sval))
        sell_plot_mask = ((self.data.signal.shift(-1) == sval) & (self.data.signal == bval))

        #initialize with a buy position
        buy_plot_mask[0] = True

        mask = ((self.data.signal == 1) & (self.data.signal.shift(1)==0)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

        #self.data.to_csv("Test.csv")

        ## display chart
        if charts:
            plt.plot(self.data['Date'].to_numpy(), self.data['Close'].to_numpy(), color='black', label='Price')
            plt.plot(self.data.loc[buy_plot_mask, 'Date'].to_numpy(), self.data.loc[buy_plot_mask, 'Close'].to_numpy(), r'^', ms=15,
                     label="Entry Signal", color='green', markeredgecolor='k', markeredgewidth=1)
            plt.plot(self.data.loc[sell_plot_mask, 'Date'].to_numpy(), self.data.loc[sell_plot_mask, 'Close'].to_numpy(), r'^', ms=15,
                     label="Exit Signal", color='red', markeredgecolor='k', markeredgewidth=1)
            plt.title('Strategy Backtest')
            plt.legend(loc=0)
            d_color = {}
            d_color[1] = '#90ee90'  ## light green
            d_color[-1] = "#ffcccb"  ## light red
            d_color[0] = '#ffffff'

            j = 0
            for i in range(1, self.data.shape[0]):
                if np.isnan(self.data.signal[i - 1]):
                    j = i
                elif (self.data.signal[i - 1] == self.data.signal[i]) and (i < (self.data.shape[0] - 1)):
                    continue
                else:
                    plt.axvspan(self.data['Date'][j], self.data['Date'][i],
                                alpha=0.5, color=d_color[self.data.signal[i - 1]], label="interval")
                    j = i
            plt.show()


            # plt.plot(self.data['Date'], self.data[f"dev_from_{self.level}_{self.n}"], color='black', label=f"dev_from_{self.level}_{self.n}")
            # plt.plot(self.data['Date'], 0.2*self.data["signal"],color='purple', label=f"Signal")
            # plt.plot(self.data['Date'], self.data['lb'], color='green', label='Lower Bound for Buy')
            # plt.plot(self.data['Date'], self.data['ub'], color='red', label='Upper Bound for Buy')
            #
            # # plt.plot(self.data.loc[buy_plot_mask]['Date'], self.data.loc[buy_plot_mask][f"dev_from_{self.level}_{self.n}"],
            # #          r'^', ms=15,
            # #          label="Entry Signal", color='green', markeredgecolor='k', markeredgewidth=1)
            # # plt.plot(self.data.loc[sell_plot_mask]['Date'], self.data.loc[sell_plot_mask][f"dev_from_{self.level}_{self.n}"],
            # #          r'^', ms=15,
            # #          label="Exit Signal", color='red', markeredgecolor='k', markeredgewidth=1)
            # plt.title('Strategy Backtest')
            # plt.legend(loc=0)
            # d_color = {}
            # d_color[1] = '#90ee90'  ## light green
            # d_color[-1] = "#ffcccb"  ## light red
            # d_color[0] = '#ffffff'
            #
            # j = 0
            # for i in range(1, self.data.shape[0]):
            #     if np.isnan(self.data.signal[i - 1]):
            #         j = i
            #     elif (self.data.signal[i - 1] == self.data.signal[i]) and (i < (self.data.shape[0] - 1)):
            #         continue
            #     else:
            #         plt.axvspan(self.data['Date'][j], self.data['Date'][i],
            #                     alpha=0.5, color=d_color[self.data.signal[i - 1]], label="interval")
            #         j = i
            # plt.show()

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
        self.data["S_Return"] = self.data["S_Return"] + (self.int/25200)*(1-self.data['signal'].shift(1))
        self.data.dropna(inplace=True)
        self.data['Market_Return'] = self.data['Return'].expanding().sum()
        self.data['Strategy_Return'] = self.data['S_Return'].expanding().sum()
        self.data['Portfolio Value'] = ((self.data['Strategy_Return'] + 1) * self.allocation)
        # self.data['Wins'] = np.where(self.data['S_Return'] > 0, 1, 0)
        # self.data['Losses'] = np.where(self.data['S_Return'] < 0, 1, 0)

        ## Daywise Performance
        d_perform = {}
        num_trades = self.data.iloc[-1]["trade_num"]
        sortino_of_trades = []
        for i in range(1, num_trades + 1):
            try:
                if self.data.loc[(self.data["trade_num"] == i) & (self.data["signal"] == 1) & (self.data["S_Return"] < 0), "S_Return"].std() != 0:
                    sortino_of_trades.append(
                        (self.data.loc[self.data["trade_num"] == i, "S_Return"].mean() - 0.06 / 252) / self.data.loc[
                            (self.data["trade_num"] == i) & (self.data["signal"] == 1) & (self.data["S_Return"] < 0), "S_Return"].std() * (
                                    252 ** .5))
                else:
                    sortino_of_trades.append(5)
            except:
                sortino_of_trades.append(0)
        if len(sortino_of_trades) > 0:
            d_perform['avg_sortino_of_trades'] = sum(sortino_of_trades) / len(sortino_of_trades)
        else:
            d_perform['avg_sortino_of_trades'] = 0
        # d_perform['TotalWins'] = self.data['Wins'].sum()
        # d_perform['TotalLosses'] = self.data['Losses'].sum()
        # d_perform['TotalTrades'] = d_perform['TotalWins'] + d_perform['TotalLosses']
        # if d_perform['TotalTrades']==0:
        #     d_perform['HitRatio'] = 0
        # else:
        #     d_perform['HitRatio'] = round(d_perform['TotalWins'] / d_perform['TotalTrades'], 2)
        # d_perform['SharpeRatio'] = (self.data["S_Return"].mean() -0.06/252)/ self.data["S_Return"].std() * (252 ** .5)
        # d_perform['StDev Annualized Downside Return'] = self.data.loc[self.data["S_Return"]<0, "S_Return"].std() * (252 ** .5)
        # #print(self.data["S_Return"])#.isnull().sum().sum())
        # if math.isnan(d_perform['StDev Annualized Downside Return']):
        #     d_perform['StDev Annualized Downside Return'] = 0.0
        # #print(d_perform['StDev Annualized Downside Return'])
        # if d_perform['StDev Annualized Downside Return'] != 0.0:
        #     d_perform['SortinoRatio'] = (self.data["S_Return"].mean()-0.06/252)*252/ d_perform['StDev Annualized Downside Return']
        # else:
        #     d_perform['SortinoRatio'] = 0
        # if len(self.data['Strategy_Return'])!=0:
        #     d_perform['CAGR'] = (1 + self.data['Strategy_Return']).iloc[-1] ** (365.25 / self.n_days.days) - 1
        # else:
        #     d_perform['CAGR'] = 0
        # d_perform['MaxDrawdown'] = (1.0 - self.data['Portfolio Value'] / self.data['Portfolio Value'].cummax()).max()
        self.daywise_performance = pd.Series(d_perform)
        #
        # ## Tradewise performance
        # ecdf = self.data[self.data["signal"] == 1]
        # trade_wise_results = []
        # if len(ecdf) > 0:
        #     for i in range(max(ecdf['trade_num'])):
        #         trade_num = i + 1
        #         entry = ecdf[ecdf["trade_num"] == trade_num].iloc[0]["Close"]
        #         exit = ecdf[ecdf["trade_num"] == trade_num].iloc[-1]["Close"]
        #         trade_wise_results.append({'Trade Number': trade_num, 'Entry': entry, 'Exit': exit})
        # trade_wise_results = pd.DataFrame(trade_wise_results)
        # d_tp = {}
        # if len(trade_wise_results) > 0:
        #     trade_wise_results["Win/Loss"] = np.where(trade_wise_results["Exit"] > trade_wise_results["Entry"], "Win",
        #                                               "Loss")
        #     trade_wise_results["Return on Trade"] = trade_wise_results["Exit"] / trade_wise_results["Entry"] - 1
        #     d_tp["TotalWins"] = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Win"])
        #     d_tp["TotalLosses"] = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"])
        #     d_tp['TotalTrades'] = d_tp["TotalWins"] + d_tp["TotalLosses"]
        #     if d_tp['TotalTrades'] == 0:
        #         d_tp['HitRatio'] = 0
        #     else:
        #         d_tp['HitRatio'] = round(d_tp['TotalWins'] / d_tp['TotalTrades'], 4)
        #     d_tp['AvgWinRet'] = np.round(
        #         trade_wise_results[trade_wise_results["Win/Loss"] == "Win"]["Return on Trade"].mean(), 4)
        #     if math.isnan(d_tp['AvgWinRet']):
        #         d_tp['AvgWinRet'] = 0.0
        #     d_tp['AvgLossRet'] = np.round(
        #         trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"]["Return on Trade"].mean(), 4)
        #     if math.isnan(d_tp['AvgLossRet']):
        #         d_tp['AvgLossRet'] = 0.0
        #     if d_tp['AvgLossRet'] != 0:
        #         d_tp['WinByLossRet'] = np.round(abs(d_tp['AvgWinRet'] / d_tp['AvgLossRet']), 2)
        #     else:
        #         d_tp['WinByLossRet'] = 0
        #     if math.isnan(d_tp['WinByLossRet']):
        #         d_tp['WinByLossRet'] = 0.0
        #     if math.isinf(d_tp['WinByLossRet']):
        #         d_tp['WinByLossRet'] = 0.0
        # else:
        #     d_tp["TotalWins"] = 0
        #     d_tp["TotalLosses"] = 0
        #     d_tp['TotalTrades'] = 0
        #     d_tp['HitRatio'] = 0
        #     d_tp['AvgWinRet'] = 0
        #     d_tp['AvgLossRet'] = 0
        #     d_tp['WinByLossRet'] = 0
        # self.tradewise_performance = pd.Series(d_tp)

        return self.data[['Date', 'Close', 'signal', 'Return', 'S_Return', 'trade_num']]

    # @staticmethod
    # def kelly(p, b):
    #     """
    #     Static method: No object or class related arguments
    #     p: win prob, b: net odds received on wager, output(f*) = p - (1-p)/b
    #
    #     Spreadsheet example
    #         from sympy import symbols, solve, diff
    #         x = symbols('x')
    #         y = (1+3.3*x)**37 *(1-x)**63
    #         solve(diff(y, x), x)[1]
    #     Shortcut
    #         .37 - 0.63/3.3
    #     """
    #     return np.round(p - (1 - p) / b, 4)

    def plot_performance(self, allocation=1, interest_rate = 6):
        # intializing a variable for initial allocation
        # to be used to create equity curve
        self.signal_performance(allocation, interest_rate)

        # yearly performance
        # self.yearly_performance()

        # Plotting the Performance of the strategy
        plt.plot(self.data['Date'], self.data['Market_Return'], color='black', label='Market Returns')
        plt.plot(self.data['Date'], self.data['Strategy_Return'], color='blue', label='Strategy Returns')
        # plt.plot(self.data['Date'],self.data['fisher_rate'],color='red', label= 'Fisher Rate')
        plt.title('Strategy Backtest')
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()

        plt.plot(self.data['Date'], self.data['Portfolio Value'], color='blue')
        plt.title('Portfolio Value')
        plt.show()
        # return(self.data)

    # def yearly_performance(self):
    #     """
    #     Instance method
    #     Adds an instance attribute: yearly_df
    #     """
    #     _yearly_df = self.data.groupby(['yr', 'signal']).S_Return.sum().unstack()
    #     _yearly_df.rename(columns={-1.0: 'Sell', 1.0: 'Buy'}, inplace=True)
    #     _yearly_df['Return'] = _yearly_df.sum(1)
    #
    #     # yearly_df
    #     self.yearly_df = _yearly_df.style.bar(color=["#ffcccb", '#90ee90'], align='zero').format({
    #         'Sell': '{:,.2%}'.format, 'Buy': '{:,.2%}'.format, 'Return': '{:,.2%}'.format})

    # def update_metrics(self):
    #     """
    #     Called from the opt_matrix class method
    #     """
    #     d_field = {}
    #
    #     d_field['PortfolioValue'] = self.data['Portfolio Value']
    #     d_field['Sharpe'] = self.daywise_performance.SharpeRatio
    #     d_field['Sortino'] = self.daywise_performance.SortinoRatio
    #     d_field['CAGR'] = self.daywise_performance.CAGR
    #     d_field['MDD'] = self.daywise_performance.MaxDrawdown
    #     d_field['NHR'] = self.tradewise_performance.NormHitRatio
    #     #d_field['OTS'] = self.tradewise_performance.OptimalTradeSize
    #     d_field['AvgWinLoss'] = self.tradewise_performance.WinByLossRet
    #
    #     return d_field

    # @classmethod
    # def opt_matrix(cls, data, buy_fish, sell_fish, metrics, optimal_sol=True):
    #     """
    #
    #     """
    #     c_green = sns.light_palette("green", as_cmap=True)
    #     c_red = sns.light_palette("red", as_cmap=True)
    #
    #     d_mats = {m: [] for m in metrics}
    #
    #
    #     for lows in buy_fish:
    #         d_row = {m: [] for m in metrics}
    #         for highs in sell_fish:
    #             # if highs>=lows:
    #             obj = cls(data, zone_high=highs, zone_low=lows)  ## object being created from the class
    #             obj.generate_signals(charts=False)
    #             obj.signal_performance(10000, 6)
    #             d_field = obj.update_metrics()
    #             for m in metrics: d_row[m].append(d_field.get(m, np.nan))
    #             # else:
    #             #     for m in metrics: d_row[m].append(0)
    #         for m in metrics: d_mats[m].append(d_row[m])
    #
    #     d_df = {m: pd.DataFrame(d_mats[m], index=buy_fish, columns=sell_fish) for m in metrics}
    #
    #     def optimal(_df):
    #
    #         _df = _df.stack().rank()
    #         _df = (_df - _df.mean()) / _df.std()
    #         return _df.unstack()
    #
    #     if optimal_sol:
    #         # d_df['Metric'] = 0
    #         # if 'Sortino' in metrics: d_df['Metric'] += optimal(d_df['Sortino'])
    #         # if 'PVal' in metrics: d_df['Metric'] += optimal(d_df['PortfolioValue'])
    #         # if 'Sharpe' in metrics: d_df['Metric'] += 2 * optimal(d_df['Sharpe'])
    #         # if 'NHR' in metrics: d_df['Metric'] += optimal(d_df['NHR'])
    #         # if 'CAGR' in metrics: d_df['Metric'] += optimal(d_df['CAGR'])
    #         # if 'MDD' in metrics: d_df['Metric'] -= 2 * optimal(d_df['MDD'])
    #         # d1 = pd.DataFrame(d_df['Metric'])
    #         # val = np.amax(d1.to_numpy())
    #         # bf = d1.index[np.where(d1 == val)[0][0]]
    #         # sf = d1.columns[np.where(d1 == val)[1][0]]
    #
    #         #******
    #         d2 = pd.DataFrame(d_df[metrics[0]])
    #         val = np.amax(d2.to_numpy())
    #         bf = d2.index[np.where(d2 == val)[0][0]]
    #         sf = d2.columns[np.where(d2 == val)[1][0]]
    #         #******
    #         #return d_df
    #
    #         #print(f"Most optimal pair is Lower Bound:{bf}, Upper Bound:{sf}, with max {metrics[0]}:{val}")
    #         #metrics.insert(0, 'Signal')
    #
    #     # for m in metrics:
    #     #     display(HTML(d_df[m].style.background_gradient(axis=None, cmap=
    #     #     c_red if m == "MDD" else c_green).format(
    #     #         ("{:,.2}" if m in ["Sharpe", "Signal"] else "{:.2%}")).set_caption(m).render()))
    #
    #
    #     return (bf, sf, val)

class vol_mean_reverting_percentage_strategy:
    metrics = {'Sharpe': 'Sharpe Ratio',
               'CAGR': 'Compound Average Growth Rate',
               'MDD': 'Maximum Drawdown',
               'NHR': 'Normalized Hit Ratio',
               'OTS': 'Optimal trade size'}

    def __init__(self, data, band_buy, level_sell, level, lookback, start=None, end=None):

        self.bbh = band_buy
        self.bbl = -band_buy
        self.ls = level_sell
        self.level = level
        self.n = lookback
        self.data = data[["Date", "Close", f"dev_from_{level}_{lookback}"]]  # the dataframe
        self.data['yr'] = self.data['Date'].dt.year
        self.n_days = (self.data.Date.iloc[-1] - self.data.Date.iloc[0])
        self.start = start  # the beginning date of the sample period
        self.end = end  # the ending date of the sample period

    def generate_signals(self, charts=False):

        self.data["dev_lag"] = self.data[f"dev_from_{self.level}_{self.n}"].shift(1)
        self.data["lb"] = self.bbl
        self.data["ub"] = self.bbh
        self.data["ls"] = self.ls
        self.data.dropna()
        self.data.reset_index(inplace=True)
        self.data = self.data.drop(['index'], axis=1)

        ## creating signal
        ##Upper and lower bounds
        buy_mask = (self.data[f"dev_from_{self.level}_{self.n}"]>=self.data["lb"])&(self.data[f"dev_from_{self.level}_{self.n}"]<=self.data["ub"])
        sell_mask = self.data[f"dev_from_{self.level}_{self.n}"]>=self.data["ls"]

        #buy_mask = ((self.data["fisher"] >= self.data["lb"]) & (self.data["fisher"]>=self.data["ub"]))
        #sell_mask = ((self.data["fisher"] < self.data["lb"])|(self.data["fisher"]<self.data["ub"])&(self.data["fisher"]>=self.data["lb"]))

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

        # Closing positions at end of time period
        self.data["signal"][-1:] = 0

        buy_plot_mask = ((self.data.signal.shift(-1) == bval) & (self.data.signal == sval))
        sell_plot_mask = ((self.data.signal.shift(-1) == sval) & (self.data.signal == bval))

        #initialize with a buy position
        buy_plot_mask[0] = True

        mask = ((self.data.signal == 1) & (self.data.signal.shift(1)==0)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

        #self.data.to_csv("Test.csv")

        ## display chart
        if charts:
            plt.plot(self.data['Date'].to_numpy(), self.data['Close'].to_numpy(), color='black', label='Price')
            plt.plot(self.data.loc[buy_plot_mask, 'Date'].to_numpy(), self.data.loc[buy_plot_mask, 'Close'].to_numpy(), r'^', ms=15,
                     label="Entry Signal", color='green', markeredgecolor='k', markeredgewidth=1)
            plt.plot(self.data.loc[sell_plot_mask, 'Date'].to_numpy(), self.data.loc[sell_plot_mask, 'Close'].to_numpy(), r'^', ms=15,
                     label="Exit Signal", color='red', markeredgecolor='k', markeredgewidth=1)
            plt.title('Strategy Backtest')
            plt.legend(loc=0)
            d_color = {}
            d_color[1] = '#90ee90'  ## light green
            d_color[-1] = "#ffcccb"  ## light red
            d_color[0] = '#ffffff'

            j = 0
            for i in range(1, self.data.shape[0]):
                if np.isnan(self.data.signal[i - 1]):
                    j = i
                elif (self.data.signal[i - 1] == self.data.signal[i]) and (i < (self.data.shape[0] - 1)):
                    continue
                else:
                    plt.axvspan(self.data['Date'][j], self.data['Date'][i],
                                alpha=0.5, color=d_color[self.data.signal[i - 1]], label="interval")
                    j = i
            plt.show()

            plt.plot(self.data['Date'], self.data[f"dev_from_{self.level}_{self.n}"], color='black', label=f"dev_from_{self.level}_{self.n}")


            plt.plot(self.data['Date'], 0.2*self.data["signal"],color='purple', label=f"Signal")


            plt.plot(self.data['Date'], self.data['lb'], color='green', label='Lower Bound for Buy')
            plt.plot(self.data['Date'], self.data['ub'], color='red', label='Upper Bound for Buy')
            plt.plot(self.data['Date'], self.data['ls'], color='blue', label='Level for Sell')

            # plt.plot(self.data.loc[buy_plot_mask]['Date'], self.data.loc[buy_plot_mask][f"dev_from_{self.level}_{self.n}"],
            #          r'^', ms=15,
            #          label="Entry Signal", color='green', markeredgecolor='k', markeredgewidth=1)
            # plt.plot(self.data.loc[sell_plot_mask]['Date'], self.data.loc[sell_plot_mask][f"dev_from_{self.level}_{self.n}"],
            #          r'^', ms=15,
            #          label="Exit Signal", color='red', markeredgecolor='k', markeredgewidth=1)
            plt.title('Strategy Backtest')
            plt.legend(loc=0)
            d_color = {}
            d_color[1] = '#90ee90'  ## light green
            d_color[-1] = "#ffcccb"  ## light red
            d_color[0] = '#ffffff'

            j = 0
            for i in range(1, self.data.shape[0]):
                if np.isnan(self.data.signal[i - 1]):
                    j = i
                elif (self.data.signal[i - 1] == self.data.signal[i]) and (i < (self.data.shape[0] - 1)):
                    continue
                else:
                    plt.axvspan(self.data['Date'][j], self.data['Date'][i],
                                alpha=0.5, color=d_color[self.data.signal[i - 1]], label="interval")
                    j = i
            plt.show()

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
        self.data["S_Return"] = self.data["S_Return"] + (self.int/25200)*(1-self.data['signal'].shift(1))
        self.data.dropna(inplace=True)
        self.data['Market_Return'] = self.data['Return'].expanding().sum()
        self.data['Strategy_Return'] = self.data['S_Return'].expanding().sum()
        self.data['Portfolio Value'] = ((self.data['Strategy_Return'] + 1) * self.allocation)
        # self.data['Wins'] = np.where(self.data['S_Return'] > 0, 1, 0)
        # self.data['Losses'] = np.where(self.data['S_Return'] < 0, 1, 0)

        ## Daywise Performance
        d_perform = {}
        num_trades = self.data.iloc[-1]["trade_num"]
        sortino_of_trades = []
        for i in range(1, num_trades + 1):
            try:
                if self.data.loc[(self.data["trade_num"] == i) & (self.data["S_Return"] < 0), "S_Return"].std() != 0:
                    sortino_of_trades.append(
                        (self.data.loc[self.data["trade_num"] == i, "S_Return"].mean() - 0.06 / 252) / self.data.loc[
                            (self.data["trade_num"] == i) & (self.data["S_Return"] < 0), "S_Return"].std() * (
                                    252 ** .5))
                else:
                    sortino_of_trades.append(5)
            except:
                sortino_of_trades.append(0)
        if len(sortino_of_trades) > 0:
            d_perform['avg_sortino_of_trades'] = sum(sortino_of_trades) / len(sortino_of_trades)
        else:
            d_perform['avg_sortino_of_trades'] = 0
        # d_perform['TotalWins'] = self.data['Wins'].sum()
        # d_perform['TotalLosses'] = self.data['Losses'].sum()
        # d_perform['TotalTrades'] = d_perform['TotalWins'] + d_perform['TotalLosses']
        # if d_perform['TotalTrades']==0:
        #     d_perform['HitRatio'] = 0
        # else:
        #     d_perform['HitRatio'] = round(d_perform['TotalWins'] / d_perform['TotalTrades'], 2)
        # d_perform['SharpeRatio'] = (self.data["S_Return"].mean() -0.06/252)/ self.data["S_Return"].std() * (252 ** .5)
        # d_perform['StDev Annualized Downside Return'] = self.data.loc[self.data["S_Return"]<0, "S_Return"].std() * (252 ** .5)
        # #print(self.data["S_Return"])#.isnull().sum().sum())
        # if math.isnan(d_perform['StDev Annualized Downside Return']):
        #     d_perform['StDev Annualized Downside Return'] = 0.0
        # #print(d_perform['StDev Annualized Downside Return'])
        # if d_perform['StDev Annualized Downside Return'] != 0.0:
        #     d_perform['SortinoRatio'] = (self.data["S_Return"].mean()-0.06/252)*252/ d_perform['StDev Annualized Downside Return']
        # else:
        #     d_perform['SortinoRatio'] = 0
        # if len(self.data['Strategy_Return'])!=0:
        #     d_perform['CAGR'] = (1 + self.data['Strategy_Return']).iloc[-1] ** (365.25 / self.n_days.days) - 1
        # else:
        #     d_perform['CAGR'] = 0
        # d_perform['MaxDrawdown'] = (1.0 - self.data['Portfolio Value'] / self.data['Portfolio Value'].cummax()).max()
        self.daywise_performance = pd.Series(d_perform)
        #
        # ## Tradewise performance
        # ecdf = self.data[self.data["signal"] == 1]
        # trade_wise_results = []
        # if len(ecdf) > 0:
        #     for i in range(max(ecdf['trade_num'])):
        #         trade_num = i + 1
        #         entry = ecdf[ecdf["trade_num"] == trade_num].iloc[0]["Close"]
        #         exit = ecdf[ecdf["trade_num"] == trade_num].iloc[-1]["Close"]
        #         trade_wise_results.append({'Trade Number': trade_num, 'Entry': entry, 'Exit': exit})
        # trade_wise_results = pd.DataFrame(trade_wise_results)
        # d_tp = {}
        # if len(trade_wise_results) > 0:
        #     trade_wise_results["Win/Loss"] = np.where(trade_wise_results["Exit"] > trade_wise_results["Entry"], "Win",
        #                                               "Loss")
        #     trade_wise_results["Return on Trade"] = trade_wise_results["Exit"] / trade_wise_results["Entry"] - 1
        #     d_tp["TotalWins"] = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Win"])
        #     d_tp["TotalLosses"] = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"])
        #     d_tp['TotalTrades'] = d_tp["TotalWins"] + d_tp["TotalLosses"]
        #     if d_tp['TotalTrades'] == 0:
        #         d_tp['HitRatio'] = 0
        #     else:
        #         d_tp['HitRatio'] = round(d_tp['TotalWins'] / d_tp['TotalTrades'], 4)
        #     d_tp['AvgWinRet'] = np.round(
        #         trade_wise_results[trade_wise_results["Win/Loss"] == "Win"]["Return on Trade"].mean(), 4)
        #     if math.isnan(d_tp['AvgWinRet']):
        #         d_tp['AvgWinRet'] = 0.0
        #     d_tp['AvgLossRet'] = np.round(
        #         trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"]["Return on Trade"].mean(), 4)
        #     if math.isnan(d_tp['AvgLossRet']):
        #         d_tp['AvgLossRet'] = 0.0
        #     if d_tp['AvgLossRet'] != 0:
        #         d_tp['WinByLossRet'] = np.round(abs(d_tp['AvgWinRet'] / d_tp['AvgLossRet']), 2)
        #     else:
        #         d_tp['WinByLossRet'] = 0
        #     if math.isnan(d_tp['WinByLossRet']):
        #         d_tp['WinByLossRet'] = 0.0
        #     if math.isinf(d_tp['WinByLossRet']):
        #         d_tp['WinByLossRet'] = 0.0
        # else:
        #     d_tp["TotalWins"] = 0
        #     d_tp["TotalLosses"] = 0
        #     d_tp['TotalTrades'] = 0
        #     d_tp['HitRatio'] = 0
        #     d_tp['AvgWinRet'] = 0
        #     d_tp['AvgLossRet'] = 0
        #     d_tp['WinByLossRet'] = 0
        # self.tradewise_performance = pd.Series(d_tp)

        return self.data[['Date', 'Close', 'signal', 'Return', 'S_Return', 'trade_num']]

    # @staticmethod
    # def kelly(p, b):
    #     """
    #     Static method: No object or class related arguments
    #     p: win prob, b: net odds received on wager, output(f*) = p - (1-p)/b
    #
    #     Spreadsheet example
    #         from sympy import symbols, solve, diff
    #         x = symbols('x')
    #         y = (1+3.3*x)**37 *(1-x)**63
    #         solve(diff(y, x), x)[1]
    #     Shortcut
    #         .37 - 0.63/3.3
    #     """
    #     return np.round(p - (1 - p) / b, 4)

    def plot_performance(self, allocation=1, interest_rate = 6):
        # intializing a variable for initial allocation
        # to be used to create equity curve
        self.signal_performance(allocation, interest_rate)

        # yearly performance
        # self.yearly_performance()

        # Plotting the Performance of the strategy
        plt.plot(self.data['Date'], self.data['Market_Return'], color='black', label='Market Returns')
        plt.plot(self.data['Date'], self.data['Strategy_Return'], color='blue', label='Strategy Returns')
        # plt.plot(self.data['Date'],self.data['fisher_rate'],color='red', label= 'Fisher Rate')
        plt.title('Strategy Backtest')
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()

        plt.plot(self.data['Date'], self.data['Portfolio Value'], color='blue')
        plt.title('Portfolio Value')
        plt.show()
        # return(self.data)

    # def yearly_performance(self):
    #     """
    #     Instance method
    #     Adds an instance attribute: yearly_df
    #     """
    #     _yearly_df = self.data.groupby(['yr', 'signal']).S_Return.sum().unstack()
    #     _yearly_df.rename(columns={-1.0: 'Sell', 1.0: 'Buy'}, inplace=True)
    #     _yearly_df['Return'] = _yearly_df.sum(1)
    #
    #     # yearly_df
    #     self.yearly_df = _yearly_df.style.bar(color=["#ffcccb", '#90ee90'], align='zero').format({
    #         'Sell': '{:,.2%}'.format, 'Buy': '{:,.2%}'.format, 'Return': '{:,.2%}'.format})

    # def update_metrics(self):
    #     """
    #     Called from the opt_matrix class method
    #     """
    #     d_field = {}
    #
    #     d_field['PortfolioValue'] = self.data['Portfolio Value']
    #     d_field['Sharpe'] = self.daywise_performance.SharpeRatio
    #     d_field['Sortino'] = self.daywise_performance.SortinoRatio
    #     d_field['CAGR'] = self.daywise_performance.CAGR
    #     d_field['MDD'] = self.daywise_performance.MaxDrawdown
    #     d_field['NHR'] = self.tradewise_performance.NormHitRatio
    #     #d_field['OTS'] = self.tradewise_performance.OptimalTradeSize
    #     d_field['AvgWinLoss'] = self.tradewise_performance.WinByLossRet
    #
    #     return d_field

    # @classmethod
    # def opt_matrix(cls, data, buy_fish, sell_fish, metrics, optimal_sol=True):
    #     """
    #
    #     """
    #     c_green = sns.light_palette("green", as_cmap=True)
    #     c_red = sns.light_palette("red", as_cmap=True)
    #
    #     d_mats = {m: [] for m in metrics}
    #
    #
    #     for lows in buy_fish:
    #         d_row = {m: [] for m in metrics}
    #         for highs in sell_fish:
    #             # if highs>=lows:
    #             obj = cls(data, zone_high=highs, zone_low=lows)  ## object being created from the class
    #             obj.generate_signals(charts=False)
    #             obj.signal_performance(10000, 6)
    #             d_field = obj.update_metrics()
    #             for m in metrics: d_row[m].append(d_field.get(m, np.nan))
    #             # else:
    #             #     for m in metrics: d_row[m].append(0)
    #         for m in metrics: d_mats[m].append(d_row[m])
    #
    #     d_df = {m: pd.DataFrame(d_mats[m], index=buy_fish, columns=sell_fish) for m in metrics}
    #
    #     def optimal(_df):
    #
    #         _df = _df.stack().rank()
    #         _df = (_df - _df.mean()) / _df.std()
    #         return _df.unstack()
    #
    #     if optimal_sol:
    #         # d_df['Metric'] = 0
    #         # if 'Sortino' in metrics: d_df['Metric'] += optimal(d_df['Sortino'])
    #         # if 'PVal' in metrics: d_df['Metric'] += optimal(d_df['PortfolioValue'])
    #         # if 'Sharpe' in metrics: d_df['Metric'] += 2 * optimal(d_df['Sharpe'])
    #         # if 'NHR' in metrics: d_df['Metric'] += optimal(d_df['NHR'])
    #         # if 'CAGR' in metrics: d_df['Metric'] += optimal(d_df['CAGR'])
    #         # if 'MDD' in metrics: d_df['Metric'] -= 2 * optimal(d_df['MDD'])
    #         # d1 = pd.DataFrame(d_df['Metric'])
    #         # val = np.amax(d1.to_numpy())
    #         # bf = d1.index[np.where(d1 == val)[0][0]]
    #         # sf = d1.columns[np.where(d1 == val)[1][0]]
    #
    #         #******
    #         d2 = pd.DataFrame(d_df[metrics[0]])
    #         val = np.amax(d2.to_numpy())
    #         bf = d2.index[np.where(d2 == val)[0][0]]
    #         sf = d2.columns[np.where(d2 == val)[1][0]]
    #         #******
    #         #return d_df
    #
    #         #print(f"Most optimal pair is Lower Bound:{bf}, Upper Bound:{sf}, with max {metrics[0]}:{val}")
    #         #metrics.insert(0, 'Signal')
    #
    #     # for m in metrics:
    #     #     display(HTML(d_df[m].style.background_gradient(axis=None, cmap=
    #     #     c_red if m == "MDD" else c_green).format(
    #     #         ("{:,.2}" if m in ["Sharpe", "Signal"] else "{:.2%}")).set_caption(m).render()))
    #
    #
    #     return (bf, sf, val)

class vol_mom_strategy:
    metrics = {'Sharpe': 'Sharpe Ratio',
               'CAGR': 'Compound Average Growth Rate',
               'MDD': 'Maximum Drawdown',
               'NHR': 'Normalized Hit Ratio',
               'OTS': 'Optimal trade size'}

    def __init__(self, data, kb, xb, levelb, lookbackb, ks, xs, levels, lookbacks, start=None, end=None):

        self.kb = kb
        self.xb = xb
        self.ks = ks
        self.xs = xs
        self.levelb = levelb
        self.nb = lookbackb
        self.levels = levels
        self.ns = lookbacks
        self.data = data[["Date", "Close", f"{levelb}_{lookbackb}", f"{levels}_{lookbacks}"]]  # the dataframe
        self.data['yr'] = self.data['Date'].dt.year
        self.n_days = (self.data.Date.iloc[-1] - self.data.Date.iloc[0])
        self.start = start  # the beginning date of the sample period
        self.end = end  # the ending date of the sample period

    def generate_signals(self, charts=False):
        self.data = self.data.loc[:, ~self.data.columns.duplicated()]
        self.data.dropna()
        self.data.reset_index(inplace=True)
        self.data = self.data.drop(['index'], axis=1)

        ## creating signal
        ##Upper and lower bounds
        buy_cond1 = (self.data[f"{self.levelb}_{self.nb}"].shift(self.kb+1).fillna(method="bfill") > self.data["Close"].shift(self.kb+1).fillna(method="bfill"))
        buy_cond2 = self.data["Close"] > math.inf
        for i in range(len(self.data)):
            buy_cond2.iloc[i] = all(self.data[f"{self.levelb}_{self.nb}"].shift(self.kb - j).fillna(method="bfill").iloc[i] <
                                    self.data["Close"].shift(self.kb - j).fillna(method="bfill").iloc[i] for j in range(self.kb + 1))
        buy_mask = (buy_cond1) & (buy_cond2) & (abs(self.data[f"{self.levelb}_{self.nb}"]-self.data["Close"])/self.data["Close"]>self.xb)

        sell_cond1 = (self.data[f"{self.levels}_{self.ns}"].shift(self.ks + 1).fillna(method="bfill") < self.data["Close"].shift(self.ks + 1).fillna(method="bfill"))
        sell_cond2 = self.data["Close"] > math.inf
        for i in range(len(self.data)):
            sell_cond2.iloc[i] = all(self.data[f"{self.levels}_{self.ns}"].shift(self.ks - j).fillna(method="bfill").iloc[i] >self.data["Close"].shift(self.ks - j).fillna(method="bfill").iloc[i] for j in range(self.ks + 1))
        sell_mask = (sell_cond1) & (sell_cond2) & (abs(self.data[f"{self.levels}_{self.ns}"]-self.data["Close"])/self.data["Close"]>self.xs)

        #buy_mask = ((self.data["fisher"] >= self.data["lb"]) & (self.data["fisher"]>=self.data["ub"]))
        #sell_mask = ((self.data["fisher"] < self.data["lb"])|(self.data["fisher"]<self.data["ub"])&(self.data["fisher"]>=self.data["lb"]))

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

        # Closing positions at end of time period
        self.data["signal"][-1:] = 0

        buy_plot_mask = ((self.data.signal.shift(-1) == bval) & (self.data.signal == sval))
        sell_plot_mask = ((self.data.signal.shift(-1) == sval) & (self.data.signal == bval))

        #initialize with a buy position
        buy_plot_mask[0] = True

        mask = ((self.data.signal == 1) & (self.data.signal.shift(1)==0)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

        #self.data.to_csv("Test.csv")

        ## display chart
        if charts:
            plt.plot(self.data['Date'].to_numpy(), self.data['Close'].to_numpy(), color='black', label='Price')
            plt.plot(self.data['Date'].to_numpy(), self.data[f"{self.level}_{self.n}"].to_numpy(), color='orange', label=f"{self.level}_{self.n}")
            plt.plot(self.data.loc[buy_plot_mask, 'Date'].to_numpy(), self.data.loc[buy_plot_mask, 'Close'].to_numpy(), r'^', ms=15,
                     label="Entry Signal", color='green', markeredgecolor='k', markeredgewidth=1)
            plt.plot(self.data.loc[sell_plot_mask, 'Date'].to_numpy(), self.data.loc[sell_plot_mask, 'Close'].to_numpy(), r'^', ms=15,
                     label="Exit Signal", color='red', markeredgecolor='k', markeredgewidth=1)
            plt.title('Strategy Backtest')
            plt.legend(loc=0)
            d_color = {}
            d_color[1] = '#90ee90'  ## light green
            d_color[-1] = "#ffcccb"  ## light red
            d_color[0] = '#ffffff'

            j = 0
            for i in range(1, self.data.shape[0]):
                if np.isnan(self.data.signal[i - 1]):
                    j = i
                elif (self.data.signal[i - 1] == self.data.signal[i]) and (i < (self.data.shape[0] - 1)):
                    continue
                else:
                    plt.axvspan(self.data['Date'][j], self.data['Date'][i],
                                alpha=0.5, color=d_color[self.data.signal[i - 1]], label="interval")
                    j = i
            plt.show()

        return self.data#[["Date", "signal"]]

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
        self.data["S_Return"] = self.data["S_Return"] + (self.int/25200)*(1-self.data['signal'].shift(1))
        self.data.dropna(inplace=True)
        self.data['Market_Return'] = self.data['Return'].expanding().sum()
        self.data['Strategy_Return'] = self.data['S_Return'].expanding().sum()
        self.data['Portfolio Value'] = ((self.data['Strategy_Return'] + 1) * self.allocation)
        # self.data['Wins'] = np.where(self.data['S_Return'] > 0, 1, 0)
        # self.data['Losses'] = np.where(self.data['S_Return'] < 0, 1, 0)

        ## Daywise Performance
        d_perform = {}
        num_trades = self.data.iloc[-1]["trade_num"]

        sortino_of_trades =[]
        for i in range(1, num_trades + 1):
            try:
                if self.data.loc[(self.data["trade_num"] == i) & (self.data["signal"] == 1) & (self.data["S_Return"] < 0), "S_Return"].std() != 0:
                    sortino_of_trades.append((self.data.loc[self.data["trade_num"] == i, "S_Return"].mean() - 0.06 / 252) / self.data.loc[
                        (self.data["trade_num"] == i) & (self.data["signal"] == 1) & (self.data["S_Return"] < 0), "S_Return"].std() * (252 ** .5))
                else:
                    sortino_of_trades.append(5)
            except:
                sortino_of_trades.append(0)

        if len(sortino_of_trades)>0:
            d_perform['avg_sortino_of_trades'] = sum(sortino_of_trades) / len(sortino_of_trades)
        else:
            d_perform['avg_sortino_of_trades'] = 0

        # d_perform['TotalWins'] = self.data['Wins'].sum()
        # d_perform['TotalLosses'] = self.data['Losses'].sum()
        # d_perform['TotalTrades'] = d_perform['TotalWins'] + d_perform['TotalLosses']
        # if d_perform['TotalTrades']==0:
        #     d_perform['HitRatio'] = 0
        # else:
        #     d_perform['HitRatio'] = round(d_perform['TotalWins'] / d_perform['TotalTrades'], 2)
        # d_perform['SharpeRatio'] = (self.data["S_Return"].mean() -0.06/252)/ self.data["S_Return"].std() * (252 ** .5)
        # d_perform['StDev Annualized Downside Return'] = self.data.loc[self.data["S_Return"]<0, "S_Return"].std() * (252 ** .5)
        # #print(self.data["S_Return"])#.isnull().sum().sum())
        # if math.isnan(d_perform['StDev Annualized Downside Return']):
        #     d_perform['StDev Annualized Downside Return'] = 0.0
        # #print(d_perform['StDev Annualized Downside Return'])
        # if d_perform['StDev Annualized Downside Return'] != 0.0:
        #     d_perform['SortinoRatio'] = (self.data["S_Return"].mean()-0.06/252)*252/ d_perform['StDev Annualized Downside Return']
        # else:
        #     d_perform['SortinoRatio'] = 0
        # if len(self.data['Strategy_Return'])!=0:
        #     d_perform['CAGR'] = (1 + self.data['Strategy_Return']).iloc[-1] ** (365.25 / self.n_days.days) - 1
        # else:
        #     d_perform['CAGR'] = 0
        # d_perform['MaxDrawdown'] = (1.0 - self.data['Portfolio Value'] / self.data['Portfolio Value'].cummax()).max()
        self.daywise_performance = pd.Series(d_perform)
        #
        # ## Tradewise performance
        # ecdf = self.data[self.data["signal"] == 1]
        # trade_wise_results = []
        # if len(ecdf) > 0:
        #     for i in range(max(ecdf['trade_num'])):
        #         trade_num = i + 1
        #         entry = ecdf[ecdf["trade_num"] == trade_num].iloc[0]["Close"]
        #         exit = ecdf[ecdf["trade_num"] == trade_num].iloc[-1]["Close"]
        #         trade_wise_results.append({'Trade Number': trade_num, 'Entry': entry, 'Exit': exit})
        # trade_wise_results = pd.DataFrame(trade_wise_results)
        # d_tp = {}
        # if len(trade_wise_results) > 0:
        #     trade_wise_results["Win/Loss"] = np.where(trade_wise_results["Exit"] > trade_wise_results["Entry"], "Win",
        #                                               "Loss")
        #     trade_wise_results["Return on Trade"] = trade_wise_results["Exit"] / trade_wise_results["Entry"] - 1
        #     d_tp["TotalWins"] = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Win"])
        #     d_tp["TotalLosses"] = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"])
        #     d_tp['TotalTrades'] = d_tp["TotalWins"] + d_tp["TotalLosses"]
        #     if d_tp['TotalTrades'] == 0:
        #         d_tp['HitRatio'] = 0
        #     else:
        #         d_tp['HitRatio'] = round(d_tp['TotalWins'] / d_tp['TotalTrades'], 4)
        #     d_tp['AvgWinRet'] = np.round(
        #         trade_wise_results[trade_wise_results["Win/Loss"] == "Win"]["Return on Trade"].mean(), 4)
        #     if math.isnan(d_tp['AvgWinRet']):
        #         d_tp['AvgWinRet'] = 0.0
        #     d_tp['AvgLossRet'] = np.round(
        #         trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"]["Return on Trade"].mean(), 4)
        #     if math.isnan(d_tp['AvgLossRet']):
        #         d_tp['AvgLossRet'] = 0.0
        #     if d_tp['AvgLossRet'] != 0:
        #         d_tp['WinByLossRet'] = np.round(abs(d_tp['AvgWinRet'] / d_tp['AvgLossRet']), 2)
        #     else:
        #         d_tp['WinByLossRet'] = 0
        #     if math.isnan(d_tp['WinByLossRet']):
        #         d_tp['WinByLossRet'] = 0.0
        #     if math.isinf(d_tp['WinByLossRet']):
        #         d_tp['WinByLossRet'] = 0.0
        # else:
        #     d_tp["TotalWins"] = 0
        #     d_tp["TotalLosses"] = 0
        #     d_tp['TotalTrades'] = 0
        #     d_tp['HitRatio'] = 0
        #     d_tp['AvgWinRet'] = 0
        #     d_tp['AvgLossRet'] = 0
        #     d_tp['WinByLossRet'] = 0
        # self.tradewise_performance = pd.Series(d_tp)

        return self.data

    # @staticmethod
    # def kelly(p, b):
    #     """
    #     Static method: No object or class related arguments
    #     p: win prob, b: net odds received on wager, output(f*) = p - (1-p)/b
    #
    #     Spreadsheet example
    #         from sympy import symbols, solve, diff
    #         x = symbols('x')
    #         y = (1+3.3*x)**37 *(1-x)**63
    #         solve(diff(y, x), x)[1]
    #     Shortcut
    #         .37 - 0.63/3.3
    #     """
    #     return np.round(p - (1 - p) / b, 4)

    def plot_performance(self, allocation=1, interest_rate = 6):
        # intializing a variable for initial allocation
        # to be used to create equity curve
        self.signal_performance(allocation, interest_rate)

        # yearly performance
        #self.yearly_performance()

        # Plotting the Performance of the strategy
        plt.plot(self.data['Date'], self.data['Market_Return'], color='black', label='Market Returns')
        plt.plot(self.data['Date'], self.data['Strategy_Return'], color='blue', label='Strategy Returns')
        # plt.plot(self.data['Date'],self.data['fisher_rate'],color='red', label= 'Fisher Rate')
        plt.title('Strategy Backtest')
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()

        plt.plot(self.data['Date'], self.data['Portfolio Value'], color='blue')
        plt.title('Portfolio Value')
        plt.show()
        # return(self.data)

    # def yearly_performance(self):
    #     """
    #     Instance method
    #     Adds an instance attribute: yearly_df
    #     """
    #     _yearly_df = self.data.groupby(['yr', 'signal']).S_Return.sum().unstack()
    #     _yearly_df.rename(columns={-1.0: 'Sell', 1.0: 'Buy'}, inplace=True)
    #     _yearly_df['Return'] = _yearly_df.sum(1)
    #
    #     # yearly_df
    #     self.yearly_df = _yearly_df.style.bar(color=["#ffcccb", '#90ee90'], align='zero').format({
    #         'Sell': '{:,.2%}'.format, 'Buy': '{:,.2%}'.format, 'Return': '{:,.2%}'.format})

    # def update_metrics(self):
    #     """
    #     Called from the opt_matrix class method
    #     """
    #     d_field = {}
    #
    #     d_field['PortfolioValue'] = self.data['Portfolio Value']
    #     d_field['Sharpe'] = self.daywise_performance.SharpeRatio
    #     d_field['Sortino'] = self.daywise_performance.SortinoRatio
    #     d_field['CAGR'] = self.daywise_performance.CAGR
    #     d_field['MDD'] = self.daywise_performance.MaxDrawdown
    #     d_field['NHR'] = self.tradewise_performance.NormHitRatio
    #     #d_field['OTS'] = self.tradewise_performance.OptimalTradeSize
    #     d_field['AvgWinLoss'] = self.tradewise_performance.WinByLossRet
    #
    #     return d_field
    #
    # @classmethod
    # def opt_matrix(cls, data, buy_fish, sell_fish, metrics, optimal_sol=True):
    #     """
    #
    #     """
    #     c_green = sns.light_palette("green", as_cmap=True)
    #     c_red = sns.light_palette("red", as_cmap=True)
    #
    #     d_mats = {m: [] for m in metrics}
    #
    #
    #     for lows in buy_fish:
    #         d_row = {m: [] for m in metrics}
    #         for highs in sell_fish:
    #             # if highs>=lows:
    #             obj = cls(data, zone_high=highs, zone_low=lows)  ## object being created from the class
    #             obj.generate_signals(charts=False)
    #             obj.signal_performance(10000, 6)
    #             d_field = obj.update_metrics()
    #             for m in metrics: d_row[m].append(d_field.get(m, np.nan))
    #             # else:
    #             #     for m in metrics: d_row[m].append(0)
    #         for m in metrics: d_mats[m].append(d_row[m])
    #
    #     d_df = {m: pd.DataFrame(d_mats[m], index=buy_fish, columns=sell_fish) for m in metrics}
    #
    #     def optimal(_df):
    #
    #         _df = _df.stack().rank()
    #         _df = (_df - _df.mean()) / _df.std()
    #         return _df.unstack()
    #
    #     if optimal_sol:
    #         # d_df['Metric'] = 0
    #         # if 'Sortino' in metrics: d_df['Metric'] += optimal(d_df['Sortino'])
    #         # if 'PVal' in metrics: d_df['Metric'] += optimal(d_df['PortfolioValue'])
    #         # if 'Sharpe' in metrics: d_df['Metric'] += 2 * optimal(d_df['Sharpe'])
    #         # if 'NHR' in metrics: d_df['Metric'] += optimal(d_df['NHR'])
    #         # if 'CAGR' in metrics: d_df['Metric'] += optimal(d_df['CAGR'])
    #         # if 'MDD' in metrics: d_df['Metric'] -= 2 * optimal(d_df['MDD'])
    #         # d1 = pd.DataFrame(d_df['Metric'])
    #         # val = np.amax(d1.to_numpy())
    #         # bf = d1.index[np.where(d1 == val)[0][0]]
    #         # sf = d1.columns[np.where(d1 == val)[1][0]]
    #
    #         #******
    #         d2 = pd.DataFrame(d_df[metrics[0]])
    #         val = np.amax(d2.to_numpy())
    #         bf = d2.index[np.where(d2 == val)[0][0]]
    #         sf = d2.columns[np.where(d2 == val)[1][0]]
    #         #******
    #         #return d_df
    #
    #         #print(f"Most optimal pair is Lower Bound:{bf}, Upper Bound:{sf}, with max {metrics[0]}:{val}")
    #         #metrics.insert(0, 'Signal')
    #
    #     # for m in metrics:
    #     #     display(HTML(d_df[m].style.background_gradient(axis=None, cmap=
    #     #     c_red if m == "MDD" else c_green).format(
    #     #         ("{:,.2}" if m in ["Sharpe", "Signal"] else "{:.2%}")).set_caption(m).render()))
    #
    #
    #     return (bf, sf, val)

def return_volume_features_minute_hourly(temp_og, temp_og1):
    temp = []
    for i in tqdm(range(len(temp_og))):
        res = {}
        res["Datetime"] = temp_og.iloc[i]["Datetime"]
        for n in [2, 5, 10, 21, 42, 63, 126, 252, 504]:
            try:
                if i >= n - 1:
                    volumes, high_prices, low_prices = return_dataframe_minute(temp_og1, temp_og, i, n)
                    res[f"CalcHow_{n}"] = "Minute"
                    price_levels = calc_distribution(high_prices, low_prices, volumes)
                    res[f"PriceLevels_{n}"] = price_levels
                else:
                    res[f"PriceLevels_{n}"] = {}
                    res[f"CalcHow_{n}"] = "DataNotAvailable"
            except:
                res[f"PriceLevels_{n}"] = {}
                res[f"CalcHow_{n}"] = "DataNotAvailable"
        temp.append(res)
    return temp

def return_volume_features(temp_og, temp_og1):
    temp = []
    for i in tqdm(range(len(temp_og))):
        res = {}
        res["Date"] = temp_og.iloc[i]["Date"]
        for n in [2, 5, 10, 21, 42, 63, 126, 252, 504]:
            try:
                if n <= 21:
                    if i >= n - 1:
                        volumes, high_prices, low_prices = return_dataframe_hourly(temp_og1, temp_og, i, n)
                        res[f"CalcHow_{n}"] = "Hourly"
                        if len(volumes) == 0:
                            res[f"CalcHow_{n}"] = "Daily"
                            volumes, high_prices, low_prices = return_dataframe_daily(temp_og, i, n)
                        price_levels = calc_distribution(high_prices, low_prices, volumes)
                        res[f"PriceLevels_{n}"] = price_levels

                    else:
                        res[f"PriceLevels_{n}"] = {}
                        res[f"CalcHow_{n}"] = "DataNotAvailable"
                else:
                    if i >= n - 1:
                        volumes, high_prices, low_prices = return_dataframe_daily(temp_og, i, n)
                        price_levels = calc_distribution(high_prices, low_prices, volumes)
                        res[f"PriceLevels_{n}"] = price_levels
                        res[f"CalcHow_{n}"] = "Daily"
                    else:
                        res[f"PriceLevels_{n}"] = {}
                        res[f"CalcHow_{n}"] = "DataNotAvailable"
            except NameError:
                res[f"PriceLevels_{n}"] = {"poc": np.nan,"profile_high": np.nan,"profile_low": np.nan,"vah": np.nan,"val": np.nan}
                res[f"CalcHow_{n}"] = "Insufficient Data"
        temp.append(res)
    return temp

def calc_distribution(highs,lows,volumes, plot_hist=False):
    x = []
    y = []
    for i in range(len(volumes)):
        prices = np.round(np.linspace(lows[i], highs[i], num=10),2)
        for j in range(10):
            x.append(prices[j])
            y.append(volumes[i]/10)
    prices = np.linspace(min(x), max(x), num=25)
    p = [0]*(len(prices)-1)
    v = [0]*(len(prices)-1)
    for j in range(len(prices)-1):
        p[j] = (prices[j] + prices[j+1])/2
    for i in range(len(x)):
        for j in range(len(prices)-1):
            if (x[i]>prices[j]-0.001) & (x[i]<=prices[j+1]):
                v[j] = v[j] + (y[i])

    poc = p[v.index(max(v))]
    profile_high = max(highs)
    profile_low = min(lows)
    target_volume = 0.7*sum(v)
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
        plt.bar(p,v)
        plt.plot(p, v)
        plt.axvline(x=poc, color="orange")
        plt.axvline(x=profile_high, color="darkgreen")
        plt.axvline(x=profile_low, color="maroon")
        plt.axvline(x=vah, color="lime")
        plt.axvline(x=val, color="red")
        plt.legend(["Point of Control", "Profile High", "Profile Low", "Value Area High", "Value Area Low"])
    return {"poc": poc,"profile_high": profile_high,"profile_low": profile_low,"vah": vah,"val": val}

def return_dataframe_minute(temp_og1, temp_og, i, n):
    volumes = list(temp_og1[(temp_og1["Datetime"] > pd.to_datetime(
            datetime.datetime(temp_og["Datetime"].iloc[i - n].year, temp_og["Datetime"].iloc[i - n].month,
                              temp_og["Datetime"].iloc[i - n].day, temp_og["Datetime"].iloc[i - n].hour, temp_og["Datetime"].iloc[i - n].minute))) &
                                (temp_og1["Datetime"] <= pd.to_datetime(
                                    datetime.datetime(temp_og["Datetime"].iloc[i].year, temp_og["Datetime"].iloc[i].month,
                                                      temp_og["Datetime"].iloc[i].day, temp_og["Datetime"].iloc[i].hour, temp_og["Datetime"].iloc[i].minute)))]["Volume"])
    high_prices = list(temp_og1[(temp_og1["Datetime"] > pd.to_datetime(
            datetime.datetime(temp_og["Datetime"].iloc[i - n].year, temp_og["Datetime"].iloc[i - n].month,
                              temp_og["Datetime"].iloc[i - n].day, temp_og["Datetime"].iloc[i - n].hour, temp_og["Datetime"].iloc[i - n].minute))) &
                                (temp_og1["Datetime"] <= pd.to_datetime(
                                    datetime.datetime(temp_og["Datetime"].iloc[i].year, temp_og["Datetime"].iloc[i].month,
                                                      temp_og["Datetime"].iloc[i].day, temp_og["Datetime"].iloc[i].hour, temp_og["Datetime"].iloc[i].minute)))]["High"])
    low_prices = list(temp_og1[(temp_og1["Datetime"] > pd.to_datetime(
            datetime.datetime(temp_og["Datetime"].iloc[i - n].year, temp_og["Datetime"].iloc[i - n].month,
                              temp_og["Datetime"].iloc[i - n].day, temp_og["Datetime"].iloc[i - n].hour, temp_og["Datetime"].iloc[i - n].minute))) &
                                (temp_og1["Datetime"] <= pd.to_datetime(
                                    datetime.datetime(temp_og["Datetime"].iloc[i].year, temp_og["Datetime"].iloc[i].month,
                                                      temp_og["Datetime"].iloc[i].day, temp_og["Datetime"].iloc[i].hour, temp_og["Datetime"].iloc[i].minute)))]["Low"])

    return volumes, high_prices, low_prices

def return_dataframe_daily(temp_og, i, n):
    volumes = list(temp_og.iloc[i-(n-1):i+1]["Volume"])
    high_prices = list(temp_og.iloc[i-(n-1):i+1]["High"])
    low_prices = list(temp_og.iloc[i-(n-1):i+1]["Low"])
    return volumes, high_prices, low_prices

def return_dataframe_hourly(temp_og1, temp_og, i, n):
    volumes = list(temp_og1[(temp_og1["Date"].apply(lambda x: x.replace(hour=0)) >= pd.to_datetime(
        datetime.datetime(temp_og["Date"].iloc[i - n].year, temp_og["Date"].iloc[i - n].month,
                          temp_og["Date"].iloc[i - n].day)).tz_localize(pytz.utc).tz_convert('Asia/Kolkata').replace(
        hour=0, minute=0)) &
                            (temp_og1["Date"].apply(lambda x: x.replace(hour=0)) <= pd.to_datetime(
                                datetime.datetime(temp_og["Date"].iloc[i].year, temp_og["Date"].iloc[i].month,
                                                  temp_og["Date"].iloc[i].day)).tz_localize(pytz.utc).tz_convert(
                                'Asia/Kolkata').replace(hour=0, minute=0))]["Volume"])
    high_prices = list(temp_og1[(temp_og1["Date"].apply(lambda x: x.replace(hour=0)) >= pd.to_datetime(
        datetime.datetime(temp_og["Date"].iloc[i - n].year, temp_og["Date"].iloc[i - n].month,
                          temp_og["Date"].iloc[i - n].day)).tz_localize(pytz.utc).tz_convert('Asia/Kolkata').replace(
        hour=0, minute=0)) &
                                (temp_og1["Date"].apply(lambda x: x.replace(hour=0)) <= pd.to_datetime(
                                    datetime.datetime(temp_og["Date"].iloc[i].year,
                                                      temp_og["Date"].iloc[i].month,
                                                      temp_og["Date"].iloc[i].day)).tz_localize(
                                    pytz.utc).tz_convert('Asia/Kolkata').replace(hour=0, minute=0))]["High"])
    low_prices = list(temp_og1[(temp_og1["Date"].apply(lambda x: x.replace(hour=0)) >= pd.to_datetime(
        datetime.datetime(temp_og["Date"].iloc[i - n].year, temp_og["Date"].iloc[i - n].month,
                          temp_og["Date"].iloc[i - n].day)).tz_localize(pytz.utc).tz_convert('Asia/Kolkata').replace(
        hour=0, minute=0)) &
                               (temp_og1["Date"].apply(lambda x: x.replace(hour=0)) <= pd.to_datetime(
                                   datetime.datetime(temp_og["Date"].iloc[i].year,
                                                     temp_og["Date"].iloc[i].month,
                                                     temp_og["Date"].iloc[i].day)).tz_localize(pytz.utc).tz_convert(
                                   'Asia/Kolkata').replace(hour=0, minute=0))]["Low"])

    return volumes, high_prices, low_prices

def rolling_percentile_freturns(df, lookforward_freturn, lookback_percentile):
    df["Percentile"] = np.nan
    for i in range(len(df)):
        try:
            df.loc[df.index[i], "Percentile"] = stats.percentileofscore(df.iloc[i - lookforward_freturn - lookback_percentile + 1:i - lookforward_freturn]["FReturn"], df.iloc[i]["FReturn"])
        except:
            continue
    return df

def rolling_percentile(inp):
    df = inp[0]
    lookback_percentile = inp[1]
    columns = inp[2]
    for column in tqdm(columns):
        df[f"{column}_percentile_over_{lookback_percentile}"] = np.nan
        for i in range(len(df)):
            try:
                df.loc[df.index[i], f"{column}_percentile_over_{lookback_percentile}"] = stats.percentileofscore(
                    df.iloc[i - lookback_percentile + 1:i][column], df.iloc[i][column])/100
            except:
                continue
    return df[[f"{column}_percentile_over_{lookback_percentile}" for column in columns]]

def rolling_percentile_parallelized(df_inp, lookback_percentiles, columns):
    df = df_inp.copy()
    inputs = []
    for lookback_percentile in lookback_percentiles:
        inputs.append([df, lookback_percentile, columns])
    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results = pool.map(rolling_percentile, inputs)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    for result in results:
        df = pd.concat([df, result], axis=1)
    return df

def prepare_volume_features_for_ML(ohlcv, vol_feat, return_lookforwards, percentile_lookbacks, train_val_split_date, val_test_split_date, freturn, split=True):
    if ("Date" in ohlcv.columns) & ("Date" in vol_feat.columns):
        date_col = "Date"
    if ("Datetime" in ohlcv.columns) & ("Datetime" in vol_feat.columns):
        date_col = "Datetime"

    vol_temp = pd.DataFrame()
    vol_feat = pd.concat([vol_feat.set_index(date_col), ohlcv.set_index(date_col)], axis=1, join="inner").reset_index()
    vol_temp[date_col] = vol_feat[date_col]



    # #adding vol_levels, pct deviation wrt price level, delta of price wrt level, velocity of price wrt level, acceleration of price wrt level
    # for n in [2, 5, 10, 21, 42, 63, 126, 252, 504]:
    #     for i in range(len(vol_temp)):
    #         try:
    #             vol_temp.loc[i, f"poc_{n}"] = vol_feat.iloc[i][f"PriceLevels_{n}"]['poc']
    #             vol_temp.loc[i, f"vah_{n}"] = vol_feat.iloc[i][f"PriceLevels_{n}"]['vah']
    #             vol_temp.loc[i, f"val_{n}"] = vol_feat.iloc[i][f"PriceLevels_{n}"]['val']
    #             vol_temp.loc[i, f"dev_from_poc_{n}"] = vol_feat.iloc[i]["Close"] / vol_feat.iloc[i][f"PriceLevels_{n}"]['poc'] - 1
    #             vol_temp.loc[i, f"dev_from_vah_{n}"] = vol_feat.iloc[i]["Close"] / vol_feat.iloc[i][f"PriceLevels_{n}"]['vah'] - 1
    #             vol_temp.loc[i, f"dev_from_val_{n}"] = vol_feat.iloc[i]["Close"] / vol_feat.iloc[i][f"PriceLevels_{n}"]['val'] - 1
    #             vol_temp.loc[i, f"delta_from_poc_{n}"] = vol_feat.iloc[i]["Close"] - vol_feat.iloc[i][f"PriceLevels_{n}"]['poc']
    #             vol_temp.loc[i, f"delta_from_vah_{n}"] = vol_feat.iloc[i]["Close"] - vol_feat.iloc[i][f"PriceLevels_{n}"]['vah']
    #             vol_temp.loc[i, f"delta_from_val_{n}"] = vol_feat.iloc[i]["Close"] - vol_feat.iloc[i][f"PriceLevels_{n}"]['val']
    #         except:
    #             try:
    #                 vol_temp.loc[i, f"poc_{n}"] = vol_feat.iloc[i-1][f"PriceLevels_{n}"]['poc']
    #                 vol_temp.loc[i, f"vah_{n}"] = vol_feat.iloc[i-1][f"PriceLevels_{n}"]['vah']
    #                 vol_temp.loc[i, f"val_{n}"] = vol_feat.iloc[i-1][f"PriceLevels_{n}"]['val']
    #                 vol_temp.loc[i, f"dev_from_poc_{n}"] = vol_feat.iloc[i]["Close"] / vol_feat.iloc[i-1][f"PriceLevels_{n}"]['poc'] - 1
    #                 vol_temp.loc[i, f"dev_from_vah_{n}"] = vol_feat.iloc[i]["Close"] / vol_feat.iloc[i-1][f"PriceLevels_{n}"]['vah'] - 1
    #                 vol_temp.loc[i, f"dev_from_val_{n}"] = vol_feat.iloc[i]["Close"] / vol_feat.iloc[i-1][f"PriceLevels_{n}"]['val'] - 1
    #                 vol_temp.loc[i, f"delta_from_poc_{n}"] = vol_feat.iloc[i]["Close"] - vol_feat.iloc[i-1][f"PriceLevels_{n}"]['poc']
    #                 vol_temp.loc[i, f"delta_from_vah_{n}"] = vol_feat.iloc[i]["Close"] - vol_feat.iloc[i-1][f"PriceLevels_{n}"]['vah']
    #                 vol_temp.loc[i, f"delta_from_val_{n}"] = vol_feat.iloc[i]["Close"] - vol_feat.iloc[i-1][f"PriceLevels_{n}"]['val']
    #             except:
    #                 continue
    #     vol_temp[f"velocity_poc_{n}"] = vol_temp[f"poc_{n}"].diff()
    #     vol_temp[f"velocity_val_{n}"] = vol_temp[f"val_{n}"].diff()
    #     vol_temp[f"velocity_vah_{n}"] = vol_temp[f"vah_{n}"].diff()
    #     vol_temp[f"acceleration_poc_{n}"] = vol_temp[f"velocity_poc_{n}"].diff()
    #     vol_temp[f"acceleration_val_{n}"] = vol_temp[f"velocity_val_{n}"].diff()
    #     vol_temp[f"acceleration_vah_{n}"] = vol_temp[f"velocity_vah_{n}"].diff()
    #     vol_temp[f"relvelocity_of_Price_from_poc_{n}"] = vol_temp[f"delta_from_poc_{n}"].diff()
    #     vol_temp[f"relvelocity_of_Price_from_val_{n}"] = vol_temp[f"delta_from_val_{n}"].diff()
    #     vol_temp[f"relvelocity_of_Price_from_vah_{n}"] = vol_temp[f"delta_from_vah_{n}"].diff()
    #     vol_temp[f"relacceleration_of_Price_from_poc_{n}"] = vol_temp[f"relvelocity_of_Price_from_poc_{n}"].diff()
    #     vol_temp[f"relacceleration_of_Price_from_vah_{n}"] = vol_temp[f"relvelocity_of_Price_from_vah_{n}"].diff()
    #     vol_temp[f"relacceleration_of_Price_from_val_{n}"] = vol_temp[f"relvelocity_of_Price_from_val_{n}"].diff()
    # #add relative deltas, velocities and accelerations between levels
    # for n1 in [2, 5, 10, 21, 42, 63, 126, 252, 504]:
    #     for l1 in ["vah", "val", "poc"]:
    #         for n2 in [2, 5, 10, 21, 42, 63, 126, 252, 504]:
    #             for l2 in ["vah", "val", "poc"]:
    #                 if (n1==n2)&(l1==l2):
    #                     continue
    #                 if (f"delta_{l2}{n2}_{l1}{n1}") in list(vol_temp.columns):
    #                     continue
    #                 vol_temp[f"delta_{l1}{n1}_{l2}{n2}"] = (vol_temp[f"{l1}_{n1}"] - vol_temp[f"{l2}_{n2}"])
    #                 vol_temp[f"velocity_{l1}{n1}_{l2}{n2}"] = vol_temp[f"delta_{l1}{n1}_{l2}{n2}"].diff()
    #                 vol_temp[f"acceleration_{l1}{n1}_{l2}{n2}"] = vol_temp[f"velocity_{l1}{n1}_{l2}{n2}"].diff()


    vol_temp = pd.concat([vol_temp.set_index(date_col), ohlcv.set_index(date_col)], axis=1, join="outer").reset_index()
    #Adding metrics
    for return_lookforward in tqdm(return_lookforwards):
        for percentile_lookback in percentile_lookbacks:
            if freturn == "max":
                for i in range(1, return_lookforward+1):
                    vol_temp[f"FReturn{i}"] = vol_temp["Close"].shift(-i) / vol_temp["Close"] - 1
                vol_temp[f"FReturn"] = vol_temp[[f"FReturn{i}" for i in range(1, return_lookforward+1)]].max(axis=1)
                vol_temp = vol_temp.drop(columns=[f"FReturn{i}" for i in range(1, return_lookforward+1)])
            if freturn == "min":
                for i in range(1, return_lookforward+1):
                    vol_temp[f"FReturn{i}"] = vol_temp["Close"].shift(-i) / vol_temp["Close"] - 1
                vol_temp[f"FReturn"] = vol_temp[[f"FReturn{i}" for i in range(1, return_lookforward+1)]].min(axis=1)
                vol_temp = vol_temp.drop(columns=[f"FReturn{i}" for i in range(1, return_lookforward+1)])
            if freturn == "vanilla":
                vol_temp[f"FReturn"] = vol_temp["Close"].shift(-return_lookforward) / vol_temp["Close"] - 1
            vol_temp[f"{return_lookforward}FReturn_percentile_over_{percentile_lookback}"] = rolling_percentile_freturns(vol_temp, return_lookforward, percentile_lookback)["Percentile"]/100
            vol_temp.drop(columns="FReturn", inplace=True)
            vol_temp.drop(columns="Percentile", inplace=True)

    vol_temp.dropna(inplace=True)
    vol_temp.reset_index(drop=True, inplace=True)
    if split==True:
        vol_temp_train = vol_temp[vol_temp[date_col]<train_val_split_date]
        vol_feats_train = vol_temp_train[[col for col in list(vol_temp_train.columns) if "FReturn_percentile_over" not in col]]
        vol_metrics_train = vol_temp_train[[col for col in list(vol_temp_train.columns) if "FReturn_percentile_over" in col]]
        vol_temp_val = vol_temp[(vol_temp[date_col] >train_val_split_date)&(vol_temp[date_col] < val_test_split_date)]
        vol_feats_val = vol_temp_val[[col for col in list(vol_temp_val.columns) if "FReturn_percentile_over" not in col]]
        vol_metrics_val = vol_temp_val[[col for col in list(vol_temp_val.columns) if "FReturn_percentile_over" in col]]
        vol_temp_test = vol_temp[(vol_temp[date_col] > val_test_split_date)]
        vol_feats_test = vol_temp_test[[col for col in list(vol_temp_test.columns) if "FReturn_percentile_over" not in col]]
        vol_metrics_test = vol_temp_test[[col for col in list(vol_temp_test.columns) if "FReturn_percentile_over" in col]]
        return vol_feats_train, vol_metrics_train, vol_feats_val, vol_metrics_val, vol_feats_test, vol_metrics_test
    else:
        return vol_temp


def prepare_volume_features_from_cache(temp_og, add_absolute_vals_of_levels=False):
    with open(f'NSEI_VolumeLevels.pkl', 'rb') as file:
        temp = pickle.load(file)
    vol_temp = pd.DataFrame()
    temp = pd.concat([temp.set_index("Date"), temp_og.set_index("Date")], axis=1, join="inner").reset_index()
    vol_temp["Date"] = temp['Date']
    for n in [2, 5, 10, 21, 42, 63, 126, 252, 504]:
        vol_temp[f"dev_from_poc_{n}"] = np.nan
        for i in range(len(vol_temp)):
            try:
                if add_absolute_vals_of_levels:
                    vol_temp.loc[i, f"poc_{n}"] = temp.iloc[i][f"PriceLevels_{n}"]['poc']
                    vol_temp.loc[i, f"vah_{n}"] = temp.iloc[i][f"PriceLevels_{n}"]['vah']
                    vol_temp.loc[i, f"val_{n}"] = temp.iloc[i][f"PriceLevels_{n}"]['val']
                vol_temp.loc[i, f"dev_from_poc_{n}"] = temp.iloc[i]["Close"]/temp.iloc[i][f"PriceLevels_{n}"]['poc']-1
                vol_temp.loc[i, f"dev_from_vah_{n}"] = temp.iloc[i]["Close"] / temp.iloc[i][f"PriceLevels_{n}"]['vah'] - 1
                vol_temp.loc[i, f"dev_from_val_{n}"] = temp.iloc[i]["Close"] / temp.iloc[i][f"PriceLevels_{n}"]['val'] - 1
            except:
                continue
    vol_temp = pd.concat([vol_temp.set_index("Date"), temp_og.set_index("Date")], axis=1, join="inner").reset_index()
    return vol_temp

def valid_dates(dates_all):
    dates = []
    i = 0
    while True:
        dates.append(dates_all[i])
        #if dates_all[i] > pd.to_datetime(date.today()):
        if dates_all[i] > pd.to_datetime(date.today().replace(month=11, day=12)):
            break
        i = i + 1
    return dates

def return_nmi_bw_pctdev_of_close_from_nday_volumelevels_and_freturn30(df):
    def alpha(*args):
        return df[f"dev_from_{level}_{n}"]
    def prior(params):
        return 1

    res = []
    for level in ['poc', 'vah', 'val']:
        for n in [2, 5, 10, 21, 42, 63, 126, 252, 504]:
            mc = MCMC(alpha_fn=alpha, alpha_fn_params_0=[0],
                                  target=df["FReturn30"], num_iters=1,
                                  prior=prior, optimize_fn=None, lower_limit=0, upper_limit=1)
            rs = mc.optimize()
            nmi = mc.analyse_results(rs, top_n=1)[1][0]
            res.append({"Level": level, "lookback": n, "NMI": nmi})
    res = pd.DataFrame(res)
    res = res.sort_values("NMI", axis=0, ascending=False).reset_index(drop=True)
    return res

def return_outliers(df_train, how):
    df_cluster = df_train.copy()

    if how == "UpsideMaxAll":
        for i in range(1,31):
            df_cluster[f"FReturn{i}"] = df_cluster["Close"].pct_change(i)
        df_cluster[f"FReturn"] = df_cluster[[f"FReturn{i}" for i in range(1,31)]].max( axis=1)
        df_cluster = df_cluster.drop(columns=[f"FReturn{i}" for i in range(1,30)])
        df_cluster.dropna(inplace=True)

    if how == "UpsideMax":
        for i in range(1,31):
            df_cluster[f"FReturn{i}"] = df_cluster["Close"].pct_change(i)
        df_cluster[f"FReturn"] = df_cluster[[f"FReturn{i}" for i in range(1,31)]].max( axis=1)
        df_cluster = df_cluster.drop(columns=[f"FReturn{i}" for i in range(1,30)])
        df_cluster.dropna(inplace=True)
        df_cluster = rolling_percentile_freturns(df_cluster, 300)
        df_cluster = df_cluster[df_cluster.Percentile>85]
        df_cluster = df_cluster.drop(columns=["Percentile"])

    if how == "Upside":
        df_cluster[f"FReturn"] = df_cluster["Close"].pct_change(30)
        df_cluster.dropna(inplace=True)
        df_cluster = rolling_percentile_freturns(df_cluster, 300)
        df_cluster = df_cluster[df_cluster.Percentile > 85]
        df_cluster = df_cluster.drop(columns=["Percentile"])

    if how == "DownsideMinAll":
        for i in range(1,31):
            df_cluster[f"FReturn{i}"] = df_cluster["Close"].pct_change(i)
        df_cluster[f"FReturn"] = df_cluster[[f"FReturn{i}" for i in range(1,31)]].min( axis=1)
        df_cluster = df_cluster.drop(columns=[f"FReturn{i}" for i in range(1,30)])
        df_cluster.dropna(inplace=True)

    if how == "DownsideMin":
        for i in range(1,31):
            df_cluster[f"FReturn{i}"] = df_cluster["Close"].pct_change(i)
        df_cluster[f"FReturn"] = df_cluster[[f"FReturn{i}" for i in range(1,31)]].min( axis=1)
        df_cluster = df_cluster.drop(columns=[f"FReturn{i}" for i in range(1,30)])
        df_cluster.dropna(inplace=True)
        df_cluster = rolling_percentile_freturns(df_cluster, 300)
        df_cluster = df_cluster[df_cluster.Percentile < 15]
        df_cluster = df_cluster.drop(columns=["Percentile"])

    if how == "Downside":
        df_cluster[f"FReturn"] = df_cluster["Close"].pct_change(30)
        df_cluster.dropna(inplace=True)
        df_cluster = rolling_percentile_freturns(df_cluster, 300)
        df_cluster = df_cluster[df_cluster.Percentile < 15]
        df_cluster = df_cluster.drop(columns=["Percentile"])

    # df_cluster = df_cluster.drop(columns=["BinaryOutcome", "Open", "High", "Low", "Close", 'Max_CB_20', 'Min_CB_20',
    #        'Max_CB_30', 'Min_CB_30', 'Max_CB_60', 'Min_CB_60', 'Max_FMACB_20', 'Min_FMACB_20', 'Max_FMACB_30',
    #        'Min_FMACB_30', 'Max_FMACB_60', 'Min_FMACB_60', 'Max_SMACB_20', 'Min_SMACB_20', 'Max_SMACB_30',
    #        'Min_SMACB_30', 'Max_SMACB_60', 'Min_SMACB_60',
    #        'ROC_CB_20', 'ROC_CB_30', 'ROC_CB_60', 'ROC_FMACB_20',
    #        'ROC_FMACB_30', 'ROC_FMACB_60', 'Convexity_FMACB',
    #        'ROC_SMACB_20', 'ROC_SMACB_30', 'ROC_SMACB_60'])

    return df_cluster

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

def get_data(ticker):

    temp_og = ek.get_timeseries(ticker, start_date='2007-01-01', end_date=str(date.today() + timedelta(1)))
    temp_og.reset_index(inplace=True)
    temp_og.rename(columns={"HIGH": "High", "CLOSE": "Close", "LOW": "Low", "OPEN": "Open", "VOLUME": "Volume"},
                   inplace=True)
    temp_og.drop(['COUNT'], axis=1, inplace=True)

    temp_og1 = ek.get_timeseries(ticker, start_date=str(date.today() - timedelta(400)),
                                 end_date=str(date.today() + timedelta(1)), interval='hour')
    temp_og1.index = temp_og1.index.tz_localize(pytz.utc).tz_convert('Asia/Kolkata')
    temp_og1.reset_index(inplace=True)
    temp_og1.rename(columns={"HIGH": "High", "CLOSE": "Close", "LOW": "Low", "OPEN": "Open", "VOLUME": "Volume"},
                    inplace=True)
    temp_og1.drop(['COUNT'], axis=1, inplace=True)
    temp_og1["Volume"] = temp_og1["Volume"].astype(float)
    temp_og1.dropna(inplace=True)

    return temp_og, temp_og1

def get_daily_data(ticker):
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
    temp_og.reset_index(drop=True, inplace=True)
    temp_og.drop(columns="API", inplace=True)
    temp_og["Date"] = pd.to_datetime(temp_og["Date"])
    temp_og["High"] = temp_og["High"].astype(float)
    temp_og["Low"] = temp_og["Low"].astype(float)
    temp_og["Close"] = temp_og["Close"].astype(float)
    temp_og["Open"] = temp_og["Open"].astype(float)

    return temp_og


def run_RFregressor_volume_features(vol_feats_train, vol_metrics_train, vol_feats_val, vol_metrics_val, vol_feats_test, vol_metrics_test, n_estimators, max_depth, min_samples_split, max_features, min_samples_leaf, m, n,
                                    criterion="squared_error", plot=False):
    rf_clf = RandomForestRegressor(criterion=criterion, n_estimators=n_estimators,
                                   max_depth=max_depth, min_samples_split=min_samples_split,
                                   max_features=max_features, min_samples_leaf=min_samples_leaf,
                                   n_jobs=-1, random_state=RANDOM_STATE)
    # print("fitting...")
    rf_clf.fit(vol_feats_train.to_numpy(),
               vol_metrics_train[
                   f"{m}MaxFReturn_percentile_over_{n}"].to_numpy())

    y_pred = rf_clf.predict(vol_feats_train.to_numpy())
    score = metrics.mean_absolute_error(
        vol_metrics_train[f"{m}MaxFReturn_percentile_over_{n}"].to_numpy(), y_pred)
    print(f"internal score for train: {score:.4f}")

    train = pd.DataFrame()
    train["Actual"] = vol_metrics_train[f"{m}MaxFReturn_percentile_over_{n}"].to_numpy()
    train["Predicted"] = y_pred

    if plot == True:
        train.plot()
        plt.title("Performance on Train Dataset")
        plt.show()

    y_pred = rf_clf.predict(vol_feats_val.to_numpy())
    score = metrics.mean_absolute_error(
        vol_metrics_val[f"{m}MaxFReturn_percentile_over_{n}"].to_numpy(), y_pred)
    print(f"internal score for validation: {score:.4f}")

    val = pd.DataFrame()
    val["Actual"] = vol_metrics_val[f"{m}MaxFReturn_percentile_over_{n}"].to_numpy()
    val["Predicted"] = y_pred

    if plot == True:
        val.plot()
        plt.title("Performance on Validation Dataset")
        plt.show()

    y_pred = rf_clf.predict(vol_feats_test.to_numpy())
    score = metrics.mean_absolute_error(
        vol_metrics_test[f"{m}MaxFReturn_percentile_over_{n}"].to_numpy(), y_pred)
    print(f"internal score for test: {score:.4f}")

    test = pd.DataFrame()
    test["Actual"] = vol_metrics_test[f"{m}MaxFReturn_percentile_over_{n}"].to_numpy()
    test["Predicted"] = y_pred

    if plot == True:
        test.plot()
        plt.title("Performance on Test Dataset")
        plt.show()

    feat_imp = []
    # Print the name and gini importance of each feature
    for feature in zip(vol_feats_train.columns, rf_clf.feature_importances_):
        feat_imp.append({"Feature": feature[0], "Feature importance": feature[1]})

    feat_imp = pd.DataFrame(feat_imp)
    feat_imp.sort_values(by="Feature importance", ascending=False, inplace=True)
    return feat_imp, train, val, test


def hyperparametric_tuning_rf_regressor_volume_features(vol_feats_train, vol_metrics_train, vol_feats_val, vol_metrics_val):
    def run_RFregressor_volume_features_MCMC(*params):

        if 1 * round(params[0] / 1) == 0:
            criterion = "squared_error"
        if 1 * round(params[0] / 1) == 1:
            criterion = "absolute_error"
        if 1 * round(params[0] / 1) == 2:
            criterion = "poisson"

        rf_clf = RandomForestRegressor(criterion=criterion, n_estimators=50 * round(params[1] / 50),
                                       max_depth=1 * round(params[2] / 1), min_samples_split=params[3],
                                       max_features=params[4], min_samples_leaf=params[5],
                                       n_jobs=-1, random_state=RANDOM_STATE)
        print("fitting...")

        rf_clf.fit(vol_feats_train.to_numpy(),
                   vol_metrics_train[
                       f"{5 * round(params[6] / 5)}MaxFReturn_percentile_over_{25 * round(params[7] / 25)}"].to_numpy())
        y_pred = rf_clf.predict(vol_feats_val.to_numpy())
        return np.vstack((vol_metrics_val[
                              f"{5 * round(params[6] / 5)}MaxFReturn_percentile_over_{25 * round(params[7] / 25)}"].to_numpy(),
                          y_pred))


    def prior(params):
        # if (params[0] < 0) | (params[0] > 2):
        #     return 0
        if (params[0] < 0) | (params[0] > 1):
            return 0
        if (params[1] < 50) | (params[1] > 1500):
            return 0
        if (params[2] < 2) | (params[2] > 20):
            return 0
        if (params[3] < 0.1) | (params[3] > 1.0):
            return 0
        if (params[4] < 0.1) | (params[4] > 1.0):
            return 0
        if (params[5] < 0.1) | (params[5] > 0.5):
            return 0
        if (params[6] < 5) | (params[6] > 50):
            return 0
        if (params[7] < 150) | (params[7] > 400):
            return 0

        # if (params[6] < 5) | (params[6] > 7):
        #     return 0
        # if (params[7] < 389) | (params[7] > 400):
        #     return 0

        return 1

    def metric(x, y, bins):
        score = metrics.mean_absolute_error(x[0, :], x[1, :])
        print(f"internal score: {score:.4f}")
        return -score

    guess = [0.35, 1313, 15, 0.2123173256732749, 0.7936366021300792, 0.10021994442738029, 32, 225]
    mc = MCMC(alpha_fn=run_RFregressor_volume_features_MCMC, alpha_fn_params_0=guess, num_iters=100, prior=prior,
              optimize_fn=metric, target=0, lower_limit=0, upper_limit=1500)
    rs = mc.optimize()

    top_n = 100
    res = [{"Criterion": 1 * round(mc.analyse_results(rs, top_n=top_n)[0][i][0] / 1),
            "n_estimators": 50 * round(mc.analyse_results(rs, top_n=top_n)[0][i][1] / 50),
            "max_depth": 1 * round(mc.analyse_results(rs, top_n=top_n)[0][i][2] / 1),
            "min_samples_split": mc.analyse_results(rs, top_n=top_n)[0][i][3],
            "max_features": mc.analyse_results(rs, top_n=top_n)[0][i][4],
            "min_samples_leaf": mc.analyse_results(rs, top_n=top_n)[0][i][5],
            "m": 5 * round(mc.analyse_results(rs, top_n=top_n)[0][i][6] / 5),
            "n": 25 * round(mc.analyse_results(rs, top_n=top_n)[0][i][7] / 25),
            "metric": mc.analyse_results(rs, top_n=top_n)[1][i]
            } \
           for i in range(top_n)]


    res = pd.DataFrame(res)
    res["metric"] = res["metric"] * (-1)
    res.sort_values(by="metric", ascending=True, inplace=True)
    return res

def hyperparametric_tuning_SVregressor_volume_features(vol_feats_train, vol_metrics_train, vol_feats_val, vol_metrics_val):
    def run_SVregressor_volume_features_MCMC(*params):

        regressor = SVR(kernel="rbf", gamma = params[0] , C = params[1], epsilon= params[2])

        print("fitting...")

        regressor.fit(vol_feats_train.to_numpy(),
                   vol_metrics_train[
                       f"{5 * round(params[3] / 5)}MaxFReturn_percentile_over_{25 * round(params[4] / 25)}"].to_numpy())
        y_pred = regressor.predict(vol_feats_val.to_numpy())
        return np.vstack((vol_metrics_val[
                              f"{5 * round(params[3] / 5)}MaxFReturn_percentile_over_{25 * round(params[4] / 25)}"].to_numpy(),
                          y_pred))

    def prior(params):

        if (params[0] < 0.0001) | (params[0] > 1):
            return 0
        if (params[1] < 0.1) | (params[1] > 1000):
            return 0
        if (params[2] < 0) | (params[2] > 1):
            return 0
        if (params[3] < 5) | (params[3] > 50):
            return 0
        if (params[4] < 150) | (params[4] > 400):
            return 0

        return 1

    def metric(x, y, bins):

        score = metrics.mean_absolute_error(x[0, :], x[1, :])
        print(f"internal score: {score:.4f}")
        return -score

    guess = [0.5, 1, 0.5, 32, 225]
    iters = 500

    mc = MCMC(alpha_fn=run_SVregressor_volume_features_MCMC, alpha_fn_params_0=guess, num_iters=iters, prior=prior,
              optimize_fn=metric, target=0, lower_limit=0, upper_limit=1000)
    rs = mc.optimize()

    top_n = iters

    res = [{
            "gamma": mc.analyse_results(rs, top_n=top_n)[0][i][0],
            "C": mc.analyse_results(rs, top_n=top_n)[0][i][1],
            "epsilon": mc.analyse_results(rs, top_n=top_n)[0][i][2],
            "m": 5 * round(mc.analyse_results(rs, top_n=top_n)[0][i][3] / 5),
            "n": 25 * round(mc.analyse_results(rs, top_n=top_n)[0][i][4] / 25),
            "metric": mc.analyse_results(rs, top_n=top_n)[1][i]
            } \
           for i in range(top_n)]


    res = pd.DataFrame(res)
    res["metric"] = res["metric"] * (-1)
    res.sort_values(by="metric", ascending=True, inplace=True)
    return res

def run_SVregressor_volume_features(vol_feats_train, vol_metrics_train, vol_feats_val, vol_metrics_val, vol_feats_test, vol_metrics_test, dates_train, dates_val, dates_test, train_unscaled, val_unscaled, test_unscaled, gamma, C, epsilon, m, n, lb, ub, plot=False):
    regressor = SVR(kernel="rbf", gamma = gamma , C = C, epsilon= epsilon)
    print("fitting...")
    regressor.fit(vol_feats_train[[col for col in vol_feats_train.columns if f"percentile_over_{n}" in col]].to_numpy(),
               vol_metrics_train[
                   f"{m}FReturn_percentile_over_{n}"].to_numpy())

    y_pred = regressor.predict(vol_feats_train[[col for col in vol_feats_train.columns if f"percentile_over_{n}" in col]].to_numpy())
    # avg_sortino, train = custom_score_avg_sortino_per_trade(vol_feats_train, stats, y_pred, m,n, lb,ub)
    outperf, train = custom_score_mean_of_rolling_outperformance(train_unscaled, dates_train, y_pred,lb,ub)
    score = metrics.mean_absolute_error(
        vol_metrics_train[f"{m}FReturn_percentile_over_{n}"].to_numpy(), y_pred)
    print(f"internal score for train: {score:.4f}")
    # print(f"internal avg_sortino for train: {avg_sortino:.4f}")
    print(f"Mean of rolling outperformance for train: {outperf:.4f}")

    train["Actual"] =  vol_metrics_train[[f"{m}FReturn_percentile_over_{n}"]]
    train["Predicted"] = y_pred

    if plot == True:
        train[["Actual", "Predicted"]].plot()
        plt.title("Performance on Train Dataset")
        plt.show()

    y_pred = regressor.predict(vol_feats_val[[col for col in vol_feats_val.columns if f"percentile_over_{n}" in col]].to_numpy())
    # avg_sortino, val = custom_score_avg_sortino_per_trade(vol_feats_val, stats, y_pred, m, n, lb, ub)
    # score = metrics.mean_absolute_error(
    #     vol_metrics_val[f"{m}MaxFReturn_percentile_over_{n}"].to_numpy(), y_pred)
    # print(f"internal avg_sortino for validation: {avg_sortino:.4f}")
    outperf, val = custom_score_mean_of_rolling_outperformance(val_unscaled, dates_val, y_pred, lb,ub)
    print(f"Mean of rolling outperformance for Val: {outperf:.4f}")

    val["Actual"] = vol_metrics_val[[f"{m}FReturn_percentile_over_{n}"]]
    val["Predicted"] = y_pred

    if plot == True:
        val[["Actual", "Predicted"]].plot()
        plt.title("Performance on Validation Dataset")
        plt.show()

    y_pred = regressor.predict(vol_feats_test[[col for col in vol_feats_test.columns if f"percentile_over_{n}" in col]].to_numpy())
    # score = metrics.mean_absolute_error(
    #     vol_metrics_test[f"{m}MaxFReturn_percentile_over_{n}"].to_numpy(), y_pred)
    # print(f"internal score for test: {score:.4f}")
    # avg_sortino, test = custom_score_avg_sortino_per_trade(vol_feats_test, stats, y_pred, m, n, lb, ub)
    # print(f"internal avg_sortino for test: {avg_sortino:.4f}")

    outperf, test = custom_score_mean_of_rolling_outperformance(test_unscaled, dates_test, y_pred, lb, ub)
    print(f"Mean of rolling outperformance for test: {outperf:.4f}")

    test["Actual"] = vol_metrics_test[[f"{m}FReturn_percentile_over_{n}"]]
    test["Predicted"] =  y_pred

    if plot == True:
        test[["Actual", "Predicted"]].plot()
        plt.title("Performance on Test Dataset")
        plt.show()

    return train, val, test


def performance_evaluation_report_multiclass(
    model,
    X_test_,
    y_test_,
    show_plot=False,
    labels=None,
    show_pr_curve=False,
    custom_threshold=None,
    average=None,
):
    """
    Function for creating a performance report of a classification model.

    Parameters
    ----------
    model : scikit-learn estimator
        A fitted estimator for classification problems.
    X_test_ : pd.DataFrame
        DataFrame with features matching y_test_
    y_test_ : array/pd.Series
        Target of a classification problem.
    show_plot : bool
        Flag whether to show the plot
    labels : list
        List with the class names.
    show_pr_curve : bool
        Flag whether to also show the PR-curve. For this to take effect,
        show_plot must be True.

    Return
    ------
    stats : pd.Series
        A series with the most important evaluation metrics
    """

    def plot_multiclass_roc(y_test_, y_pred, classes, ax):
        # structures
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # calculate dummies once
        y_test__dummies = pd.get_dummies(y_test_, drop_first=False).values
        for i, label in zip(range(len(classes)), classes):
            fpr[i], tpr[i], _ = roc_curve(y_test__dummies[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # roc for each class
        ax.plot([0, 1], [0, 1], "r--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        for i, label in zip(range(len(classes)), classes):
            ax.plot(
                fpr[i],
                tpr[i],
                label="ROC curve (area = %0.2f) for label %i" % (roc_auc[i], label),
            )
        ax.legend(loc="best")
        ax.set_title("ROC-AUC")
        # sns.despine()
        return

    def plot_multiclass_precision_recall_curve(y_test_, y_pred, classes, ax):
        # structures
        fpr = dict()
        tpr = dict()
        # pr_auc = dict()

        # precision recall curve
        precision = dict()
        recall = dict()

        # calculate dummies once
        y_test__dummies = pd.get_dummies(y_test_, drop_first=False).values
        for i, label in zip(range(len(classes)), classes):
            precision[i], recall[i], _ = precision_recall_curve(
                y_test__dummies[:, i], y_pred[:, i]
            )
            ax.plot(recall[i], precision[i], lw=2, label=f"class {label}")

        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.legend(loc="best")
        ax.set_title("precision vs. recall curve")
        return

    if custom_threshold is None:  # default is 50%
        y_pred = model.predict(X_test_)
    else:
        # TODO UPDATE FOR THE MULTICLASS CASE
        y_pred = (model.predict_proba(X_test_)[:, 1] > 0.5).astype(int)
        y_pred = np.where(y_pred == 0, -1, 1)

    y_pred_prob = model.predict_proba(X_test_)  # [:, 1]

    conf_mat = metrics.confusion_matrix(y_test_, y_pred)
    # REF:
    # https://stackoverflow.com/questions/50666091/true-positive-rate-and-false-positive-rate-tpr-fpr-for-multi-class-data-in-py
    fp = conf_mat.sum(axis=0) - np.diag(conf_mat)
    fn = conf_mat.sum(axis=1) - np.diag(conf_mat)
    tp = np.diag(conf_mat)
    tn = conf_mat.sum() - (fp + fn + tp)
    fp = fp.astype(float)
    fn = fn.astype(float)
    tp = tp.astype(float)
    tn = tn.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    tpr = tp / (tp + fn)
    # Specificity or true negative rate
    tnr = tn / (tn + fp)
    # Precision or positive predictive value
    ppv = tp / (tp + fp)
    # Negative predictive value
    npv = tn / (tn + fn)
    # Fall out or false positive rate
    fpr = fp / (fp + tn)
    # False negative rate
    fnr = fn / (tp + fn)
    # False discovery rate
    fdr = fp / (tp + fp)
    # Overall accuracy
    acc = (tp + tn) / (tp + fp + fn + tn)

    precision = (metrics.precision_score(y_test_, y_pred, average=average),)
    recall = (metrics.recall_score(y_test_, y_pred, average=average),)

    if show_plot:

        if labels is None:
            labels = ["Negative", "Positive"]

        N_SUBPLOTS = 3 if show_pr_curve else 2
        N_SUBPLOT_ROWS = 1 if show_pr_curve else 1
        PLOT_WIDTH = 17 if show_pr_curve else 12
        PLOT_HEIGHT = 10 if show_pr_curve else 6

        fig = plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT), tight_layout=True)
        gs = gridspec.GridSpec(N_SUBPLOT_ROWS, N_SUBPLOTS)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        # ax3 = fig.add_subplot(gs[1, 1])

        fig.suptitle("Performance Evaluation", fontsize=16, y=1.05)

        total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
        normed_conf_mat = conf_mat.astype("float") / total_samples

        text_array = np.empty_like(conf_mat, dtype="object")
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                norm_val = normed_conf_mat[i, j]
                int_val = conf_mat[i, j]
                text_array[i, j] = f"({norm_val:.1%})\n{int_val}"

        g = sns.heatmap(
            conf_mat,
            annot=text_array,
            fmt="s",
            linewidths=0.5,
            cmap="Blues",
            square=True,
            cbar=False,
            ax=ax0,
            annot_kws={"ha": "center", "va": "center"},
        )

        ax0.set(
            xlabel="Predicted label", ylabel="Actual label", title="Confusion Matrix"
        )
        ax0.xaxis.set_ticklabels(labels)
        ax0.yaxis.set_ticklabels(labels)

        _ = plot_multiclass_roc(y_test_, y_pred_prob, labels, ax1)
        ax1.plot(
            fp / (fp + tn), tp / (tp + fn), "ro", markersize=8, label="Decision Point"
        )

        if show_pr_curve:
            _ = plot_multiclass_precision_recall_curve(
                y_test_, y_pred_prob, labels, ax2
            )

    stats = {
        "accuracy": np.round(acc, 4),
        "precision": np.round(ppv, 4),
        "recall": np.round(tpr, 4),
        "mcc": round(metrics.matthews_corrcoef(y_test_, y_pred), 4),
        "specificity": np.round(tnr, 4),
        "f1_score": np.round(metrics.f1_score(y_test_, y_pred, average=average), 4),
        "cohens_kappa": round(metrics.cohen_kappa_score(y_test_, y_pred), 4),
        # "roc_auc": round(roc_auc, 4),
        # "pr_auc": round(pr_auc, 4),
    }

    return stats


def hyperparametric_tuning_rf_classifier_volume_features(vol_feats_train, vol_metrics_train, vol_feats_val, vol_metrics_val):

    def binary_outcome(df):
        df[df > 0] = 1
        df[df < 0] = -1
        df[df == 0] = 0
        return df

    def run_RFclassifier_volume_features_MCMC(*params):

        if 1 * round(params[0] / 1) == 0:
            criterion = "entropy"
        if 1 * round(params[0] / 1) == 1:
            criterion = "gini"

        rf_clf = RandomForestClassifier(criterion=criterion, n_estimators=50 * round(params[1] / 50),class_weight="balanced_subsample",
                                       max_depth=1 * round(params[2] / 1), min_samples_split=params[3],
                                       max_features=params[4], min_samples_leaf=params[5],
                                       n_jobs=-1, random_state=RANDOM_STATE)
        print("fitting...")

        rf_clf.fit(vol_feats_train.to_numpy(),
                   binary_outcome(vol_metrics_train[
                       f"{5 * round(params[6] / 5)}MaxFReturn_percentile_over_{25 * round(params[7] / 25)}"]).to_numpy())

        y_pred = rf_clf.predict(vol_feats_val.to_numpy())
        return np.vstack((binary_outcome(vol_metrics_val[
                              f"{5 * round(params[6] / 5)}MaxFReturn_percentile_over_{25 * round(params[7] / 25)}"].to_numpy()),
                          y_pred))

    def prior(params):
        if (params[0] < 0) | (params[0] > 1):
            return 0
        if (params[1] < 50) | (params[1] > 1500):
            return 0
        if (params[2] < 2) | (params[2] > 20):
            return 0
        if (params[3] < 0.1) | (params[3] > 1.0):
            return 0
        if (params[4] < 0.1) | (params[4] > 1.0):
            return 0
        if (params[5] < 0.1) | (params[5] > 0.5):
            return 0
        if (params[6] < 5) | (params[6] > 50):
            return 0
        if (params[7] < 150) | (params[7] > 400):
            return 0

        # if (params[6] < 5) | (params[6] > 7):
        #     return 0
        # if (params[7] < 389) | (params[7] > 400):
        #     return 0

        return 1

    def metric(x, y, bins):
        # print(x)
        score = metrics.matthews_corrcoef(x[0, :], x[1, :])
        print(f"internal score: {score:.4f}")
        return score

    guess = [0.35, 1313, 15, 0.2123173256732749, 0.7936366021300792, 0.10021994442738029, 32, 225]
    mc = MCMC(alpha_fn=run_RFclassifier_volume_features_MCMC, alpha_fn_params_0=guess, num_iters=100, prior=prior,
              optimize_fn=metric, target=0, lower_limit=0, upper_limit=1500)
    rs = mc.optimize()

    top_n = 100
    res = [{"Criterion": 1 * round(mc.analyse_results(rs, top_n=top_n)[0][i][0] / 1),
            "n_estimators": 50 * round(mc.analyse_results(rs, top_n=top_n)[0][i][1] / 50),
            "max_depth": 1 * round(mc.analyse_results(rs, top_n=top_n)[0][i][2] / 1),
            "min_samples_split": mc.analyse_results(rs, top_n=top_n)[0][i][3],
            "max_features": mc.analyse_results(rs, top_n=top_n)[0][i][4],
            "min_samples_leaf": mc.analyse_results(rs, top_n=top_n)[0][i][5],
            "m": 5 * round(mc.analyse_results(rs, top_n=top_n)[0][i][6] / 5),
            "n": 25 * round(mc.analyse_results(rs, top_n=top_n)[0][i][7] / 25),
            "metric": mc.analyse_results(rs, top_n=top_n)[1][i]
            } \
           for i in range(top_n)]

    res = pd.DataFrame(res)
    res["metric"] = res["metric"] * (-1)
    res.sort_values(by="metric", ascending=True, inplace=True)
    return res

def custom_score_mean_of_rolling_outperformance(vol_feats, dates, y_pred, lb, ub):
    backtest = vol_feats.copy()
    backtest["Predicted_Percentile"] = pd.DataFrame(y_pred)

    backtest["lb"] = lb
    backtest["ub"] = ub
    buy_mask = (backtest["Predicted_Percentile"] >= backtest["ub"])
    sell_mask = (backtest["Predicted_Percentile"] <= backtest["lb"])
    backtest["Datetime"] = dates
    bval = +1
    sval = 0

    backtest['signal'] = np.nan
    backtest.loc[buy_mask, 'signal'] = bval
    backtest.loc[sell_mask, 'signal'] = sval
    # initialize with long
    # backtest["signal"][0] = 1
    backtest.signal = backtest.signal.fillna(method="ffill")
    # Closing positions at end of time period
    backtest["signal"][-1:] = 0
    mask = ((backtest.signal == 1) & (backtest.signal.shift(1) == 0)) & (backtest.signal.notnull())
    backtest['trade_num'] = np.where(mask, 1, 0).cumsum()

    allocation = 10000
    interest_rate = 6

    backtest["signal"] = backtest["signal"].fillna(0)
    # creating returns and portfolio value series
    backtest['Return'] = np.log(backtest['Close'] / backtest['Close'].shift(1))
    backtest['Return'] = backtest['Return'].fillna(0)
    backtest['S_Return'] = backtest['signal'].shift(1) * backtest['Return']
    backtest["S_Return"] = backtest["S_Return"] + (interest_rate / (25200 * 24)) * (1 - backtest['signal'].shift(1))
    backtest['S_Return'] = backtest['S_Return'].fillna(0)

    # backtest.dropna(inplace=True)
    backtest['Market_Return'] = backtest['Return'].expanding().sum()
    backtest['Strategy_Return'] = backtest['S_Return'].expanding().sum()
    backtest['Portfolio Value'] = ((backtest['Strategy_Return'] + 1) * allocation)

    if len(backtest) == 0:
        score = 0
    else:
        r_window = int(0.1 * len(backtest))
        backtest["Datetime"] = pd.to_datetime(backtest["Datetime"])
        ecdf1 = backtest['S_Return'].to_frame().set_index(backtest["Datetime"])
        ecdf2 = backtest['Return'].to_frame().set_index(backtest["Datetime"])
        ecdf1['Portfolio'] = 1 + ecdf1['S_Return'].expanding().sum()
        ecdf1['Portfolio_hist'] = ecdf1['Portfolio'].to_frame().shift(r_window, freq='H')
        ecdf2['Portfolio'] = 1 + ecdf2['Return'].expanding().sum()
        ecdf2['Portfolio_hist'] = ecdf2['Portfolio'].to_frame().shift(r_window, freq='H')
        ecdf1['Portfolio_hist'] = ecdf1['Portfolio_hist'].fillna(method="ffill")
        ecdf1['RCAGR_Strategy_Series'] = ecdf1['Portfolio'] / ecdf1['Portfolio_hist'] - 1
        ecdf2['Portfolio_hist'] = ecdf2['Portfolio_hist'].fillna(method="ffill")
        ecdf2['RCAGR_Market_Series'] = ecdf2['Portfolio'] / ecdf2['Portfolio_hist'] - 1
        RCAGR_Strategy = ecdf1['RCAGR_Strategy_Series'].mean()
        RCAGR_Market = ecdf2['RCAGR_Market_Series'].mean()
        ROutperformance = RCAGR_Strategy - RCAGR_Market
        score = ROutperformance

        if math.isnan(score):
            score = 0

    return score, backtest

def custom_score_avg_sortino_per_trade(vol_feats, stats, y_pred, m, n, lb, ub):

    backtest = unscale(vol_feats.copy(), stats)
    backtest["Predicted_Percentile"] = unscale(
        pd.DataFrame(y_pred, columns=[f"{m}MaxFReturn_percentile_over_{n}"]), stats).values.reshape(
        (y_pred.shape[0],))
    backtest["lb"] = lb
    backtest["ub"] = ub
    buy_mask = (backtest["Predicted_Percentile"] >= backtest["ub"])
    sell_mask = (backtest["Predicted_Percentile"] <= backtest["lb"])

    bval = +1
    sval = 0

    backtest['signal'] = np.nan
    backtest.loc[buy_mask, 'signal'] = bval
    backtest.loc[sell_mask, 'signal'] = sval
    # initialize with long
    # backtest["signal"][0] = 1
    backtest.signal = backtest.signal.fillna(method="ffill")
    # Closing positions at end of time period
    backtest["signal"][-1:] = 0
    mask = ((backtest.signal == 1) & (backtest.signal.shift(1) == 0)) & (backtest.signal.notnull())
    backtest['trade_num'] = np.where(mask, 1, 0).cumsum()

    allocation = 10000
    interest_rate = 6

    # creating returns and portfolio value series
    backtest['Return'] = np.log(backtest['Close'] / backtest['Close'].shift(1))
    backtest['S_Return'] = backtest['signal'].shift(1) * backtest['Return']
    backtest['S_Return'] = backtest['S_Return'].fillna(0)
    backtest["S_Return"] = backtest["S_Return"] + (interest_rate / (25200*24)) * (1 - backtest['signal'].shift(1))

    # backtest.dropna(inplace=True)
    backtest['Market_Return'] = backtest['Return'].expanding().sum()
    backtest['Strategy_Return'] = backtest['S_Return'].expanding().sum()
    backtest['Portfolio Value'] = ((backtest['Strategy_Return'] + 1) * allocation)

    if len(backtest) == 0:
        score =  0
    else:
        num_trades = int(backtest.iloc[-1]["trade_num"])
        sortino_of_trades = []
        for i in range(1, num_trades + 1):
            try:
                if backtest.loc[(backtest["trade_num"] == i) & (backtest["S_Return"] < 0), "S_Return"].std() != 0:
                    sortino = (backtest.loc[backtest["trade_num"] == i, "S_Return"].mean() - 0.06 / (252*24)) / backtest.loc[(backtest["trade_num"] == i) & (backtest["S_Return"] < 0), "S_Return"].std() * ((252*24) ** .5)
                    if math.isnan(sortino):
                        sortino=5
                    sortino_of_trades.append(sortino)
                else:
                    sortino_of_trades.append(5)
            except:
                sortino_of_trades.append(0)
        if len(sortino_of_trades) > 0:
            avg_sortino_of_trades = sum(sortino_of_trades) / len(sortino_of_trades)
        else:
            avg_sortino_of_trades = 0

        score = avg_sortino_of_trades
    return score, backtest

def plot_equity_curve(df):
    df['Benchmark'] = ((df['Market_Return'] + 1) * df["Portfolio Value"].iloc[0])
    df[["Portfolio Value", "Benchmark"]].plot()