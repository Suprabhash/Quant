import os

Midcap_all_tickers = ['ESCO', 'MINT', 'BLKI', 'BAJE', 'COFO', 'VODA', 'SRFL', 'MAXI', 'CCRI', 'MNFL', 'TOPO',\
                           'VOLT', 'CNBK', 'GMRI', 'BHEL', 'AUFI', 'GODR', \
                           'ZEE', 'TTPW', 'ASOK', 'SRTR', 'BATA', 'SUTV', 'PAGE', 'PWFC', 'EXID', 'MGAS', 'CHLA',
                           'TVSM', 'UNBK', 'BOB', 'CUMM', 'TRCE', 'IDFB', 'CAST', 'FED', 'GLEN', \
                           'LICH', 'RECM', 'MMFS', 'INBF', 'AMAR', 'BOI', 'INIR', 'LTFH', 'APLO', 'SAIL', 'RATB',\
                           'JNSP', 'BFRG']

tickers_nifty_live_current = ["INFY", "TEML", "NEST", "HCLT", "WIPR", "ASPN", "CIPL", "BJFS", "REDY", "ICBK"]

tickers_nifty_live_all = ['VDAN','AXBK','TAMdv','ICBK','BAJA','ASPN','HDFC','PGRD','LART','COAL','TAMO','HCLT','NEST','SBI','GAIL','TCS','REDY','INFY','WIPR', 'SUN','TEML','ITC','YESB','TISC','HROM','BJFS','HDBK','NTPC','CIPL','ULTC']

tickers_midcap_live_all = ['BOB', 'MAXI', 'VODA', 'BHEL', 'BOI', 'GLEN', 'LICH', 'GODR', 'ESCO', 'INBF', 'JNSP', 'LTFH', 'MNFL', 'VOLT', 'PAGE', 'TRCE', 'EXID', 'GMRI', 'CNBK', 'ZEE', 'CHLA', 'BFRG', 'MGAS', 'APLO', 'BAJE', 'SRFL', 'SAIL', 'SUTV', 'IDFB', 'SRTR', 'TTPW', 'CCRI', 'RECM', 'AMAR', 'FED', 'MMFS', 'CUMM', 'TVSM', 'PWFC', 'BATA', 'CAST', 'ASOK', 'TOPO', 'BLKI', 'COFO', 'MINT', 'AUFI']

ticekrs_nasdaq_live_all = ['GOOGL', 'DISCA', 'VRSN', 'TCOM', 'CTXS', 'XEL', 'XLNX', 'PEP', 'WDAY', 'CSCO', 'BMRN', 'AKAM', 'NLOK', 'DLTR', 'CDNS', 'SBAC', 'SNPS', 'CHKP', 'CHTR', 'NVDA', 'TRIP', 'GILD', 'AMAT', 'UAL', 'MSFT', 'WYNN', 'SPLK', 'MU', 'KLAC', 'FB', 'MELI', 'AVGO', 'VRTX', 'CPRT', 'BKNG', 'VRSK', 'FAST', 'MAT', 'ADSK', 'AMD', 'TMUS', 'ALGN', 'CTAS', 'HSIC', 'XRAY', 'TSLA', 'CERN', 'MXIM', 'PCAR', 'OKTA', 'MAR', 'TTWO', 'SIRI', 'NXPI', 'PYPL', 'ULTA', 'SWKS', 'MRVL', 'MNST', 'SBUX', 'AAPL', 'HOLX', 'HON', 'INCY', 'TXN', 'CSGP', 'ATVI', 'HAS', 'ROST', 'SGEN', 'CSX', 'AMGN', 'NTAP', 'ORLY', 'VTRS', 'QRTEA', 'ADI', 'AEP', 'MDLZ', 'IDXX', 'PAYX', 'INTC', 'LBTYA', 'ANSS', 'BIDU', 'DXCM', 'CTSH', 'EBAY', 'ILMN', 'NTES', 'LULU', 'REGN', 'WDC', 'CDW', 'EXC', 'AAL', 'AMZN', 'CMCSA', 'COST', 'LRCX', 'JD', 'ASML', 'QCOM', 'KDP', 'BIIB', 'ISRG', 'WBA', 'VOD', 'EA', 'NFLX', 'INTU', 'EXPE', 'ADP', 'STX', 'MTCH', 'FISV', 'DISH']

tickers_arkk_live_all = ['TER', 'ROKU', 'NSTG', 'NVTA', 'TWTR', 'SE', 'FATE', 'Z', 'CRSP', 'MCRB', 'TRMB', 'MTLS', 'NTDOY', 'SSYS', 'EXAS', 'TSLA', 'IRDM', 'CGEN', 'EDIT', 'NTLA', 'TWLO', 'PRLB', 'SHOP', 'PACB', 'TWOU', 'SQ', 'CERS', 'TDOC', 'VCYT', 'DOCU', 'IOVA']

tickers_arkw_live_all = ['VUZI', 'ROKU', 'DIS', 'TWTR', 'SNAP', 'SE', 'Z', 'MELI', 'TTD', 'GBTC', 'NTDOY', 'TSLA', 'ETSY', 'LPSN', 'LC', 'TWLO', 'OKTA', 'SHOP', 'SPLK', 'TWOU', 'NNDM', 'PYPL', 'SQ', 'TDOC', 'VCYT', 'DOCU']

tickers_arkq_live_all = ['TER', 'VUZI', 'KTOS', 'JD', 'CAT', 'DE', 'TRMB', 'MGA', 'MTLS', 'SSYS', 'ISRG', 'TSLA', 'IRDM', 'SNPS', 'ESLT', 'NVDA', 'BYDDY', 'LMT', 'AVAV', 'ANSS', 'KMTUY', 'PRLB', 'BIDU', 'TWOU', 'NNDM', 'NXPI', 'TDY', 'DDD', 'GOOG']

tickers_nifty_all_minus_current = ['VDAN', 'BAJA', 'HDFC', 'LART', 'COAL', 'GAIL', 'TCS', 'ITC', 'TISC', 'HDBK']

ticekrs_nasdaq_leftover = ['MAR']

dir = 'InputFiles'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))

TEST = False
tickers = ["TLT", "^IXIC"]

if not TEST:
    for ticker in tickers:
        with open('Template_file.py', "rt") as fin:
            with open(f"InputFiles/{ticker}.py".replace("^",""), "wt") as fout:
                for line in fin:
                    if (ticker == "TLT") | (ticker == "^IXIC"):
                        fout.write(line.replace("ticker_inp", f"'{ticker}'").replace("interest_inp", "2"))
                    else:
                        fout.write(line.replace("ticker_inp", f"'{ticker}'").replace("interest_inp", "6"))

else:
    for ticker in tickers:
        with open('Tester.py', "rt") as fin:
            with open(f"InputFiles/{ticker}.py".replace("^",""), "wt") as fout:
                for line in fin:
                    if (ticker == "TLT") | (ticker == "^IXIC"):
                        fout.write(line.replace("ticker_inp", f"'{ticker}'").replace("interest_inp", "2"))
                    else:
                        fout.write(line.replace("ticker_inp", f"'{ticker}'").replace("interest_inp", "6"))