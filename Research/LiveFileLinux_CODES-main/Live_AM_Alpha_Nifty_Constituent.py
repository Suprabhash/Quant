import time
import warnings
import pandas as pd
import traceback
warnings.filterwarnings('ignore')

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from helper_functions import *
from helper_functions_constituent_alpha_for_dual_momo import  *
from CacheUpdater import recalibrate_indices, recalibrate_constituent_ticker

from datetime import timedelta
import os
import investpy
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from Portfolio_Values_Table_Updation_Logic import *

# import eikon as ek
# ek.set_app_key('9a249e0411184cf49e553b61a6e76c52d295ec17')

def get_data_alpha(ticker):

    if ticker == "GOLDBEES.NS":
        ticker_inv = "GBES"
    if ticker == "^NSEI":
        ticker_inv = "NSEI"

    temp_og = get_data(ticker, "yfinance", "india")

    if ticker == "GOLDBEES.NS":
        temp_og = temp_og[temp_og["Close"] < 100]

    today_data = yf.download(ticker, start=str(date.today() - timedelta(days=1)), interval="1m")
    if len(today_data)==0:
        today_time_close = temp_og.iloc[-1]["Date"]
    else:
        today_data.reset_index(inplace=True)
        today_data.drop(['Adj Close'], axis=1, inplace=True)
        if ticker == "GOLDBEES.NS":
            today_data = today_data.loc[today_data["Close"] > 1]

        today_time_close = today_data.iloc[-1]["Datetime"]

        temp_og = pd.concat([temp_og, pd.DataFrame([{"Date": pd.Timestamp(year=today_data.iloc[-1]["Datetime"].year,
                                                                          month=today_data.iloc[-1]["Datetime"].month,
                                                                          day=today_data.iloc[-1]["Datetime"].day),\
                                                     "Open": today_data.iloc[-1]["Open"],
                                                     "Close": today_data.iloc[-1]["Close"],
                                                     "High": today_data.iloc[-1]["High"],
                                                     "Low": today_data.iloc[-1]["Low"],
                                                     "Volume": today_data.iloc[-1]["Volume"]}])],axis=0).reset_index().drop(['index'], axis=1)
        temp_og.drop_duplicates(subset="Date",
                                keep='first', inplace=True)
    temp_og = add_fisher(temp_og)
    return temp_og, today_time_close

def get_data_constituents(ticker, api,country):

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

def get_data_alpha_investpy_yf(ticker, ticker_yfinance):

    if ticker == "GOLDBEES.NS":
        ticker_inv = "GBES"
    if ticker == "^NSEI":
        ticker_inv = "NSEI"

    temp_og = get_data_constituents(ticker, "investpy", "india")

    if ticker == "GOLDBEES.NS":
        temp_og = temp_og[temp_og["Close"] < 100]

    today_data = yf.download(ticker_yfinance, start=str(date.today() - timedelta(days=1)), interval="1m")
    today_data.reset_index(inplace=True)
    today_data.drop(['Adj Close'], axis=1, inplace=True)
    if ticker == "GOLDBEES.NS":
        today_data = today_data.loc[today_data["Close"] > 1]

    today_time_close = today_data.iloc[-1]["Datetime"]

    temp_og = pd.concat([temp_og, pd.DataFrame([{"Date": pd.Timestamp(year=today_data.iloc[-1]["Datetime"].year,
                                                                      month=today_data.iloc[-1]["Datetime"].month,
                                                                      day=today_data.iloc[-1]["Datetime"].day),\
                                                 "Open": today_data.iloc[-1]["Open"],
                                                 "Close": today_data.iloc[-1]["Close"],
                                                 "High": today_data.iloc[-1]["High"],
                                                 "Low": today_data.iloc[-1]["Low"],
                                                 "Volume": today_data.iloc[-1]["Volume"]}])],axis=0).reset_index().drop(['index'], axis=1)
    temp_og.drop_duplicates(subset="Date",
                            keep='first', inplace=True)
    temp_og = add_fisher(temp_og)
    return temp_og, today_time_close

def select_all_strategies(train_monthsf, datesf, temp_ogf, ticker, save=True):
    inputs =[]
    for date_i in range(len(datesf)-(int(train_monthsf/3)+1)):
        inputs.append([date_i, datesf, temp_ogf, train_monthsf])
    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results = pool.map(get_strategies_brute_force, inputs)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    res_test_update = [pd.DataFrame(columns=["Lookback", "Low Bound", "High Bound", "AvgWinLoss",\
                                       "Optimization_Years"])] * (len(datesf)-(int(24/3)+1))

    for i in range(len(results)):
        res_test_update[results[i][0]+int((train_monthsf-24)/3)] = pd.concat([res_test_update[results[i][0]],results[i][1].reset_index().drop(['index'], axis=1)], axis=0)

    if save==True:
        with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_{int(train_monthsf/12)}_All_Strategies.pkl','rb') as file:
            res_test = pickle.load(file)
        res_test.append(res_test_update[-1])
        with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_{int(train_monthsf/12)}_All_Strategies.pkl', 'wb') as file:
            pickle.dump(res_test, file)
    return res_test

def select_strategies_from_corr_filter(res_testf2,res_testf4,res_testf8, datesf, temp_ogf, num_opt_periodsf,num_strategiesf, ticker, save=True):
    train_monthsf = 24  #minimum optimization lookback
    res_total = [None]*(len(datesf)-(int(train_monthsf/3)+1))
    for i in range(len(datesf)-(int(train_monthsf/3)+1)):
        if num_opt_periodsf==1:
            res_total[i] = pd.concat([res_testf2[i]], axis = 0)
        if num_opt_periodsf==2:
            res_total[i] = pd.concat([res_testf2[i],res_testf4[i]], axis=0)
        if num_opt_periodsf==3:
            res_total[i] = pd.concat([res_testf2[i],res_testf4[i],res_testf8[i]], axis=0)
        res_total[i] = res_total[i].reset_index().drop(['index'], axis=1)

    ss_test_update = [None]*(len(datesf)-(int(train_monthsf/3)+1))
    res_test_update = [None]*(len(datesf)-(int(train_monthsf/3)+1))
    inputs = []
    for date_i in range(len(datesf)-(int(train_monthsf/3)+1)):
        inputs.append([date_i, datesf, temp_ogf,res_total, num_strategiesf,train_monthsf])
    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results_filtered = pool.map(corr_sortino_filter, inputs)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    for i in range(len(datesf)-(int(train_monthsf/3)+1)):
        ss_test_update[results_filtered[i][0]] = results_filtered[i][1]
        res_test_update[results_filtered[i][0]] = results_filtered[i][2]

    if save==True:
        with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{num_opt_periodsf}_Selected_Strategies_ss.pkl','rb') as file:
            ss_test = pickle.load(file)
        with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{num_opt_periodsf}_Selected_Strategies_res.pkl','rb') as file:
            res_test = pickle.load(file)

        ss_test.append(ss_test_update[-1])
        res_test.append(res_test_update[-1])

        with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{num_opt_periodsf}_Selected_Strategies_ss.pkl', 'wb') as file:
            pickle.dump(ss_test, file)
        with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{num_opt_periodsf}_Selected_Strategies_res.pkl', 'wb') as file:
            pickle.dump(res_test, file)

    return ss_test, res_test

def SendMail(subject_text_add, rebalance_recalibration_df, text_restricted,text_nifty, text_gold, text, printdf_nifty, printdf_gold, assets, ImgFileNameList, csv_file_path):
    msg = MIMEMultipart()
    msg['Subject'] = f'{subject_text_add} [ACSYS] Strategy Update on NIFTY50 Dual Momentum - Alpha & Accelerating Momentum'
    msg['From'] = 'algo_notifications@acsysindia.com'
    msg['Cc'] = 'suprabhashsahu@acsysindia.com, pratiksaxena@acsysindia.com, aditya@shankar.biz, divakarank@acsysindia.com'
    msg['To'] = 'algo_notifications@acsysindia.com'

    for ImgFileName in ImgFileNameList:
        with open(ImgFileName, 'rb') as f:
            img_data = f.read()
        image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
        msg.attach(image)

    with open(csv_file_path, 'rb') as file:
        msg.attach(MIMEApplication(file.read(), Name=csv_file_path))

    rebalance_recalibration_df = """\
                        <html>
                          <head></head>
                          <body>
                            {0}
                          </body>
                        </html>
                        """.format(rebalance_recalibration_df.to_html())

    rebalance_recalibration_df = MIMEText(rebalance_recalibration_df, 'html')
    msg.attach(rebalance_recalibration_df)

    text_restricted = MIMEText(text_restricted)
    msg.attach(text_restricted)

    text_nifty = MIMEText(text_nifty)
    msg.attach(text_nifty)

    strategies_nifty = """\
    <html>
      <head></head>
      <body>
        {0}
      </body>
    </html>
    """.format(printdf_nifty.to_html())

    part1 = MIMEText(strategies_nifty, 'html')
    msg.attach(part1)

    text_gold = MIMEText(text_gold)
    msg.attach(text_gold)

    strategies_gold = """\
        <html>
          <head></head>
          <body>
            {0}
          </body>
        </html>
        """.format(printdf_gold.to_html())

    part2 = MIMEText(strategies_gold, 'html')
    msg.attach(part2)

    text = MIMEText(text)
    msg.attach(text)

    assetsprint = """\
            <html>
              <head></head>
              <body>
                {0}
              </body>
            </html>
            """.format(assets.to_html())

    part3 = MIMEText(assetsprint, 'html')
    msg.attach(part3)

    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login('algo_notifications@acsysindia.com', 'esahYah8')
    s.sendmail('algo_notifications@acsysindia.com', ['suprabhashsahu@acsysindia.com', 'algo_notifications@acsysindia.com', 'pratiksaxena@acsysindia.com','divakarank@acsysindia.com', 'aditya@shankar.biz'], msg.as_string())  #
    s.quit()

def signal_print(inp, ticker):
    if inp == 0:
        signal = f"Neutral on {ticker}, Long on Fixed Income"
    else:
        signal = f"Long on {ticker}, Neutral on Fixed Income"
    return signal

def execute_nifty(ticker, number_of_optimization_periods,recalib_months,num_strategies,metric):
    # Download data
    temp_og, today_time_close = get_data_alpha(ticker)

    dates_all_ss = pd.date_range(start=str(temp_og.iloc[1]['Date'] + timedelta(days=365))[:10], end="2024-06-15",
                                 freq=f'3M')
    dates_ss = valid_dates(dates_all_ss)

    print(f"Importing selected strategies: {datetime.now()}")
    with open(
            f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_ss.pkl',
            'rb') as file:
        ss_test_imp = pickle.load(file)
    with open(
            f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_res.pkl',
            'rb') as file:
        res_test_imp = pickle.load(file)

    res_test = []
    ss_test = []
    dates = []

    date_i_max = 0
    for date_i in range(len(dates_ss) - (int(24 / 3) + 1)):
        if (3 * date_i) % recalib_months == 0:
            date_i_max = date_i

    if date_i_max>=len(ss_test_imp):
        recalibrate_indices(ticker, number_of_optimization_periods, "strategies")
        print(f"Importing selected strategies: {datetime.now()}")
        with open(
                f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_ss.pkl',
                'rb') as file:
            ss_test_imp = pickle.load(file)
        with open(
                f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_res.pkl',
                'rb') as file:
            res_test_imp = pickle.load(file)


    for date_i in range(len(dates_ss) - (int(24 / 3) + 1)):
        if (3 * date_i) % recalib_months == 0:
            dates.append(dates_ss[date_i + int(24 / 3)])
            ss_test.append(ss_test_imp[date_i])
            res_test.append(res_test_imp[date_i])

    dates.append(date.today()+timedelta(1))
    date_p = [date_i for date_i in range(len(dates) - 1)][-1]
    print(
        f"Selected Strategies for Testing period beginning: {str(dates[date_p])} and ending: {str(dates[date_p + 1])}")
    print(res_test[date_p])

    print(f"Importing Weights: {datetime.now()}")
    with open(
            f'{ticker}/weights/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.pkl',
            'rb') as file:
        weights = pickle.load(file)

    inputs = []

    if len(dates) - 2>= len(weights):
        recalibrate_indices(ticker, number_of_optimization_periods, "weights")

    for date_i in range(len(dates) - 1):
        inputs.append(
            [date_i, dates, temp_og, ss_test, res_test, num_strategies, weights[date_i], recalib_months, dates_ss])
    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results_backtest = pool.map(backtest_live, inputs)
    finally:  # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    results_final = pd.DataFrame()
    for tt in results_backtest:
        results_final = pd.concat([results_final, tt[0]], axis=0)
    temp_res = results_final

    temp_res['Return'] = np.log(temp_res['Close'] / temp_res['Close'].shift(1))
    temp_res['Market_Return'] = np.exp(temp_res['Return'].expanding().sum())-1
    temp_res['Strategy_Return'] = np.exp(temp_res['S_Return'].expanding().sum())-1
    temp_res['Portfolio Value'] = ((temp_res['Strategy_Return'] + 1) * 10000)
    temp_res.reset_index(inplace=True)
    temp_res_nifty = temp_res.copy()
    temp_res_nifty.columns = ["Date"] + [column + "_nifty" for column in temp_res_nifty.columns if column != 'Date']

    plt.plot(temp_res['Date'], temp_res['Market_Return'], color='black', label='Market Returns')
    plt.plot(temp_res['Date'], temp_res['Strategy_Return'], color='blue', label='Strategy Returns')
    plt.title('Strategy Backtest: Nifty Alpha')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(f"Performance{ticker}.jpg")
    plt.clf()

    text = ""
    text = text + "*"*100 + "\n"
    text = text + "NIFTY Alpha" + "\n"
    text = text + "Phase-1: Top strategies are ones which maximises the Average Win/Average Loss of the trades made over the lookback period" + "\n"
    text = text + "Top Strategies are selected based on lookbacks of 24 months" + "\n"
    text = text + f"Top Strategies are selected every {recalib_months} months" + "\n"
    text = text + f"Top {num_strategies} strategies are selected" + "\n"
    text = text + f"Strategies selected based on training data from: {str(dates[date_p] - relativedelta(months = 24))[:11]} to: {str(dates[date_p])[:11]} are selected" + "\n"
    text = text + f"Phase-2: Weights of the selected strategies are calculated such that rolling outperformance against benchmark is maximised over a lookback of 24 months "+ "\n"
    text = text + f"Last Recalibrated Nifty Alpha on {str(dates[-2])[:11]}" + "\n"
    # text = text + f"Recalibrating Nifty Alpha on {str(dates[-2] + relativedelta(months = recalib_months))[:11]}" + "\n" + "\n"
    text = text + "Alpha Stats for last 252 trading days:" + "\n"
    text = text + f"Sortino: {np.round(backtest_sortino(temp_res[-252:], 0, 0), 2)}" + "\n"
    text = text + f"Sharpe: {np.round(backtest_sharpe(temp_res[-252:], 0, 0), 2)}" + "\n"
    text = text + f"Rolling Sortino: {np.round(backtest_rolling_sortino(temp_res[-252:], 0, 0), 2)}" + "\n"
    text = text + f"Rolling CAGR: {np.round(backtest_rolling_cagr(temp_res[-252:], 0, 0), 2)}" + "\n"
    text = text + f"MaxDrawup/MaxDrawdown: {np.round(backtest_maxdrawup_by_maxdrawdown(temp_res[-252:], 0, 0), 2)}" + "\n" + "\n"
    text = text + f"Signal at : {str(today_time_close)[:19]}: {signal_print(temp_res.iloc[-1]['signal'],ticker)}" + "\n" f"Signal at Yesterday Close: {signal_print(temp_res.iloc[-2]['signal'],ticker)}" + "\n" f"Signal at Day before Yesterday Close: {signal_print(temp_res.iloc[-3]['signal'],ticker)}" + "\n" + "\n" + "Overall Performance:" + "\n" + f"Portfolio Value: {np.round(temp_res.iloc[-1]['Portfolio Value'], 2)}" + "\n" + "\n"
    text = text + "Selected Strategies: " + "\n" + "\n"
    date_recalibrate_nifty = dates[-2] + relativedelta(months=recalib_months)
    return temp_og,str(today_time_close)[:19],dates_all_ss,dates_ss,results_backtest[-1][1],temp_res_nifty, text, date_recalibrate_nifty

def execute_gold(ticker, number_of_optimization_periods,recalib_months,num_strategies,metric):

    # Download data
    temp_og, today_time_close = get_data_alpha(ticker)

    dates_all_ss = pd.date_range(start=str(temp_og.iloc[1]['Date'] + timedelta(days=365))[:10],
                                 end="2024-06-15", freq=f'3M')
    dates_ss = valid_dates(dates_all_ss)

    print(f"Importing selected strategies: {datetime.now()}")
    with open(
            f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_ss.pkl',
            'rb') as file:
        ss_test_imp = pickle.load(file)
    with open(
            f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_res.pkl',
            'rb') as file:
        res_test_imp = pickle.load(file)

    res_test = []
    ss_test = []
    dates = []

    date_i_max = 0
    for date_i in range(len(dates_ss) - (int(24 / 3) + 1)):
        if (3 * date_i) % recalib_months == 0:
            date_i_max = date_i

    if date_i_max >= len(ss_test_imp):
        recalibrate_indices(ticker, number_of_optimization_periods, "strategies")
        print(f"Importing selected strategies: {datetime.now()}")
        with open(
                f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_ss.pkl',
                'rb') as file:
            ss_test_imp = pickle.load(file)
        with open(
                f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_res.pkl',
                'rb') as file:
            res_test_imp = pickle.load(file)

    for date_i in range(len(dates_ss) - (int(24 / 3) + 1)):
        if (3 * date_i) % recalib_months == 0:
            dates.append(dates_ss[date_i + int(24 / 3)])
            ss_test.append(ss_test_imp[date_i])
            res_test.append(res_test_imp[date_i])

    dates.append(date.today()+timedelta(1))
    date_p = [date_i for date_i in range(len(dates) - 1)][-1]
    print(
        f"Selected Strategies for Testing period beginning: {str(dates[date_p])} and ending: {str(dates[date_p + 1])}")
    print(res_test[date_p])

    print(f"Importing Weights: {datetime.now()}")
    with open(
            f'{ticker}/weights/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.pkl',
            'rb') as file:
        weights = pickle.load(file)

    inputs = []

    if len(dates) - 2>= len(weights):
        recalibrate_indices(ticker, number_of_optimization_periods, "weights")

    for date_i in range(len(dates) - 1):
        inputs.append(
            [date_i, dates, temp_og, ss_test, res_test, num_strategies, weights[date_i], recalib_months,
             dates_ss])
    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results_backtest = pool.map(backtest_live, inputs)
    finally:  # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    results_final = pd.DataFrame()
    for tt in results_backtest:
        results_final = pd.concat([results_final, tt[0]], axis=0)
    temp_res = results_final

    temp_res['Return'] = np.log(temp_res['Close'] / temp_res['Close'].shift(1))
    temp_res['Market_Return'] = np.exp(temp_res['Return'].expanding().sum())-1
    temp_res['Strategy_Return'] = np.exp(temp_res['S_Return'].expanding().sum())-1
    temp_res['Portfolio Value'] = ((temp_res['Strategy_Return'] + 1) * 10000)
    temp_res.reset_index(inplace=True)
    temp_res_gold = temp_res.copy()
    temp_res_gold.columns = ["Date"] + [column + "_gold" for column in temp_res_gold.columns if
                                        column != 'Date']

    temp_res.to_csv("Gold_Alpha_Nifty_Non_Naive.csv")

    plt.plot(temp_res['Date'], temp_res['Market_Return'], color='black', label='Market Returns')
    plt.plot(temp_res['Date'], temp_res['Strategy_Return'], color='blue', label='Strategy Returns')
    plt.title('Strategy Backtest: Gold Alpha')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(f"Performance{ticker}.jpg")
    plt.clf()

    text = ""
    text = text + "*" * 100 + "\n"
    text = text + "GOLD Alpha" + "\n"
    text = text + "Phase-1: Top strategies are ones which maximises the Average Win/Average Loss of the trades made over the lookback period" + "\n"
    text = text + "Top Strategies are selected based on lookbacks of 24 and 48 months" + "\n"
    text = text + f"Top Strategies are selected every {recalib_months} months" + "\n"
    text = text + f"Top {num_strategies} strategies are selected" + "\n"
    text = text + f"24-month Lookback Strategies selected based on training data from: {str(dates[date_p] - relativedelta(months=24))[:11]} to: {str(dates[date_p])[:11]} are selected" + "\n"
    text = text + f"48-month Lookback Strategies selected based on training data from: {str(dates[date_p] - relativedelta(months=48))[:11]} to: {str(dates[date_p])[:11]} are selected" + "\n"
    text = text + f"Phase-2: Weights of the selected strategies are calculated such that rolling outperformance against benchmark is maximised over a lookback of 24 months " + "\n"
    text = text + f"Last Recalibrated Gold Alpha on {str(dates[-2])[:11]}" + "\n"
    # text = text + f"Recalibrating Gold Alpha on {str(dates[-2] + relativedelta(months = recalib_months))[:11]}" + "\n" + "\n"
    text = text + "Alpha Stats for last 252 trading days:" + "\n"
    text = text + f"Sortino: {np.round(backtest_sortino(temp_res[-252:], 0, 0), 2)}" + "\n"
    text = text + f"Sharpe: {np.round(backtest_sharpe(temp_res[-252:], 0, 0), 2)}" + "\n"
    text = text + f"Rolling Sortino: {np.round(backtest_rolling_sortino(temp_res[-252:], 0, 0), 2)}" + "\n"
    text = text + f"Rolling CAGR: {np.round(backtest_rolling_cagr(temp_res[-252:], 0, 0), 2)}" + "\n"
    text = text + f"MaxDrawup/MaxDrawdown: {np.round(backtest_maxdrawup_by_maxdrawdown(temp_res[-252:], 0, 0), 2)}" + "\n" + "\n"
    text = text + f"Signal at : {str(today_time_close)[:19]} : {signal_print(temp_res.iloc[-1]['signal'],ticker)}" + "\n" f"Signal at Yesterday Close: {signal_print(temp_res.iloc[-2]['signal'],ticker)}" + "\n" f"Signal at Day before Yesterday Close: {signal_print(temp_res.iloc[-3]['signal'],ticker)}" + "\n" + "\n" + "Overall Performance:" + "\n" + f"Portfolio Value: {np.round(temp_res.iloc[-1]['Portfolio Value'], 2)}" + "\n" + "\n"
    text = text + "Selected Strategies: " + "\n" + "\n"
    date_recalibrate_gold = dates[-2] + relativedelta(months=recalib_months)
    return temp_og,str(today_time_close)[:19],dates_all_ss,dates_ss,results_backtest[-1][1],temp_res_gold, text, date_recalibrate_gold

def execute_constituents(ticker, ticker_yfinance, number_of_optimization_periods,recalib_months,num_strategies,metric):

    # Download data
    temp_og, today_time_close = get_data_alpha_investpy_yf(ticker, ticker_yfinance)

    dates_all_ss = pd.date_range(start=str(temp_og.iloc[1]['Date'] + timedelta(days=365))[:10],
                                 end="2024-06-15", freq=f'3M')
    dates_ss = valid_dates(dates_all_ss)

    print(f"Importing selected strategies: {datetime.now()}")
    with open(
            f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_ss.pkl',
            'rb') as file:
        ss_test_imp = pickle.load(file)
    with open(
            f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_res.pkl',
            'rb') as file:
        res_test_imp = pickle.load(file)

    res_test = []
    ss_test = []
    dates = []

    date_i_max = 0
    for date_i in range(len(dates_ss) - (int(24 / 3) + 1)):
        if (3 * date_i) % recalib_months == 0:
            date_i_max = date_i
    print(f"date_i_max = {date_i_max}")
    if date_i_max >= len(ss_test_imp):
        recalibrate_constituent_ticker(ticker, number_of_optimization_periods, "strategies")
        print(f"Importing selected strategies: {datetime.now()}")
        with open(
                f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_ss.pkl',
                'rb') as file:
            ss_test_imp = pickle.load(file)
        with open(
                f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_res.pkl',
                'rb') as file:
            res_test_imp = pickle.load(file)

    for date_i in range(len(dates_ss) - (int(24 / 3) + 1)):
        if (3 * date_i) % recalib_months == 0:
            # try:
            dates.append(dates_ss[date_i + int(24 / 3)])
            ss_test.append(ss_test_imp[date_i])
            res_test.append(res_test_imp[date_i])
            # except Exception as e:
            #     print(e)
            #     print(f"length of ss_test_imp = {len(ss_test_imp)}")
            #     print(f"date_i = {date_i}")

    dates.append(date.today()+timedelta(1))
    date_p = [date_i for date_i in range(len(dates) - 1)][-1]
    print(
        f"Selected Strategies for Testing period beginning: {str(dates[date_p])} and ending: {str(dates[date_p + 1])}")
    print(res_test[date_p])

    print(f"Importing Weights: {datetime.now()}")
    with open(
            f'{ticker}/weights/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.pkl',
            'rb') as file:
        weights = pickle.load(file)

    inputs = []

    if len(dates) - 2>= len(weights):
        recalibrate_constituent_ticker(ticker, number_of_optimization_periods, "weights")
        with open(
                f'{ticker}/weights/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.pkl',
                'rb') as file:
            weights = pickle.load(file)

    for date_i in range(len(dates) - 1):
        inputs.append(
            [date_i, dates, temp_og, ss_test, res_test, num_strategies, weights[date_i], recalib_months,
             dates_ss])
    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results_backtest = pool.map(backtest_live, inputs)
    finally:  # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    results_final = pd.DataFrame()
    for tt in results_backtest:
        results_final = pd.concat([results_final, tt[0]], axis=0)
    temp_res = results_final

    temp_res['Return'] = np.log(temp_res['Close'] / temp_res['Close'].shift(1))
    temp_res['Market_Return'] = temp_res['Return'].expanding().sum()
    temp_res['Strategy_Return'] =temp_res['S_Return'].expanding().sum()
    temp_res['Portfolio Value'] = ((temp_res['Strategy_Return'] + 1) * 10000)
    temp_res.reset_index(inplace=True)
    temp_res_ticker = temp_res.copy()
    temp_res_ticker.columns = ["Date"] + [column + ticker for column in temp_res_ticker.columns if
                                        column != 'Date']

    plt.plot(temp_res['Date'], temp_res['Market_Return'], color='black', label='Market Returns')
    plt.plot(temp_res['Date'], temp_res['Strategy_Return'], color='blue', label='Strategy Returns')
    plt.title(f'Strategy Backtest: {ticker} Alpha')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(f"Performance{ticker}.jpg")
    plt.clf()
    date_recalibrate = dates[-2] + relativedelta(months=recalib_months)

    return temp_og,str(today_time_close)[:19],dates_all_ss,dates_ss,results_backtest[-1][1],temp_res_ticker, date_recalibrate

if __name__ == '__main__':

    running_for_the_first_time = True

    ticker_nifty = "^NSEI"
    number_of_optimization_periods_nifty = 1
    recalib_months_nifty = 3
    num_strategies_nifty = 5
    metric_nifty = 'outperformance'

    ticker_gold = "GOLDBEES.NS"
    number_of_optimization_periods_gold = 2
    recalib_months_gold = 6
    num_strategies_gold = 1
    metric_gold = 'outperformance'

    run_hour = 14
    run_minute = 15

    while True:
        if ((datetime.now().hour==run_hour) & (datetime.now().minute==run_minute) & (datetime.now().second==00)) or running_for_the_first_time:
            if (datetime.today().isoweekday() < 6):
                recalibrate_today = False
                text = ""
                print(f"Executing: {datetime.now()}")

                text_restricted = "ACSYS" + "\n" + "Restricted List is : [PGHH, PFIZER, AXISBANK]" + "\n"

                index = ".BSESN"
                with open(f'BSE_Constituents.pkl', 'rb') as file:
                    constituents = pickle.load(file)
                constituents_all = []
                for i in range(len(constituents)):
                    constituents_all = constituents_all + constituents.iloc[i][0]
                tickers = list(set(constituents_all))
                tickers = [ticker[:-3] for ticker in tickers]
                # Restricted Stock
                tickers.remove('AXBK')

                recalibrating_months = 1
                top_nassets = 10
                training_period = 24  # 24/48/96
                dates_recalibrating = valid_dates(
                    pd.date_range(start="2009-01-01", end="2024-06-15", freq=f'{recalibrating_months}M'))
                data_inp = prepare_portfolio_data(tickers, recalibrating_months, "datatables", "")

                data_inp = data_inp[data_inp['Date'] != '2020-11-14']
                data_inp = data_inp[data_inp['Date'] != '2020-11-17']

                if (pd.to_datetime(date.today()) in dates_recalibrating):
                    print(f"Start Recalibrating AM of BSE at {datetime.now()}")
                    with open(f'/nas/Algo/Live_emailer_updates/files_for_rebalance/BSE_RecalibPeriod_latest.pkl',
                              'rb') as file:
                        latest = pickle.load(file)
                    with open(f'BSE_RecalibPeriod_{int(1)}.pkl', 'rb') as file:
                        assets = pickle.load(file)
                    assets.append(latest)
                    with open(f'BSE_RecalibPeriod_{int(1)}.pkl', 'wb') as file:
                        pickle.dump(assets, file)
                    print(f"End Recalibrating AM of BSE at {datetime.now()}")

                with open(f'BSE_RecalibPeriod_{int(1)}.pkl', 'rb') as file:
                    assets = pickle.load(file)

                dates_recalibrate = []

                temp_og_nifty, signal_time_nifty, dates_all_ss_nifty, dates_ss_nifty, printdf_nifty, temp_res_nifty, text_nifty, date_recalibrate_nifty = execute_nifty(ticker_nifty,number_of_optimization_periods_nifty,recalib_months_nifty,num_strategies_nifty,metric_nifty)
                temp_og_gold, signal_time_gold, dates_all_ss_gold, dates_ss_gold, printdf_gold, temp_res_gold, text_gold, date_recalibrate_gold = execute_gold(ticker_gold,number_of_optimization_periods_gold,recalib_months_gold,num_strategies_gold,metric_gold)

                dates_recalibrate.append({"Rebalancing Date": "Nifty", "Date": date_recalibrate_nifty})
                dates_recalibrate.append({"Rebalancing Date": "Gold", "Date": date_recalibrate_gold})

                if temp_res_nifty.iloc[-1]['signal_nifty']==1:
                    subject_text_add = "[Long on Index]"
                if (temp_res_nifty.iloc[-1]['signal_nifty']==0) & (temp_res_gold.iloc[-1]['signal_gold']==1):
                    subject_text_add = "[Long on Gold]"
                if (temp_res_nifty.iloc[-1]['signal_nifty']==0) & (temp_res_gold.iloc[-1]['signal_gold']==0):
                    subject_text_add = "[Long on Cash]"

                data_inp_backtest = pd.concat([data_inp.set_index('Date'), temp_res_nifty.set_index('Date'), temp_res_gold.set_index('Date')], axis=1, join='inner').reset_index()

                top_nassets = 9
                rebalancing_months = 12
                starting_point = 2

                dates_rebalancing = []
                assetsb = []

                for date_i in range(len(dates_recalibrating) - (int(training_period / recalibrating_months) + 1)):
                    if ((recalibrating_months * date_i) % rebalancing_months == starting_point) & (
                            dates_recalibrating[date_i + int(training_period / recalibrating_months)] >=
                            data_inp_backtest["Date"][0]):
                        dates_rebalancing.append(
                            dates_recalibrating[date_i + int(training_period / recalibrating_months)])
                        assetsb.append(assets[date_i].iloc[:top_nassets])
                dates_rebalancing.append(date.today() + timedelta(1))

                tickers = return_all_tickers_over_backtest(assets)

                constituent_alpha_params = {
                    'TAMdv': {"ticker_yfinance": "TATAMTRDVR.NS", "number_of_optimization_periods": 1,
                              "recalib_months": 12, "num_strategies": 5, "metric": 'rolling_sharpe'},
                    'SBI': {"ticker_yfinance": "SBIN.NS", "number_of_optimization_periods": 3, "recalib_months": 3,
                            "num_strategies": 7, "metric": 'rolling_sortino'},
                    'NEST': {"ticker_yfinance": "NESTLEIND.NS", "number_of_optimization_periods": 3,
                             "recalib_months": 6, "num_strategies": 5, "metric": 'rolling_sortino'},
                    'INFY': {"ticker_yfinance": "INFY.NS", "number_of_optimization_periods": 1, "recalib_months": 12,
                             "num_strategies": 5, "metric": 'outperformance'},
                    'TCS': {"ticker_yfinance": "TCS.NS", "number_of_optimization_periods": 3, "recalib_months": 3,
                            "num_strategies": 7, "metric": 'outperformance'},
                    'COAL': {"ticker_yfinance": "COALINDIA.NS", "number_of_optimization_periods": 3,
                             "recalib_months": 12, "num_strategies": 7, "metric": 'rolling_sortino'},
                    'HCLT': {"ticker_yfinance": "HCLTECH.NS", "number_of_optimization_periods": 2, "recalib_months": 6,
                             "num_strategies": 5, "metric": 'rolling_cagr'},
                    'NTPC': {"ticker_yfinance": "NTPC.NS", "number_of_optimization_periods": 3, "recalib_months": 12,
                             "num_strategies": 7, "metric": 'rolling_sortino'},
                    'ICBK': {"ticker_yfinance": "ICICIBANK.NS", "number_of_optimization_periods": 2,
                             "recalib_months": 6, "num_strategies": 7, "metric": 'rolling_sortino'},
                    'LART': {"ticker_yfinance": "LT.NS", "number_of_optimization_periods": 3, "recalib_months": 12,
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
                    'REDY': {"ticker_yfinance": "DRREDDY.NS", "number_of_optimization_periods": 2, "recalib_months": 12,
                             "num_strategies": 7, "metric": 'rolling_cagr'},
                    'TEML': {"ticker_yfinance": "TECHM.NS", "number_of_optimization_periods": 1, "recalib_months": 3,
                             "num_strategies": 7, "metric": 'outperformance'},
                    'CIPL': {"ticker_yfinance": "CIPLA.NS", "number_of_optimization_periods": 1, "recalib_months": 12,
                             "num_strategies": 5, "metric": 'outperformance'},
                    'ULTC': {"ticker_yfinance": "ULTRACEMCO.NS", "number_of_optimization_periods": 2,
                             "recalib_months": 3, "num_strategies": 3, "metric": 'rolling_sharpe'},
                    'BJFS': {"ticker_yfinance": "BAJAJFINSV.NS", "number_of_optimization_periods": 2,
                             "recalib_months": 6, "num_strategies": 3, "metric": 'rolling_sortino'},
                    'HDFC': {"ticker_yfinance": "HDFC.NS", "number_of_optimization_periods": 2, "recalib_months": 6,
                             "num_strategies": 7, "metric": 'rolling_sharpe'},
                    'SUN': {"ticker_yfinance": "SUNPHARMA.NS", "number_of_optimization_periods": 3,
                            "recalib_months": 12, "num_strategies": 3, "metric": 'outperformance'},
                    'ITC': {"ticker_yfinance": "ITC.NS", "number_of_optimization_periods": 2, "recalib_months": 3,
                            "num_strategies": 5, "metric": 'rolling_sortino'},
                    'WIPR': {"ticker_yfinance": "WIPRO.NS", "number_of_optimization_periods": 2, "recalib_months": 3,
                             "num_strategies": 3, "metric": 'rolling_sharpe'},
                    'GAIL': {"ticker_yfinance": "GAIL.NS", "number_of_optimization_periods": 1, "recalib_months": 12,
                             "num_strategies": 1, "metric": 'rolling_sortino'},
                    'VDAN': {"ticker_yfinance": "VEDL.NS", "number_of_optimization_periods": 2, "recalib_months": 12,
                             "num_strategies": 1, "metric": 'maxdrawup_by_maxdrawdown'},
                    'PGRD': {"ticker_yfinance": "POWERGRID.NS", "number_of_optimization_periods": 2,
                             "recalib_months": 12, "num_strategies": 3, "metric": 'rolling_sortino'},
                    'HROM': {"ticker_yfinance": "HEROMOTOCO.NS", "number_of_optimization_periods": 2,
                             "recalib_months": 12, "num_strategies": 5, "metric": 'rolling_sortino'},
                    'AXBK': {"ticker_yfinance": "AXISBANK.NS", "number_of_optimization_periods": 1,
                             "recalib_months": 12, "num_strategies": 7, "metric": 'outperformance'},
                    'YESB': {"ticker_yfinance": "YESBANK.NS", "number_of_optimization_periods": 3, "recalib_months": 12,
                             "num_strategies": 5, "metric": 'maxdrawup_by_maxdrawdown'},
                    'ONGC': {"ticker_yfinance": "ONGC.NS", "number_of_optimization_periods": 2, "recalib_months": 3,
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
                    'TITN': {"ticker_yfinance": "TITAN.NS", "number_of_optimization_periods": 2,
                             "recalib_months": 6,
                             "num_strategies": 7, "metric": 'outperformance'},
                    'RELI': {"ticker_yfinance": "RELIANCE.NS", "number_of_optimization_periods": 2,
                             "recalib_months": 3,
                             "num_strategies": 1, "metric": 'rolling_sortino'},
                    'BJFN': {"ticker_yfinance": "BAJFINANCE.NS", "number_of_optimization_periods": 2,
                             "recalib_months": 12,
                             "num_strategies": 7, "metric": 'outperformance'},
                    'INBK': {"ticker_yfinance": "INDUSINDBK.NS", "number_of_optimization_periods": 3,
                             "recalib_months": 3,
                             "num_strategies": 3, "metric": 'rolling_sharpe'},
                    'MAHM': {"ticker_yfinance": "M&M.NS", "number_of_optimization_periods": 1,
                             "recalib_months": 12,
                             "num_strategies": 5, "metric": 'rolling_sharpe'},
                    'LUPN': {"ticker_yfinance": "LUPIN.NS", "number_of_optimization_periods": 3,
                             "recalib_months": 6,
                             "num_strategies": 3, "metric": 'rolling_sharpe'},
                    'VODA': {"ticker_yfinance": "IDEA.NS", "number_of_optimization_periods": 2, "recalib_months": 12,
                             "num_strategies": 5, "metric": 'rolling_sortino'},
                    'MRTI': {"ticker_yfinance": "MARUTI.NS", "number_of_optimization_periods": 2, "recalib_months": 12,
                             "num_strategies": 1, "metric": 'rolling_sortino'},
                    'EICH': {"ticker_yfinance": "EICHERMOT.NS", "number_of_optimization_periods": 1,
                             "recalib_months": 3, "num_strategies": 7, "metric": 'outperformance'},
                    'SUZL': {"ticker_yfinance": "SUZLON.NS", "number_of_optimization_periods": 1, "recalib_months": 12,
                             "num_strategies": 5, "metric": 'rolling_sharpe'},
                }

                # for i, ticker in enumerate(tickers):
                #     print("*"*100)
                #     print(ticker)
                #     temp_og_constituent, _, _, _, _, temp_res_constituent = execute_constituents(tickers[i], constituent_alpha_params[tickers[i]]["ticker_yfinance"],constituent_alpha_params[tickers[i]]["number_of_optimization_periods"],constituent_alpha_params[tickers[i]]["recalib_months"],constituent_alpha_params[tickers[i]]["num_strategies"],constituent_alpha_params[tickers[i]]["metric"])
                #     if i==0:
                #         constituent_signals = temp_res_constituent[["Date", f"signal{ticker}"]].set_index("Date")
                #     else:
                #         constituent_signals = pd.concat([constituent_signals, temp_res_constituent[["Date", f"signal{ticker}"]].set_index("Date")], axis=1, join="outer")

                for i, ticker in enumerate(tickers):
                    # if ticker not in ['ULTC']:
                    #     continue
                    print("*" * 100)
                    print(ticker)
                    temp_og_constituent, _, _, _, _, temp_res_constituent, date_recalibrate = execute_constituents(
                        tickers[i], constituent_alpha_params[tickers[i]]["ticker_yfinance"],
                        constituent_alpha_params[tickers[i]]["number_of_optimization_periods"],
                        constituent_alpha_params[tickers[i]]["recalib_months"],
                        constituent_alpha_params[tickers[i]]["num_strategies"],
                        constituent_alpha_params[tickers[i]]["metric"])
                    dates_recalibrate.append({"Rebalancing Date": ticker, "Date": date_recalibrate})
                    if i == 0:
                        constituent_signals = temp_res_constituent[["Date", f"signal{ticker}"]].set_index("Date")
                    else:
                        constituent_signals = pd.concat(
                            [constituent_signals, temp_res_constituent[["Date", f"signal{ticker}"]].set_index("Date")],
                            axis=1, join="outer")


                # constituent_signals["signalLUPN"] = 0

                data_inp_backtest = pd.concat([data_inp_backtest.set_index("Date"), constituent_signals], axis=1, join="inner")
                data_inp_backtest.reset_index(inplace=True)

                results_final, unit_ticker, unit_gold = backtest_Alpha_AM_Nifty_ConstituentAlphas(dates_rebalancing, data_inp_backtest, assetsb)

                bench = yf.download("^BSESN", start='2007-01-01', end=str(date.today()+timedelta(1)))
                bench = bench.loc[bench["Close"] > 1]
                bench["Return"] = np.log(bench["Close"] / bench["Close"].shift(1))

                results_final["S_Return"] = pd.DataFrame(np.log(results_final["Pvalue"] / results_final["Pvalue"].shift(1)))
                temp_res = pd.concat([results_final, bench["Return"]], join="inner", axis=1)
                temp_res['Market_Return'] = np.exp(temp_res['Return'].expanding().sum())-1
                temp_res['Strategy_Return'] = np.exp(temp_res['S_Return'].expanding().sum())-1
                temp_res['Portfolio Value'] = temp_res['Pvalue']
                temp_res = temp_res.reset_index().rename(columns={'index': "Date"})
                temp_res.to_csv("Nifty50_NonNaive.csv")

                azure_update_status_message = update_nifty_midcap_daily("Nifty50_NonNaive.csv")
                azure_update_status_message = azure_update_status_message + update_gold_daily("Gold_Alpha_Nifty_Non_Naive.csv")

                plt.plot(temp_res['Date'], temp_res['Market_Return'], color='black', label='Market Returns')
                plt.plot(temp_res['Date'], temp_res['Strategy_Return'], color='blue', label='Strategy Returns')
                plt.title('Strategy Backtest: Dual Momentum')
                plt.legend(loc=0)
                plt.tight_layout()
                plt.tight_layout()
                plt.savefig("PerformanceNIFTY.jpg")
                plt.clf()

                text = text + "*" * 100 + "\n"
                text = text + "DUAL MOMENTUM STRATEGY" + "\n"
                text = text + f"Recalibrated every {rebalancing_months} months, with training data of {training_period - recalibrating_months} months with a forward return  of {recalibrating_months} month" + "\n"
                text = text + f"The number of assets being selected are: {top_nassets}"
                text = text + f"Last Recalibrated Accelerating Momentum on {str(dates_rebalancing[-2])[:11]}" + "\n"
                # text = text + f"Recalibrating Accelerating Momentum on {str(dates_rebalancing[-2] + relativedelta(months=rebalancing_months))[:11]}" + "\n" + "\n"
                text = text + f"Strategies selected based on training data from: {str(dates_rebalancing[-2] - relativedelta(months=training_period))[:11]} to: {str(dates_rebalancing[-2] - relativedelta(months=recalibrating_months))[:11]} are selected" + "\n"
                text = text + f"Units and Momentum Values were calculated on {str(dates_rebalancing[-2])[:11]}" + "\n"
                text = text + "On Rebalancing Day, if Nifty Alpha is long, 100% of the Portolfio are allocated to the stocks below.\nIf Nifty Alpha is Neutral and Gold Alpha is Long, 50% of the Portfolio is allocated to GOLDBEES.NS and 50% to Fixed Income.\nIf both Alphas are Neutral, 100% of the Portfolio are allocated to Fixed Income" + "\n" + "\n"
                text = text + "Stats for last 252 trading days:" + "\n"
                text = text + f"Sortino: {np.round(backtest_sortino(temp_res[-252:], 0, 0), 2)}" + "\n"
                text = text + f"Sharpe: {np.round(backtest_sharpe(temp_res[-252:], 0, 0), 2)}" + "\n"
                text = text + f"Rolling Sortino: {np.round(backtest_rolling_sortino(temp_res[-252:], 0, 0), 2)}" + "\n"
                text = text + f"Rolling Sharpe: {np.round(backtest_rolling_sharpe(temp_res[-252:], 0, 0), 2)}" + "\n"
                # text = text + f"Rolling CAGR: {np.round(backtest_rolling_cagr(temp_res[-252:], 0, 0), 2)}" + "\n"
                text = text + f"Rolling MaxDrawup/MaxDrawdown: {np.round(backtest_maxdrawup_by_maxdrawdown(temp_res[-252:], 0, 0), 2)}" + "\n"
                text = text + f"Rolling Outperformance: {np.round(backtest_outperformance(temp_res[-252:], 0, 0), 2)}"
                text = text + "\n" + "\n" + "Overall Performance:" + "\n"
                text = text + f" A Portfolio of Rs {np.round(temp_res.iloc[0]['Portfolio Value'] / 10000000, 4)} Cr invested on: {str(dates_rebalancing[0])[:11]} is now valued at: {np.round(temp_res.iloc[-1]['Portfolio Value'] / 10000000, 4)} Cr." + "\n" + "\n"
                text = text + "Selected Assets: " + "\n"
                text = text + f"Number of units of GOLDBEES.NS: {unit_gold}" + "\n"
                text = text + f"{azure_update_status_message}" + "\n"
                # text = f"Signal at : {str(today_time_close)[:19]} : {signal_print(temp_res.iloc[-1]['signal'])}" + "\n" f"Signal at Yesterday Close: {signal_print(temp_res.iloc[-2]['signal'])}" + "\n" f"Signal at Day before Yesterday Close: {signal_print(temp_res.iloc[-3]['signal'])}" + "\n" + "\n" + "Overall Performance:" + "\n" + f"Portfolio Value: {np.round(temp_res.iloc[-1]['Portfolio Value'], 2)}" + "\n" + "\n" + text

                emailer_consts = pd.concat([assetsb[-1][["Ticker", "Accelerating Momentum"]].set_index("Ticker"),
                                            pd.DataFrame([unit_ticker]).transpose().rename(columns={0: "Units"})],
                                           axis=1)

                emailer_consts["Signal"] = 1

                for ticker in list(assetsb[-1]["Ticker"]):
                    if constituent_signals[f"signal{ticker}"].iloc[-1] == 0:
                        emailer_consts.loc[ticker, "Units"] = 0
                        emailer_consts.loc[ticker, "Signal"] = 0

                const_alpha_plots = []
                for ticker in list(assetsb[-1]["Ticker"]):
                    const_alpha_plots.append(f"Performance{ticker}.jpg")

                const_alpha_plots = const_alpha_plots + ["Performance^NSEI.jpg", "PerformanceGOLDBEES.NS.jpg", "PerformanceNIFTY.jpg"]

                dates_recalibrate.append({"Rebalancing Date": "Accelerating Momentum",
                                          "Date": dates_rebalancing[-2] + relativedelta(months=rebalancing_months)})

                if running_for_the_first_time == True:
                    pass
                else:
                    SendMail(subject_text_add, pd.DataFrame(dates_recalibrate).set_index("Rebalancing Date").sort_values(by="Date", ascending=True), text_restricted, text_nifty, text_gold, text, printdf_nifty, printdf_gold,
                     emailer_consts,const_alpha_plots, "Nifty50_NonNaive.csv")

        if running_for_the_first_time:
            pass
        else:
            if datetime.now() < datetime.now().replace(hour=run_hour).replace(minute=run_minute).replace(second=30):
                continue

        running_for_the_first_time = False

        print(f"Sleeping: {datetime.now()}")

        time_now = datetime.now()
        next_run = datetime.now()
        try:
            if (datetime.now().hour < run_hour) & (datetime.now().minute < run_minute):
                next_run = next_run.replace(day=next_run.day).replace(hour=run_hour).replace(minute=run_minute).replace(
                    second=00)
            else:
                next_run = next_run.replace(day=next_run.day + 1).replace(hour=run_hour).replace(
                    minute=run_minute).replace(second=00)
        except:
            if next_run.month == 12:
                next_run = next_run.replace(day=1).replace(month=1).replace(year=next_run.year + 1).replace(
                    hour=run_hour).replace(minute=run_minute).replace(second=00)
            else:
                next_run = next_run.replace(day=1).replace(month=next_run.month + 1).replace(hour=run_hour).replace(
                    minute=run_minute).replace(second=00)

        print(f"Supposed to wake up at: {datetime.now() + timedelta(seconds=(next_run - time_now).seconds - 150)}")
        time.sleep((next_run - time_now).seconds - 150)
        print(f"Woken Up: {datetime.now()}")