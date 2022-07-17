import time
import warnings
warnings.filterwarnings('ignore')

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from helper_functions import *

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

# import eikon as ek
# ek.set_app_key('9a249e0411184cf49e553b61a6e76c52d295ec17')

def get_data_alpha(ticker):

    temp_og = get_data(ticker, "yfinance", "")

    today_data = yf.download(ticker, start=str(date.today() - timedelta(days=1)), interval="1m")

    if len(today_data)!=0:
        today_data.reset_index(inplace=True)
        today_data.drop(['Adj Close'], axis=1, inplace=True)

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
    else:
        today_time_close = temp_og.iloc[-1]["Date"]

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

def SendMail(subject_text_add, rebalance_recalibration_df, text_restricted,text_nasdaq, text_tlt, text, printdf_nasdaq, printdf_tlt, assets, ImgFileNameList, csv_file_path):
    msg = MIMEMultipart()
    msg['Subject'] = f'{subject_text_add} [ADITYA INDIVIDUAL] Strategy Update on NASDAQ Dual Momentum - Alpha & Accelerating Momentum'
    msg['From'] = 'algo_notifications@acsysindia.com'
    msg['Cc'] = 'suprabhashsahu@acsysindia.com, aditya@shankar.biz, pratiksaxena@acsysindia.com ' #
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

    text_nasdaq = MIMEText(text_nasdaq)
    msg.attach(text_nasdaq)

    strategies_nasdaq = """\
    <html>
      <head></head>
      <body>
        {0}
      </body>
    </html>
    """.format(printdf_nasdaq.to_html())

    part1 = MIMEText(strategies_nasdaq, 'html')
    msg.attach(part1)

    text_tlt = MIMEText(text_tlt)
    msg.attach(text_tlt)

    strategies_tlt = """\
        <html>
          <head></head>
          <body>
            {0}
          </body>
        </html>
        """.format(printdf_tlt.to_html())

    part2 = MIMEText(strategies_tlt, 'html')
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
    s.sendmail('algo_notifications@acsysindia.com', ['suprabhashsahu@acsysindia.com', 'algo_notifications@acsysindia.com', 'aditya@shankar.biz', 'pratiksaxena@acsysindia.com'], msg.as_string())  #
    s.quit()

def signal_print(inp, ticker):
    if inp == 0:
        signal = f"Neutral on {ticker}, Long on Fixed Income"
    else:
        signal = f"Long on {ticker}, Neutral on Fixed Income"
    return signal

def execute_nasdaq(ticker, number_of_optimization_periods,recalib_months,num_strategies,metric):
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

    initial_amount = 2441200
    current_balance = initial_amount
    equity_allocation = 0
    cash_allocation = 0
    portfolio_value = pd.DataFrame()

    units_equity = 0

    for i in range(len(temp_res)):

        signal = temp_res["signal"].iloc[i]

        if signal == 1:
            equity_allocation = current_balance
            cash_allocation = 0
        else:
            equity_allocation = 0
            cash_allocation = current_balance

        if ((temp_res["signal"].iloc[i] == 1) & (temp_res["signal"].shift(1).fillna(0).iloc[i] == 0)):
            units_equity = equity_allocation / temp_res.iloc[i]["Close"]

        if ((temp_res["signal"].iloc[i] == 0) & (temp_res["signal"].shift(1).fillna(1).iloc[i] == 1)):
            units_equity = 0

        if signal == 1:
            equity_allocation = units_equity * temp_res.iloc[i]["Close"]

        cash_allocation = cash_allocation * (1 + 6 / 25200)
        current_balance = equity_allocation + cash_allocation
        portfolio_day = {'Date': temp_res.index[i], 'Signal_backtest': signal, 'units_equity': units_equity,
                         'equity_allocation': equity_allocation, 'cash_allocation': cash_allocation,
                         'Strategy_Return': current_balance}
        portfolio_day = pd.DataFrame([portfolio_day])
        portfolio_day = portfolio_day.set_index("Date")
        portfolio_value = pd.concat([portfolio_value, portfolio_day], axis=0, join="outer")

    temp_res = pd.concat([temp_res, portfolio_value], axis=1, join="inner")
    temp_res["Market_Return"] = (initial_amount / temp_res.iloc[0]["Close"]) * temp_res["Close"]
    temp_res.reset_index(inplace=True)
    temp_res_nasdaq = temp_res.copy()
    temp_res_nasdaq.columns = ["Date"] + [column + "_nasdaq" for column in temp_res_nasdaq.columns if column != 'Date']

    plt.plot(temp_res['Date'], temp_res['Market_Return'], color='black', label='Market Returns')
    plt.plot(temp_res['Date'], temp_res['Strategy_Return'], color='blue', label='Strategy Returns')
    plt.title('Strategy Backtest: Nasdaq Alpha')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(f"Performance{ticker}.jpg")
    plt.clf()

    text = ""
    text = text + "*"*100 + "\n"
    text = text + "NASDAQ Alpha" + "\n"
    text = text + "Phase-1: Top strategies are ones which maximises the Average Win/Average Loss of the trades made over the lookback period" + "\n"
    text = text + "Top Strategies are selected based on lookbacks of 24 and 48 months" + "\n"
    text = text + f"Top Strategies are selected every {recalib_months} months" + "\n"
    text = text + f"Top {num_strategies} strategies are selected" + "\n"
    text = text + f"24-month Lookback Strategies selected based on training data from: {str(dates[date_p] - relativedelta(months=24))[:11]} to: {str(dates[date_p])[:11]} are selected" + "\n"
    text = text + f"48-month Lookback Strategies selected based on training data from: {str(dates[date_p] - relativedelta(months=48))[:11]} to: {str(dates[date_p])[:11]} are selected" + "\n"
    text = text + f"Phase-2: Weights of the selected strategies are calculated such that maxdrawup_by_maxdrawdown against benchmark is maximised over a lookback of 24 months "+ "\n"
    text = text + f"Last Recalibrated Nasdaq Alpha on {str(dates[-2])[:11]}" + "\n"
    # text = text + f"Recalibrating Nasdaq Alpha on {str(dates[-2] + relativedelta(months = recalib_months))[:11]}" + "\n" + "\n"
    text = text + "Alpha Stats for last 252 trading days:" + "\n"
    text = text + f"Sortino: {np.round(backtest_sortino(temp_res[-252:], 0, 0), 2)}" + "\n"
    text = text + f"Sharpe: {np.round(backtest_sharpe(temp_res[-252:], 0, 0), 2)}" + "\n"
    text = text + f"Rolling Sortino: {np.round(backtest_rolling_sortino(temp_res[-252:], 0, 0), 2)}" + "\n"
    text = text + f"Rolling CAGR: {np.round(backtest_rolling_cagr(temp_res[-252:], 0, 0), 2)}" + "\n"
    text = text + f"MaxDrawup/MaxDrawdown: {np.round(backtest_maxdrawup_by_maxdrawdown(temp_res[-252:], 0, 0), 2)}" + "\n" + "\n"
    text = text + f"Signal at : {str(today_time_close)[:19]}: {signal_print(temp_res.iloc[-1]['signal'],ticker)}" + "\n" f"Signal at Yesterday Close: {signal_print(temp_res.iloc[-2]['signal'],ticker)}" + "\n" f"Signal at Day before Yesterday Close: {signal_print(temp_res.iloc[-3]['signal'],ticker)}" + "\n" + "\n" + "Overall Performance:" + "\n" + f"Portfolio Value: {np.round(temp_res.iloc[-1]['Strategy_Return'], 2)}" + "\n" + "\n"
    text = text + "Selected Strategies: " + "\n" + "\n"
    date_recalibrate_nasdaq = dates[-2] + relativedelta(months=recalib_months)
    return temp_og,str(today_time_close)[:19],dates_all_ss,dates_ss,results_backtest[-1][1],temp_res_nasdaq, text, date_recalibrate_nasdaq

def execute_tlt(ticker, number_of_optimization_periods,recalib_months,num_strategies,metric):

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

    initial_amount = 2441200
    current_balance = initial_amount
    equity_allocation = 0
    cash_allocation = 0
    portfolio_value = pd.DataFrame()

    units_equity = 0

    for i in range(len(temp_res)):

        signal = temp_res["signal"].iloc[i]

        if signal == 1:
            equity_allocation = current_balance
            cash_allocation = 0
        else:
            equity_allocation = 0
            cash_allocation = current_balance

        if ((temp_res["signal"].iloc[i] == 1) & (temp_res["signal"].shift(1).fillna(0).iloc[i] == 0)):
            units_equity = equity_allocation / temp_res.iloc[i]["Close"]

        if ((temp_res["signal"].iloc[i] == 0) & (temp_res["signal"].shift(1).fillna(1).iloc[i] == 1)):
            units_equity = 0

        if signal == 1:
            equity_allocation = units_equity * temp_res.iloc[i]["Close"]

        cash_allocation = cash_allocation * (1 + 6 / 25200)
        current_balance = equity_allocation + cash_allocation
        portfolio_day = {'Date': temp_res.index[i], 'Signal_backtest': signal, 'units_equity': units_equity,
                         'equity_allocation': equity_allocation, 'cash_allocation': cash_allocation,
                         'Strategy_Return': current_balance}
        portfolio_day = pd.DataFrame([portfolio_day])
        portfolio_day = portfolio_day.set_index("Date")
        portfolio_value = pd.concat([portfolio_value, portfolio_day], axis=0, join="outer")

    temp_res = pd.concat([temp_res, portfolio_value], axis=1, join="inner")
    temp_res["Market_Return"] = (initial_amount / temp_res.iloc[0]["Close"]) * temp_res["Close"]
    temp_res.reset_index(inplace=True)
    temp_res_tlt = temp_res.copy()
    temp_res_tlt.columns = ["Date"] + [column + "_tlt" for column in temp_res_tlt.columns if
                                        column != 'Date']

    plt.plot(temp_res['Date'], temp_res['Market_Return'], color='black', label='Market Returns')
    plt.plot(temp_res['Date'], temp_res['Strategy_Return'], color='blue', label='Strategy Returns')
    plt.title('Strategy Backtest: TLT Alpha')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(f"Performance{ticker}.jpg")
    plt.clf()

    text = ""
    text = text + "*" * 100 + "\n"
    text = text + "TLT Alpha" + "\n"
    text = text + "Phase-1: Top strategies are ones which maximises the Average Win/Average Loss of the trades made over the lookback period" + "\n"
    text = text + "Top Strategies are selected based on lookbacks of 24 and 48 months" + "\n"
    text = text + f"Top Strategies are selected every {recalib_months} months" + "\n"
    text = text + f"Top {num_strategies} strategies are selected" + "\n"
    text = text + f"24-month Lookback Strategies selected based on training data from: {str(dates[date_p] - relativedelta(months=24))[:11]} to: {str(dates[date_p])[:11]} are selected" + "\n"
    text = text + f"48-month Lookback Strategies selected based on training data from: {str(dates[date_p] - relativedelta(months=48))[:11]} to: {str(dates[date_p])[:11]} are selected" + "\n"
    text = text + f"Phase-2: Weights of the selected strategies are calculated such that rolling sharpe against benchmark is maximised over a lookback of 24 months " + "\n"
    text = text + f"Last Recalibrated TLT Alpha on {str(dates[-2])[:11]}" + "\n"
    # text = text + f"Recalibrating TLT Alpha on {str(dates[-2] + relativedelta(months = recalib_months))[:11]}" + "\n" + "\n"
    text = text + "Alpha Stats for last 252 trading days:" + "\n"
    text = text + f"Sortino: {np.round(backtest_sortino(temp_res[-252:], 0, 0), 2)}" + "\n"
    text = text + f"Sharpe: {np.round(backtest_sharpe(temp_res[-252:], 0, 0), 2)}" + "\n"
    text = text + f"Rolling Sortino: {np.round(backtest_rolling_sortino(temp_res[-252:], 0, 0), 2)}" + "\n"
    text = text + f"Rolling CAGR: {np.round(backtest_rolling_cagr(temp_res[-252:], 0, 0), 2)}" + "\n"
    text = text + f"MaxDrawup/MaxDrawdown: {np.round(backtest_maxdrawup_by_maxdrawdown(temp_res[-252:], 0, 0), 2)}" + "\n" + "\n"
    text = text + f"Signal at : {str(today_time_close)[:19]} : {signal_print(temp_res.iloc[-1]['signal'],ticker)}" + "\n" f"Signal at Yesterday Close: {signal_print(temp_res.iloc[-2]['signal'],ticker)}" + "\n" f"Signal at Day before Yesterday Close: {signal_print(temp_res.iloc[-3]['signal'],ticker)}" + "\n" + "\n" + "Overall Performance:" + "\n" + f"Portfolio Value: {np.round(temp_res.iloc[-1]['Strategy_Return'], 2)}" + "\n" + "\n"
    text = text + "Selected Strategies: " + "\n" + "\n"
    date_recalibrate_tlt = dates[-2] + relativedelta(months=recalib_months)
    return temp_og,str(today_time_close)[:19],dates_all_ss,dates_ss,results_backtest[-1][1],temp_res_tlt, text, date_recalibrate_tlt

if __name__ == '__main__':

    running_for_the_first_time = True

    ticker_nasdaq = "^IXIC"
    number_of_optimization_periods_nasdaq = 2
    recalib_months_nasdaq = 3
    num_strategies_nasdaq = 5
    metric_nasdaq = 'maxdrawup_by_maxdrawdown'

    ticker_tlt = "TLT"
    number_of_optimization_periods_tlt = 2
    recalib_months_tlt = 6
    num_strategies_tlt = 5
    metric_tlt = 'rolling_sharpe'

    run_hour = 23
    run_minute = 1

    while True:
        if  ((datetime.now().hour==run_hour) & (datetime.now().minute==run_minute) & (datetime.now().second==00)) or running_for_the_first_time:
            if datetime.today().isoweekday() < 6:
                recalibrate_today = False
                text = ""
                print(f"Executing: {datetime.now()}")

                text_restricted = "Restricted List is : [PGHH, PFIZER]" + "\n"

                index = ".NDX"
                with open(f'NDX_Constituents.pkl', 'rb') as file:
                    constituents = pickle.load(file)
                constituents_all = []
                for i in range(len(constituents)):
                    constituents_all = constituents_all + constituents.iloc[i][0]
                tickers = list(set(constituents_all))
                tickers = [ticker.partition('.')[0] for ticker in tickers]
                # Restricted Stock
                #tickers.remove('AXBK')

                # removing duplicate tickers
                tickers_remove = ['DISCK', 'FOXA', 'GOOG', 'LBTYK', 'LILAK', 'TFCFA']
                for remove in tickers_remove:
                    tickers.remove(remove)

                recalibrating_months = 1
                top_nassets = 8
                training_period = 24  # 24/48/96
                dates_recalibrating = valid_dates(pd.date_range(start="2007-01-01", end="2024-06-15", freq=f'{recalibrating_months}M'))
                data_inp = prepare_portfolio_data(tickers, recalibrating_months, "datatables", "")

                with open(f'NASDAQ_RecalibPeriod_{int(1)}.pkl', 'rb') as file:
                    assets = pickle.load(file)

                dates_recalibrate = []

                temp_og_nasdaq, signal_time_nasdaq, dates_all_ss_nasdaq, dates_ss_nasdaq, printdf_nasdaq, temp_res_nasdaq, text_nasdaq, date_recalibrate_nasdaq = execute_nasdaq(ticker_nasdaq,number_of_optimization_periods_nasdaq,recalib_months_nasdaq,num_strategies_nasdaq,metric_nasdaq)
                temp_og_tlt, signal_time_tlt, dates_all_ss_tlt, dates_ss_tlt, printdf_tlt, temp_res_tlt, text_tlt, date_recalibrate_tlt = execute_tlt(ticker_tlt,number_of_optimization_periods_tlt,recalib_months_tlt,num_strategies_tlt,metric_tlt)

                dates_recalibrate.append({"Rebalancing Date": "NASDAQ", "Date": date_recalibrate_nasdaq})
                dates_recalibrate.append({"Rebalancing Date": "TLT", "Date": date_recalibrate_tlt})

                if temp_res_nasdaq.iloc[-1]['signal_nasdaq'] == 1:
                    subject_text_add = "[Long on Index]"
                if (temp_res_nasdaq.iloc[-1]['signal_nasdaq'] == 0) & (temp_res_tlt.iloc[-1]['signal_tlt'] == 1):
                    subject_text_add = "[Long on Gold]"
                if (temp_res_nasdaq.iloc[-1]['signal_nasdaq'] == 0) & (temp_res_tlt.iloc[-1]['signal_tlt'] == 0):
                    subject_text_add = "[Long on Cash]"

                data_inp_backtest = pd.concat([data_inp.set_index('Date'), temp_res_nasdaq.set_index('Date'), temp_res_tlt.set_index('Date')], axis=1, join='inner').reset_index()

                rebalancing_months = 12
                dates_rebalancing = []
                assetsb = []

                print(len(assets))

                for date_i in range(len(dates_recalibrating) - (int(training_period / recalibrating_months) + 1)):
                    if ((recalibrating_months * date_i) % rebalancing_months == 2) & (
                            dates_recalibrating[date_i + int(training_period / recalibrating_months)] >=
                            data_inp_backtest["Date"][0]):
                        dates_rebalancing.append(dates_recalibrating[date_i + int(training_period / recalibrating_months)])

                        print(date_i)

                        assetsb.append(assets[date_i].iloc[:top_nassets])
                dates_rebalancing.append(date.today()+timedelta(1))

                data_inp_backtest.loc[(data_inp_backtest['Date'] == '2016-07-21'), 'DXCM'] = 84.63

                results_final, unit_ticker, unit_tlt = backtest_Alpha_AM_NASDAQ(dates_rebalancing, data_inp_backtest, assetsb)

                bench = yf.download("^IXIC", start='2007-01-01', end=str(date.today()+timedelta(1)))
                bench = bench.loc[bench["Close"] > 1]
                bench["Return"] = np.log(bench["Close"] / bench["Close"].shift(1))

                results_final["S_Return"] = pd.DataFrame(np.log(results_final["Pvalue"] / results_final["Pvalue"].shift(1)))
                temp_res = pd.concat([results_final, bench["Return"]], join="inner", axis=1)
                temp_res['Market_Return'] = np.exp(temp_res['Return'].expanding().sum())-1
                temp_res['Strategy_Return'] = np.exp(temp_res['S_Return'].expanding().sum())-1
                temp_res['Portfolio Value'] = temp_res['Pvalue']
                temp_res = temp_res.reset_index().rename(columns={'index': "Date"})
                temp_res.to_csv("NASDAQ_Naive.csv")

                plt.plot(temp_res['Date'], temp_res['Market_Return'], color='black', label='Market Returns')
                plt.plot(temp_res['Date'], temp_res['Strategy_Return'], color='blue', label='Strategy Returns')
                plt.title('Strategy Backtest: Dual Momentum')
                plt.legend(loc=0)
                plt.tight_layout()
                plt.savefig("PerformanceNASDAQ.jpg")
                plt.clf()

                text = text + "*" * 100+ "\n"
                text = text + "DUAL MOMENTUM STRATEGY"+ "\n"
                text = text + f"Recalibrated every {rebalancing_months} months, with training data of {training_period-recalibrating_months} months with a forward return  of {recalibrating_months} month"+ "\n"
                text = text + f"The number of assets being selected are: {top_nassets}" + "\n"
                text = text + f"Last Recalibrated Accelerating Momentum on {str(dates_rebalancing[-2])[:11]}" + "\n"
                # text = text + f"Recalibrating Accelerating Momentum on {str(dates_rebalancing[-2] + relativedelta(months = rebalancing_months))[:11]}" + "\n" + "\n"
                text = text + f"Strategies selected based on training data from: {str(dates_rebalancing[-2]- relativedelta(months = training_period))[:11]} to: {str(dates_rebalancing[-2] - relativedelta(months = recalibrating_months))[:11]} are selected" + "\n"
                text = text + f"Units and Momentum Values were calculated on {str(dates_rebalancing[-2])[:11]}" + "\n"
                text = text + "On Rebalancing Day, if Nasdaq Alpha is long, 100% of the Portolfio are allocated to the stocks below.\nIf Nasdaq Alpha is Neutral and TLT Alpha is Long, 50% of the Portfolio is allocated to TLT and 50% to Fixed Income.\nIf both Alphas are Neutral, 100% of the Portfolio are allocated to Fixed Income" + "\n"+ "\n"
                text = text + "Stats for last 252 trading days:" + "\n"
                text = text + f"Sortino: {np.round(backtest_sortino(temp_res[-252:], 0, 0), 2)}"+ "\n"
                text = text + f"Sharpe: {np.round(backtest_sharpe(temp_res[-252:], 0, 0), 2)}"+ "\n"
                text = text + f"Rolling Sortino: {np.round(backtest_rolling_sortino(temp_res[-252:], 0, 0), 2)}"+ "\n"
                text = text + f"Rolling Sharpe: {np.round(backtest_rolling_sharpe(temp_res[-252:], 0, 0), 2)}"+ "\n"
                text = text + f"Rolling CAGR: {np.round(backtest_rolling_cagr(temp_res[-252:], 0, 0), 2)}"+ "\n"
                text = text + f"Rolling MaxDrawup/MaxDrawdown: {np.round(backtest_maxdrawup_by_maxdrawdown(temp_res[-252:], 0, 0), 2)}"+ "\n"
                text = text + f"Rolling Outperformance: {np.round(backtest_outperformance(temp_res[-252:], 0, 0), 2)}"
                text = text + "\n" + "\n" + "Overall Performance:" + "\n"
                text = text + f" A Portfolio of USD {np.round(temp_res.iloc[0]['Portfolio Value']/1000, 4)}K invested on: {str(dates_rebalancing[0])[:11]} is now valued at: USD {np.round(temp_res.iloc[-1]['Portfolio Value']/1000, 4)}K " + "\n" + "\n"
                text = text + "Selected Assets: " + "\n"
                text = text + f"Number of units of TLT: {unit_tlt}"+ "\n"
                # text = f"Signal at : {str(today_time_close)[:19]} : {signal_print(temp_res.iloc[-1]['signal'])}" + "\n" f"Signal at Yesterday Close: {signal_print(temp_res.iloc[-2]['signal'])}" + "\n" f"Signal at Day before Yesterday Close: {signal_print(temp_res.iloc[-3]['signal'])}" + "\n" + "\n" + "Overall Performance:" + "\n" + f"Portfolio Value: {np.round(temp_res.iloc[-1]['Portfolio Value'], 2)}" + "\n" + "\n" + text

                dates_recalibrate.append({"Rebalancing Date": "Accelerating Momentum",
                                          "Date": dates_rebalancing[-2] + relativedelta(months=rebalancing_months)})

                if running_for_the_first_time == True:
                    pass
                else:
                    SendMail(subject_text_add, pd.DataFrame(dates_recalibrate).set_index("Rebalancing Date").sort_values(by="Date", ascending=True), text_restricted,text_nasdaq, text_tlt, text, printdf_nasdaq, printdf_tlt, pd.concat([assetsb[-1][["Ticker", "Accelerating Momentum"]].set_index("Ticker"), pd.DataFrame([unit_ticker]).transpose().rename(columns={0:"Units"})], axis=1), ["Performance^NSEI.jpg", "PerformanceTLT.jpg", "PerformanceNASDAQ.jpg"], "NASDAQ_Naive.csv")

            # #Recalibrating Nasdaq Alpha
            # if (pd.to_datetime(date.today()) in dates_all_ss_nasdaq):
            #     print(f"Recalibrating Nasdaq at {datetime.now()}")
            #     res_test2 = select_all_strategies(24,dates_ss_nasdaq, temp_og_nasdaq, ticker_nasdaq,save=True)
            #     res_test4 = select_all_strategies(48, dates_ss_nasdaq, temp_og_nasdaq, ticker_nasdaq, save=True)
            #     # res_test8 = select_all_strategies(96, dates_ss, temp_og, ticker, save=True)
            #     ss_test_imp, res_test_imp = select_strategies_from_corr_filter(res_test2,res_test4,0, dates_ss_nasdaq, temp_og_nasdaq, number_of_optimization_periods_nasdaq,10, ticker_nasdaq, save=True)
            #
            #     res_test = []
            #     ss_test = []
            #     dates = []
            #     for date_i in range(len(dates_ss_nasdaq) - (int(24 / 3) + 1)):
            #         if (3 * date_i) % recalib_months_nasdaq == 0:
            #             dates.append(dates_ss_nasdaq[date_i + int(24 / 3)])
            #             ss_test.append(ss_test_imp[date_i])
            #             res_test.append(res_test_imp[date_i])
            #
            #     print(f"Recalibrating Weights: {datetime.now()}")
            #     inputs = []
            #     for date_i in range(len(dates)-1):
            #         inputs.append([date_i, dates, temp_og_nasdaq, ss_test, res_test, num_strategies_nasdaq, metric_nasdaq, recalib_months_nasdaq,dates_ss_nasdaq])
            #     try:
            #         pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
            #         weights_all = pool.map(optimize_weights_live, inputs)
            #     finally: # To make sure processes are closed in the end, even if errors happen
            #         pool.close()
            #         pool.join()
            #
            #     weights_update = [None]*(len(dates)-1)
            #     for date_i in range(len(dates)-1):
            #         weights_update[weights_all[date_i][0]] = weights_all[date_i][1]
            #
            #     with open(f'{ticker_nasdaq}/weights/Results_Ticker{ticker_nasdaq}_LP{number_of_optimization_periods_nasdaq}_Recal{recalib_months_nasdaq}_NS{num_strategies_nasdaq}_M{metric_nasdaq}.pkl','rb') as file:
            #         weights = pickle.load(file)
            #
            #     weights.append(weights_update[-1])
            #
            #     with open(f'{ticker_nasdaq}/weights/Results_Ticker{ticker_nasdaq}_LP{number_of_optimization_periods_nasdaq}_Recal{recalib_months_nasdaq}_NS{num_strategies_nasdaq}_M{metric_nasdaq}.pkl', 'wb') as file:
            #         pickle.dump(weights, file)
            #
            #     print(f"Recalibration for Nasdaq Over at: {datetime.now()}")
            #
            # # Recalibrating TLT Alpha
            # if (pd.to_datetime(date.today()) in dates_all_ss_tlt):
            #     print(f"Recalibrating TLT at {datetime.now()}")
            #     res_test2 = select_all_strategies(24, dates_ss_tlt, temp_og_tlt, ticker_tlt, save=True)
            #     res_test4 = select_all_strategies(48, dates_ss_tlt, temp_og_tlt, ticker_tlt, save=True)
            #     # res_test8 = select_all_strategies(96, dates_ss, temp_og, ticker, save=True)
            #     ss_test_imp, res_test_imp = select_strategies_from_corr_filter(res_test2, res_test4, 0, dates_ss_tlt,
            #                                                                    temp_og_tlt,
            #                                                                    number_of_optimization_periods_tlt,
            #                                                                    10, ticker_tlt, save=True)
            #
            #     res_test = []
            #     ss_test = []
            #     dates = []
            #     for date_i in range(len(dates_ss_tlt) - (int(24 / 3) + 1)):
            #         if (3 * date_i) % recalib_months_tlt == 0:
            #             dates.append(dates_ss_tlt[date_i + int(24 / 3)])
            #             ss_test.append(ss_test_imp[date_i])
            #             res_test.append(res_test_imp[date_i])
            #
            #     print(f"Recalibrating Weights: {datetime.now()}")
            #     inputs = []
            #     for date_i in range(len(dates) - 1):
            #         inputs.append(
            #             [date_i, dates, temp_og_tlt, ss_test, res_test, num_strategies_tlt, metric_tlt,
            #              recalib_months_tlt, dates_ss_tlt])
            #     try:
            #         pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
            #         weights_all = pool.map(optimize_weights_live, inputs)
            #     finally:  # To make sure processes are closed in the end, even if errors happen
            #         pool.close()
            #         pool.join()
            #
            #     weights_update = [None] * (len(dates) - 1)
            #     for date_i in range(len(dates) - 1):
            #         weights_update[weights_all[date_i][0]] = weights_all[date_i][1]
            #
            #     with open(
            #             f'{ticker_tlt}/weights/Results_Ticker{ticker_tlt}_LP{number_of_optimization_periods_tlt}_Recal{recalib_months_tlt}_NS{num_strategies_tlt}_M{metric_tlt}.pkl',
            #             'rb') as file:
            #         weights = pickle.load(file)
            #
            #     weights.append(weights_update[-1])
            #
            #     with open(
            #             f'{ticker_tlt}/weights/Results_Ticker{ticker_tlt}_LP{number_of_optimization_periods_tlt}_Recal{recalib_months_tlt}_NS{num_strategies_tlt}_M{metric_tlt}.pkl',
            #             'wb') as file:
            #         pickle.dump(weights, file)
            #
            #     print(f"Recalibration for TLT Over at: {datetime.now()}")
            #
            # #Recalibrating AM Constituents
            # if (pd.to_datetime(date.today()) in dates_recalibrating):
            #     print(f"Start Recalibrating constituents of NDX at {datetime.now()}")
            #     index = ".NDX"
            #     constituents = get_constituents(index)
            #
            #     with open(f'NDX_Constituents.pkl', 'wb') as file:
            #         pickle.dump(constituents, file)
            #
            #     print(f"End Recalibrating constituents of NDX at {datetime.now()}")
            #
            # #Recalibrating AM
            # if (pd.to_datetime(date.today()) in dates_recalibrating):
            #     print(f"Start Recalibrating AM of NDX at {datetime.now()}")
            #     assets.append(get_weights_stocks_live(constituents, top_nassets, recalibrating_months, training_period,dates_recalibrating, data_inp, save=False)[0])
            #     with open(f'NASDAQ_RecalibPeriod_{int(1)}.pkl', 'wb') as file:
            #         pickle.dump(assets, file)
            #     print(f"End Recalibrating AM of NDX at {datetime.now()}")

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
                next_run = next_run.replace(year=next_run.year + 1).replace(month=1).replace(day=1).replace(
                    hour=run_hour).replace(minute=run_minute).replace(second=00)
            else:
                next_run = next_run.replace(day=1).replace(month=next_run.month + 1).replace(hour=run_hour).replace(
                    minute=run_minute).replace(second=00)

        print(f"Supposed to wake up at: {datetime.now() + timedelta(seconds=(next_run - time_now).seconds - 150)}")
        time.sleep((next_run - time_now).seconds-150)
        print(f"Woken Up: {datetime.now()}")





