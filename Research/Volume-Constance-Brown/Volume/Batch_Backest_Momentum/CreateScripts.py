import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from datetime import date,timedelta
import pickle
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import itertools
import multiprocessing
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import numpy as np
import os
import zipfile
import os
import shutil


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

if __name__ == '__main__':

    ticker = '.NSEI'

    print(f"Processing {ticker}")
    with open(f'NSEI_Volume_Momentum_Backtest_Azure_temp_og.pkl','rb') as file:
        temp_og = pickle.load(file)

    # temp_og = pd.read_csv("https://raw.githubusercontent.com/AcsysAlgo/VolumeData/main/NSEI_Volume_Momentum_Backtest_Azure_temp_og.csv")
    # temp_og.drop(columns=["Unnamed: 0"], inplace=True)
    # temp_og["Date"] = pd.to_datetime(temp_og["Date"])

    dates_all_ss = pd.date_range(start=str(temp_og.iloc[1]['Date'] + timedelta(days=503))[:10], end="2024-06-15", freq=f'3M')
    dates = valid_dates(dates_all_ss)

    TEST = False

    dir = 'InputFiles'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    # shutil.copyfile("NSEI_Volume_Momentum_Backtest_Azure_temp_og.pkl", "InputFiles/NSEI_Volume_Momentum_Backtest_Azure_temp_og.pkl")


    for train_months in [24,48,96]:
        for date_i in range(len(dates) - (int(train_months / 3) + 1)):

            if not TEST:
                with open('NSEI_Momentum_Backtest_Azure_Template.py', "rt") as fin:
                    with open(f"InputFiles/{ticker}_TrainYrs_{int(train_months / 12)}_All_Strategies_{date_i}.py", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace("ticker_inp", f"'{ticker}'").replace("date_i_inp", f"{date_i}").replace("train_months_inp", f"{train_months}"))

            else:
                with open('NSEI_Momentum_Backtest_Azure_Tester.py', "rt") as fin:
                    with open(f"InputFiles/{ticker}_TrainYrs_{int(train_months / 12)}_All_Strategies_{date_i}.py", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace("ticker_inp", f"'{ticker}'").replace("date_i_inp", f"{date_i}").replace("train_months_inp", f"{train_months}"))


            # def zipdir(path, ziph):
            #     length = len(path)
            #
            #     # ziph is zipfile handle
            #     for root, dirs, files in os.walk(path):
            #         folder = root[length:]  # path without "parent"
            #         for file in files:
            #             if (file==f"{ticker}_TrainYrs_{int(train_months / 12)}_All_Strategies_{date_i}.py") | (file=="NSEI_Volume_Momentum_Backtest_Azure_temp_og.pkl"):
            #                 ziph.write(os.path.join(root, file), os.path.join(folder, file))
            #
            # zipf = zipfile.ZipFile(f'InputFiles/{ticker}_TrainYrs_{int(train_months / 12)}_All_Strategies_{date_i}.zip', 'w', zipfile.ZIP_DEFLATED)
            # zipdir('InputFiles', zipf)
            # zipf.close()