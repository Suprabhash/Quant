import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import math
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import pickle
import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
import neptune.new as neptune
from datetime import datetime
from keras_visualizer import visualizer
import scipy.stats as stats
import neptune.new.integrations.optuna as optuna_utils
import optuna
import importlib.util
from azure.data.tables import TableServiceClient, UpdateMode
from azure.core.credentials import AzureNamedKeyCredential
import os
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
os.environ["PATH"] += os.pathsep + 'C:/Users/suprabhashsahu/Desktop/StrategyResearch/venv/Graphviz/bin/'

# from Utils.add_features import add_fisher,
# from Data.data_retrieval import get_data
# from Utils.neptune_ai_api_key import API_KEY

np.random.seed(12)
spec = importlib.util.spec_from_file_location("account_name_and_key", "Z:\\Algo\\keys_and_passwords\\Azure\\account_name_and_key.py")
azure_keys = importlib.util.module_from_spec(spec)
spec.loader.exec_module(azure_keys)
_STORAGE_ACCOUNT_NAME = azure_keys._STORAGE_ACCOUNT_NAME
_STORAGE_ACCOUNT_KEY = azure_keys._STORAGE_ACCOUNT_KEY
_TABLE_SERVIVCE_ENDPOINT = azure_keys._TABLE_SERVIVCE_ENDPOINT

spec = importlib.util.spec_from_file_location("email_and_password", "Z:\\Algo\\keys_and_passwords\\Gmail\\email_and_password.py")
email_and_password = importlib.util.module_from_spec(spec)
spec.loader.exec_module(email_and_password)
sender_email = email_and_password.sender_email
sender_password = email_and_password.sender_password
receiver_email1 = email_and_password.receiver_email1
receiver_email2 = email_and_password.receiver_email2
receiver_email3 = email_and_password.receiver_email3

spec = importlib.util.spec_from_file_location("current_nifty_tickers", "Z:\\Algo\\data_retrieval\\Tickers\\current_nifty_tickers.py")
current_nifty_tickers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(current_nifty_tickers)

current_nifty_tickers = current_nifty_tickers.current_nifty_tickers

API_KEY = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0Yzg5MGZmYy1iMzVlLTQ1YTItODFiNS1hMTE2MTc1Mzc3ODUifQ=="

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

def add_fisher(input):
    temp = input[0].copy()
    lookback = input[1]
    if f'Fisher{lookback}' not in temp.columns:
        temp[f'Fisher{lookback}'] = fisher(temp, lookback)
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

    return fish1


def RSI(data_df,period):
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
    return rsi_df.fillna(0).iloc[:,0]

def add_constance_brown(input):
    temp = input[0].copy()
    if f'ConstanceBrown' not in temp.columns:
        r = RSI(temp, 14)
        rsi_mom_length = 9
        ma_length = 3
        rsi_ma_length = 3
        rsidelta = [0] * len(temp)
        for i in range(len(temp)):
            if i < rsi_mom_length:
                rsidelta[i] = np.nan
            else:
                rsidelta[i] = r[i] - r[i - rsi_mom_length]
        rsisigma = RSI(temp, rsi_ma_length).rolling(window=ma_length).mean()
        rsidelta = [0 if math.isnan(x) else x for x in rsidelta]
        s = [0] * len(temp)
        for i in range(len(rsidelta)):
            s[i] = rsidelta[i] + rsisigma[i]
        # s = [0 if math.isnan(x) else x for x in s]

        temp["ConstanceBrown"] = s
    return temp
def time_consolidator(df, period):
    aggregate = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Up": "sum",
        "Down": "sum",
        "Volume": "sum",
    }
    return df.resample(f"{period}Min").agg(aggregate).dropna()

def get_stock_data(symbol):
    if symbol == 'sinx':
        df = get_data(".NSEI", 'D')
        df.drop(columns=["Volume"], inplace=True)
        df["Close"] = df["Open"] = df["High"] = df["Low"] = np.sin(df.index / 10 ) +2
    elif symbol == 'SPY':
        df = pd.read_csv('SPY.txt')
        df["Datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"], infer_datetime_format=True
        )
        df["Volume"] = df["Up"] + df["Down"]
        df = time_consolidator(df.set_index("Datetime"), 60).reset_index()
    else:
        df = get_data(symbol, 'D')
    df.set_index("Datetime", inplace=True)
    df.dropna(inplace=True)
    return df

def f_discretize(values, num_states=10):
    states_value = dict()
    step_size = 1./num_states
    for i in range(num_states):
        if i == num_states - 1:
            states_value[i] = values.max()
        else:
            states_value[i] = values.quantile((i+1)*step_size)
    return states_value

def value_to_state(value, states_value):
    if np.isnan(value):
        return np.nan
    else:
        for state, v in states_value.items():
            if value <= v:
                return state
        return state

def add_features(df, features, state_lookback, train_percent):
    lookbacks = []
    for feature, lookback in [(feature["feature"], feature["lookback"]) for feature in features]:
        lookbacks.append(lookback)

        if feature.startswith("Fisher"):
            df[feature] = add_fisher([df, lookback])[[f"Fisher{lookback}"]]

        if feature.startswith("Close_as_a_feature"):
            df[feature] = df["Close"]

        if feature.startswith("diff_of_close"):
            df[feature] = df["Close"].diff()

        if feature.startswith("Momentum"):
            def aqr_momentum(array):
                returns = np.diff(np.log(array))
                x = np.arange(len(returns))
                slope, _, rvalue, _, _ = stats.linregress(x, returns)
                return ((1 + slope) ** 252) * (rvalue ** 2)
            df[feature] = df["Close"].rolling(lookback).apply(aqr_momentum)

        if feature.startswith("IBS"):
            df[feature] = (df.Close - df.Low) / (df.High - df.Low)

        if feature.startswith("CB"):
            df[feature] = add_constance_brown([df])[["ConstanceBrown"]]

        if feature.startswith("Volume"):
            with open(f'NIFc1_non_absolute_percentage_of_poc_vah_val_polv_and_ohlc_avg_across_various_lookbacks_for_all_days.pkl','rb') as file:
                vol_feats = pickle.load(file)/100
            if feature.startswith("VolumePOC"):
                df = pd.concat([df, vol_feats[f"{lookback}_day_lookback_poc"]], axis=1)
                df = df.rename(columns={f"{lookback}_day_lookback_poc": feature})
            if feature.startswith("VolumeVAL"):
                df = pd.concat([df, vol_feats[f"{lookback}_day_lookback_val"]], axis=1)
                df = df.rename(columns={f"{lookback}_day_lookback_val": feature})
            if feature.startswith("VolumeVAH"):
                df = pd.concat([df, vol_feats[f"{lookback}_day_lookback_vah"]], axis=1)
                df = df.rename(columns={f"{lookback}_day_lookback_vah": feature})
            if feature.startswith("VolumePOLV"):
                df = pd.concat([df, vol_feats[f"{lookback}_day_lookback_polv"]], axis=1)
                df = df.rename(columns={f"{lookback}_day_lookback_polv": feature})

        if feature.startswith("AvgDeviation"):
            with open(f'percentage_deviation_avg.pkl','rb') as file:
                avg_deviation = pickle.load(file)
            df = pd.concat([df,avg_deviation["AvgDeviation"]],axis=1)

        for i in range(1, state_lookback):
            df[f"{feature}_shift{i}"] = df[feature].shift(1)

    df = df.iloc[max(lookbacks)+state_lookback:]
    df.dropna(inplace=True)
    for feature in [feature['feature'] for feature in features] + [f"{col}_shift{i}" for col in [feature['feature'] for feature in features] for i in range(1, state_lookback)]:
        states = f_discretize(df[feature].iloc[:int(df.shape[0] * train_percent)], 100)
        df[feature] = df[feature].apply(lambda x: value_to_state(x, states))/100
    return df

def custom_activation(x):
    return (K.sigmoid(x) * 2) - 1

def get_model(num_features, num_actions, num_dense_layers, neurons_per_layer):
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, num_features)))
    for i in range(num_dense_layers):
        model.add(Dense(neurons_per_layer, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

def plot_performance(df, prices, features, actions_history, equity_curve, save=False):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 7))
    # fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.plot(prices, label='Close')
    ax1_copy = ax1.twinx()
    ax1_copy.plot(actions_history, label='Actions')
    ax2.plot(actions_history, label='Actions')
    ax2_copy = ax2.twinx()
    for feature in [feature["feature"] for feature in features]:
        ax2_copy.plot(df[feature].values[:len(actions_history)], label=feature, color='green', ls='dotted')
    ax2_copy.axhline(0.0, ls='--', color='grey')
    ax3.plot(equity_curve, label='Net worth')
    ax3.plot([price*10000 / prices[0] for price in prices], label='Benchmark')
    ax1.legend()
    # ax2.legend()
    ax3.legend()
    if type(save)==str:
        plt.savefig(save)
    else:
        plt.show()
    return fig

def train_q_learning(train, state_lookback, model, alpha, epsilon, gamma, episodes, all_actions, metric, features, plot=True):

    train_data = train.copy()
    returns_vs_episodes = []
    best_episode_return = 0
    weights_best_performing = None

    arr = train_data[[feature['feature'] for feature in features] + [f"{col}_shift{i}" for col in [feature['feature'] for feature in features] for i in range(1, state_lookback)]].values

    for ii in tqdm(range(episodes)):

        #Backtester initialisation
        balance = 10000
        net_worth = balance
        in_position = False
        number_of_units_in_position = 0
        position_value = 0.0
        actions_history = []
        equity_curve = []
        rewards = []
        states = []
        prices = []
        current_q_all_states = []
        next_q_all_states = []

        q = model.predict(arr)
        actions = (-1*q).argsort()

        for i in range(1,len(train_data)):
            current_adj_close = train_data.iloc[i]["Close"]
            last_day_adj_close = train_data.iloc[i - 1]["Close"]
            prices.append(current_adj_close)
            states.append(arr[i])
            current_q_all_states.append(q[i])

            # decide action
            if epsilon > 0.1:
                epsilon = epsilon / 1.2

            if np.random.uniform(0, 1) < epsilon:
                action_priority = np.arange(0,len(all_actions))
                np.random.shuffle(action_priority)
            else:
                action_priority = actions[i]

            action = action_priority[0]
            actions_history.append(action)

            if not in_position:
                if action == 1:  # OPEN LONG
                    in_position = True
                    number_of_units_in_position = balance/current_adj_close
                    balance = balance - (number_of_units_in_position*current_adj_close)
                    position_value = number_of_units_in_position*current_adj_close
                    net_worth = balance + position_value
                    equity_curve.append(net_worth)
                    rewards.append(0)
                else:
                    net_worth = balance + position_value
                    equity_curve.append(net_worth)
                    rewards.append(0)
            else:
                if action == 1:  # HOLD LONG
                    position_value = number_of_units_in_position*current_adj_close
                    net_worth = balance + position_value
                    equity_curve.append(net_worth)
                    try:
                        if metric == "absolute":
                            rewards.append(equity_curve[-1] - equity_curve[-2])
                        else:
                            rewards.append((equity_curve[-1] - equity_curve[-2]) / equity_curve[-2])
                    except:
                        rewards.append(0)
                else:  # CLOSE LONG
                    balance = balance + (number_of_units_in_position*current_adj_close)
                    in_position = False
                    position_value = 0.0
                    number_of_units_in_position = 0
                    net_worth = balance + position_value
                    equity_curve.append(net_worth)
                    rewards.append(0)

            try:
                next_q_all_states.append(q[i+1])
            except:
                break

        arr_fit_X = np.empty(shape=(0,len(features)*state_lookback))
        arr_fit_Y = np.empty(shape=(0,len(all_actions)))
        for state, action, reward, cq, nq in zip(states, actions_history, rewards, current_q_all_states, next_q_all_states):
            target = ((1. - alpha) * cq[action]) + alpha * (reward + gamma * np.max(nq))
            cq[action] = target
            arr_fit_X = np.vstack((arr_fit_X,state))
            arr_fit_Y = np.vstack((arr_fit_Y,cq.reshape(-1, len(all_actions))))

        model.fit(arr_fit_X,arr_fit_Y,epochs=30, verbose=0)
        episode_return = equity_curve[-1]/equity_curve[0]-1
        print(f"Episode Number: {ii+1}, Total return of episode: {episode_return}")
        if plot:
            plot_performance(train_data, prices, features, actions_history, equity_curve)

        if episode_return>best_episode_return:
            weights_best_performing = model.get_weights()
            best_episode_return = episode_return

        returns_vs_episodes.append(episode_return)

    return model, returns_vs_episodes, weights_best_performing

def eval_q_learning(test_data, model, state_lookback, metric, features, save=False):

    arr = test_data[[feature['feature'] for feature in features] + [f"{col}_shift{i}" for col in [feature['feature'] for feature in features] for i in range(1, state_lookback)]].values

    #Backtester initialisation
    balance = 10000
    in_position = False
    position_value = 0.0
    price_bought = 0.0
    bet_bought = 0.0
    actions_history = []
    equity_curve = []
    rewards = []
    states = []
    prices = []
    current_q_all_states = []

    q = model.predict(arr)
    actions = (-1*q).argsort()

    for i in range(len(test_data)):
        current_adj_close = test_data.iloc[i]["Close"]
        prices.append(current_adj_close)
        states.append(arr[i])
        current_q_all_states.append(q[i])
        action_priority = actions[i]

        action = action_priority[0]
        actions_history.append(action)

        if not in_position:
            if action == 1:  # OPEN LONG
                in_position = True
                price_bought = current_adj_close
                bet_bought = balance
                balance -= bet_bought
                position_value = bet_bought
                rewards.append(0)
            else:  # KEEP LOOKING
                rewards.append(0)
        else:
            market_return = ((current_adj_close - price_bought) / price_bought)
            if action == 1:  # HOLD LONG
                position_value = bet_bought * (1.0 + market_return)
                if metric=="absolute":
                    rewards.append(bet_bought*market_return)
                else:
                    rewards.append(market_return)
            else:  # CLOSE LONG
                balance += bet_bought * (1.0 + market_return)
                in_position = False
                price_bought = 0.0
                bet_bought = 0.0
                position_value = 0.0
                rewards.append(0)

        net_worth = balance + position_value
        equity_curve.append(net_worth)
    portfolio_return = equity_curve[-1]/equity_curve[0]-1
    return plot_performance(test_data, prices, features, actions_history, equity_curve, save), portfolio_return

def hyperparametric_tuning_optuna(run):
    def objective(trial):
        params = {
            "alpha": trial.suggest_uniform("alpha", 0, 1),
            "epsilon": trial.suggest_uniform("epsilon", 0, 1),
            "gamma": trial.suggest_uniform("gamma", 0, 1),
            "metric": trial.suggest_categorical("metric", ["percent", "absolute"]),
            "num_dense_layers": trial.suggest_discrete_uniform("num_dense_layers", 1,5,1),
            "num_dense_layers_by_num_features": trial.suggest_discrete_uniform("num_dense_layers_by_num_features", 1,5,1),
            "state_lookback": trial.suggest_discrete_uniform("state_lookback", 1,10,1),
        }

        state_lookback = int(params["state_lookback"])
        num_dense_layers = int(params["num_dense_layers"])
        num_dense_layers_by_num_features = int(params["num_dense_layers_by_num_features"])
        alpha = params["alpha"]
        epsilon = params["epsilon"]
        gamma = params["gamma"]
        episodes = 100
        metric = params["metric"]

        df = get_stock_data(ticker)
        df = add_features(df, features, state_lookback, 0.6)
        train_df = df.iloc[:int(0.6 * len(df)), :]
        val_df = df.iloc[int(0.6 * len(df)):int(0.8 * len(df)), :]
        all_actions = {0: 'neutral', 1: 'long'}
        model = get_model(len(features) * state_lookback, len(all_actions), num_dense_layers,len(features) * state_lookback * num_dense_layers_by_num_features)
        model, returns_vs_episodes, weights = train_q_learning(train_df, state_lookback, model, alpha, epsilon, gamma, episodes,all_actions, metric, features, plot=False)
        model.set_weights(weights)
        _, score = eval_q_learning(val_df, model, state_lookback, metric, features)
        return score


    neptune_callback = optuna_utils.NeptuneCallback(run)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, callbacks=[neptune_callback])

    # Log Optuna charts and study object after the sweep is complete
    optuna_utils.log_study_metadata(study, run, log_plot_contour=False)

    # Stop logging
    run.stop()

    print(study.best_trials[0].values)
    print(study.best_trials[0].params)

    return study.best_trials[0].params


if __name__ == "__main__":

    RECORD_EXPERIMENT = True
    if RECORD_EXPERIMENT:
        run = neptune.init(project="pratiksaxena/V5-RL-MLP-WithHistoricalStates", api_token=API_KEY)
    save = {}
    save_images = {}
    save["ExperimentName"] = f"Run {datetime.now().strftime('%H:%M:%S')}: Experiments with Price and AvgDeviation of POC levels: .NIFc1"

    ############################################################################################################################################
    # Experiment Features
    features = [
        {"feature": "Close_as_a_feature", "lookback": 0},
        # {"feature": "IBS", "lookback": 0},
        # {"feature": "Momentum", "lookback": 5},
        # {"feature": "Momentum", "lookback": 10},
        # {"feature": "Momentum", "lookback": 20},
        # {"feature": "VolumePOC10", "lookback": 10},
        # {"feature": "VolumePOC21", "lookback": 21},
        # {"feature": "VolumePOC63", "lookback": 63},
        # {"feature": "VolumePOC126", "lookback": 126},
        # {"feature": "VolumePOC252", "lookback": 252},
        # {"feature": "VolumeVAH10", "lookback": 10},
        # {"feature": "VolumeVAH21", "lookback": 21},
        # {"feature": "VolumeVAH63", "lookback": 63},
        # {"feature": "VolumeVAH126", "lookback": 126},
        # {"feature": "VolumeVAH252", "lookback": 252},
        # {"feature": "VolumeVAL10", "lookback": 10},
        # {"feature": "VolumeVAL21", "lookback": 21},
        # {"feature": "VolumeVAL63", "lookback": 63},
        # {"feature": "VolumeVAL126", "lookback": 126},
        # {"feature": "VolumeVAL252", "lookback": 252},
        # {"feature": "VolumePOLV10", "lookback": 10},
        # {"feature": "VolumePOLV21", "lookback": 21},
        # {"feature": "VolumePOLV63", "lookback": 63},
        # {"feature": "VolumePOLV126", "lookback": 126},
        # {"feature": "VolumePOLV252", "lookback": 252},
        # {"feature": "Volume378", "lookback": 378},
        # {"feature": "Fisher50", "lookback": 50},
        # {"feature": "Fisher150", "lookback": 150},
        # {"feature": "Fisher300", "lookback": 300},
        # {"feature": "AvgDeviation", "lookback": 1},
        {"feature": "diff_of_close", "lookback": 1},
    ]

    #Experiment params
    ticker = save["ticker"] = 'sinx'#'NIFc1'
    tune = False
    if tune:
        tuned_params = hyperparametric_tuning_optuna(run)
        state_lookback = save["state_lookback"] = int(tuned_params["state_lookback"])
        num_dense_layers = save["num_dense_layers"] = int(tuned_params["num_dense_layers"])
        num_dense_layers_by_num_features = save["num_dense_layers_by_num_features"] = int(tuned_params["num_dense_layers_by_num_features"])
        alpha = save["alpha"] = tuned_params["alpha"]
        epsilon = save["epsilon"] = tuned_params["epsilon"]
        gamma = save["gamma"] = tuned_params["gamma"]
        episodes = save["episodes"] = 2
        metric = save["metric"] = tuned_params["metric"]
    else:
        train_percent = 0.8
        state_lookback = save["state_lookback"] = 1
        num_dense_layers = save["num_dense_layers"] = 2
        num_dense_layers_by_num_features = save["num_dense_layers_by_num_features"] = 2
        alpha = save["alpha"] = 0.5
        epsilon = save["epsilon"] = 0.1
        gamma = save["gamma"] = 0.1
        episodes = save["episodes"] = 100
        metric = save["metric"] = "percent"

    ############################################################################################################################################

    #Get data
    df = get_stock_data(ticker)
    save["features"] = features

    df = add_features(df, features, state_lookback, 0.8)
    train_df = df.iloc[:int(0.8 * len(df)), :]
    test_df = df.iloc[int(0.8 * len(df)):, :]

    all_actions = {0: 'neutral', 1: 'long',}
    model = get_model(len(features)*state_lookback, len(all_actions), num_dense_layers, len(features)*state_lookback*num_dense_layers_by_num_features)
    # visualizer(model, format='png', view=True)

    model, returns_vs_episodes, weights = train_q_learning(train_df, state_lookback, model, alpha, epsilon, gamma, episodes, all_actions, metric, features,plot=True)
    model.set_weights(weights)
    fig, score = eval_q_learning(test_df, model, state_lookback, metric, features,save=True)
    save_images["TestResults"] = fig
    if not (RECORD_EXPERIMENT):
        plt.show()

    fig = plt.figure()
    plt.plot(returns_vs_episodes)
    save_images["TrainResultsvsEpisodes"] = fig
    if not (RECORD_EXPERIMENT):
        plt.show()

    if RECORD_EXPERIMENT:
        run = neptune.init(project="pratiksaxena/V5-RL-MLP-WithHistoricalStates", api_token=API_KEY)
        for key in save_images.keys():
            run[key].upload(save_images[key])
        for key in save.keys():
            run[key] = save[key]
    else:
        pass