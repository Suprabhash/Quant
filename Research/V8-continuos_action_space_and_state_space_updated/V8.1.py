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
from utils import *


def custom_activation(x):
    return K.sigmoid(x)

def get_model(num_features, num_actions, num_dense_layers, neurons_per_layer):
    model = Sequential()
    get_custom_objects().update({'custom_activation': Activation(custom_activation)})
    model.add(InputLayer(batch_input_shape=(1, num_features)))
    for i in range(num_dense_layers):
        model.add(Dense(neurons_per_layer))
        model.add(Activation(custom_activation, name=f'SpecialActivation_{i}'))
    model.add(Dense(num_actions))
    model.add(Activation(custom_activation, name=f'SpecialActivation_{i + 1}'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

def train_q_learning(train, state_lookback, model, alpha, epsilon, gamma, episodes, all_outcomes, metric, features, plot=True):

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
        outcome_history = []
        equity_curve = []
        rewards = []
        states = []
        prices = []
        current_q_all_states = []
        next_q_all_states = []
        previous_outcome = 0
        q = model.predict(arr)
        outcomes = (-1*q).argsort()

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
                outcome_priority = np.arange(0, len(all_outcomes))
                np.random.shuffle(outcome_priority)
            else:
                outcome_priority = outcomes[i]

            outcome = outcome_priority[0]
            outcome_history.append(outcome)


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
        arr_fit_Y = np.empty(shape=(0,len(all_outcomes)))
        for state, action, reward, cq, nq in zip(states, outcome_history, rewards, current_q_all_states, next_q_all_states):
            target = ((1. - alpha) * cq[action]) + alpha * (reward + gamma * np.max(nq))
            cq[action] = target
            arr_fit_X = np.vstack((arr_fit_X,state))
            arr_fit_Y = np.vstack((arr_fit_Y,cq.reshape(-1, len(all_outcomes))))

        model.fit(arr_fit_X,arr_fit_Y,epochs=30, verbose=0)
        episode_return = equity_curve[-1]/equity_curve[0]-1
        print(f"Episode Number: {ii+1}, Total return of episode: {episode_return}")
        if plot:
            plot_performance(train_data, prices, features, outcome_history, equity_curve)

        if episode_return>best_episode_return:
            weights_best_performing = model.get_weights()
            best_episode_return = episode_return

        returns_vs_episodes.append(episode_return)

    return model, returns_vs_episodes, weights_best_performing

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
        model, returns_vs_episodes, weights = train_q_learning(train_df, state_lookback, model, alpha, epsilon, gamma,
                                                               episodes, all_actions, metric, features, plot=False)
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

    all_outcomes = {0: 'neutral', 1: '25% long',2: '50% long',3: '75% long',4: '100% long'}
    model = get_model(len(features)*state_lookback, len(all_outcomes), num_dense_layers, len(features)*state_lookback*num_dense_layers_by_num_features)
    # visualizer(model, format='png', view=True)

    model, returns_vs_episodes, weights = train_q_learning(train_df, state_lookback, model, alpha, epsilon, gamma,
                                                           episodes, all_outcomes, metric, features, plot=True)
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
