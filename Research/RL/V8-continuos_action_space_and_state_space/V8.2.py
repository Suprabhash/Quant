# import pandas as pd
# %load_ext autoreload
# %autoreload 2
import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
# from tqdm import tqdm
# import numpy as np
# from V8Suprabhash import get_stock_data, add_features,  plot_performance
from utils import *
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense

def get_model(num_features, num_actions, num_dense_layers, neurons_per_layer):
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, num_features+num_actions)))
    for i in range(num_dense_layers):
        model.add(Dense(neurons_per_layer, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))#sigmoid didn't work well
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

def train_q_learning(train, state_lookback, model, alpha, epsilon, gamma, episodes, all_actions, metric, features,
                     number_of_random_samples, plot=True):
    train_data = train.copy()
    returns_vs_episodes = []
    best_episode_return = 0
    weights_best_performing = None

    states_and_actions_arr = np.empty(shape=(0, len(features) * state_lookback + len(all_actions)))
    states_arr_built_from_features = train_data[[feature['feature'] for feature in features] + [f"{col}_shift{i}" for col in
                                                                      [feature['feature'] for feature in features] for i
                                                                      in range(1, state_lookback)]].values
    for i in range(len(states_arr_built_from_features)):
        random_actions = np.random.uniform(0, 1, number_of_random_samples) # Try other types of distributions as well
        for j in range(number_of_random_samples):
            states_and_actions_arr = np.vstack((states_and_actions_arr, np.r_[states_arr_built_from_features[i], random_actions[j]]))

    for ii in tqdm(range(episodes)):

        # Backtester initialisation
        initial_capital = 10000
        balance = initial_capital
        net_worth = balance
        actions_history = []
        equity_curve = []
        rewards = []
        states = []
        prices = []
        current_q_all_states = []
        next_q_all_states = []
        number_of_units_after_trade = 0

        q = model.predict(states_and_actions_arr)
        current_qs = []
        next_qs = []
        actions = []

        for i in range(len(states_arr_built_from_features)):
            q_list = []
            next_q_list = []
            for j in range(number_of_random_samples):
                q_list.append(q[i * number_of_random_samples + j])
            if i < len(states_arr_built_from_features) - 1:
                for j in range(number_of_random_samples):
                    next_q_list.append(q[(i + 1) * number_of_random_samples + j])
                next_qs.append(max(next_q_list))
            current_qs.append(max(q_list))
            actions.append(states_and_actions_arr[i * number_of_random_samples + q_list.index(max(q_list))][-1])

        current_qs = np.array(current_qs).reshape(-1, 1)
        next_qs = np.array(next_qs).reshape(-1, 1)
        actions = np.array(actions).reshape(-1, 1)

        for i in range(1,len(train_data)):
            current_adj_close = train_data.iloc[i]["Close"]
            prices.append(current_adj_close)
            states.append(states_arr_built_from_features[i])
            current_q_all_states.append(current_qs[i][0])

            # decide action
            if epsilon > 0.1:
                epsilon = epsilon / 1.2

            if np.random.uniform(0, 1) < epsilon:
                action = np.random.uniform(0,1)
            else:
                action= actions[i][0]

            actions_history.append(action)

            number_of_units_before_trade = number_of_units_after_trade
            position_value_before_trade = number_of_units_before_trade * current_adj_close
            position_value_after_trade = action*net_worth
            number_of_units_after_trade = position_value_after_trade/current_adj_close
            balance = balance - (position_value_after_trade - position_value_before_trade)
            net_worth = balance + position_value_after_trade
            equity_curve.append(net_worth)

            try:
                if metric == "absolute":
                    rewards.append(equity_curve[-1] - equity_curve[-2])
                else:
                    rewards.append((equity_curve[-1] - equity_curve[-2]) / equity_curve[-2])
            except Exception as e:
                rewards.append(0)

            try:
                next_q_all_states.append(next_qs[i][0])
            except:
                break

            if net_worth < 0.1*initial_capital:
                break

        arr_fit_X = np.empty(shape=(0, (len(features) * state_lookback) + 1))
        arr_fit_Y = np.empty(shape=(0, len(all_actions)))
        for state, action, reward, cq, nq in zip(states, actions_history, rewards, current_q_all_states,next_q_all_states):
            target = ((1. - alpha) * cq) + alpha * (reward + gamma * nq)
            arr_fit_X = np.vstack((arr_fit_X, np.r_[state, action]))
            arr_fit_Y = np.vstack((arr_fit_Y, np.array([target]).reshape(-1, 1)))
        model.fit(arr_fit_X, arr_fit_Y, epochs=30, verbose=0)
        episode_return = equity_curve[-1]/equity_curve[0]-1
        print(f"Episode Number: {ii+1}, Total return of episode: {episode_return}")
        if plot:
            plot_performance(train_data, prices, features, actions_history, equity_curve)

        if episode_return>best_episode_return:
            weights_best_performing = model.get_weights()
            best_episode_return = episode_return

        returns_vs_episodes.append(episode_return)

    return model, returns_vs_episodes, weights_best_performing

def eval_q_learning(test_data, model, state_lookback,all_actions,number_of_random_samples, metric, features, save=False):

    states_and_actions_arr = np.empty(shape=(0, len(features) * state_lookback + len(all_actions)))
    states_arr_built_from_features = test_data[[feature['feature'] for feature in features] + [f"{col}_shift{i}" for col in [feature['feature'] for feature in features] for i in range(1, state_lookback)]].values

    for i in range(len(states_arr_built_from_features)):
        random_actions = np.random.uniform(0, 1, number_of_random_samples) # Try other types of distributions as well
        for j in range(number_of_random_samples):
            states_and_actions_arr = np.vstack((states_and_actions_arr, np.r_[states_arr_built_from_features[i], random_actions[j]]))

    #Backtester initialisation
    initial_capital = 10000
    balance = initial_capital
    net_worth = balance
    actions_history = []
    equity_curve = []
    prices = []
    current_q_all_states = []
    number_of_units_after_trade = 0

    q = model.predict(states_and_actions_arr)
    current_qs = []
    next_qs = []
    actions = []

    for i in range(len(states_arr_built_from_features)):
        q_list = []
        next_q_list = []
        for j in range(number_of_random_samples):
            q_list.append(q[i * number_of_random_samples + j])
        if i < len(states_arr_built_from_features) - 1:
            for j in range(number_of_random_samples):
                next_q_list.append(q[(i + 1) * number_of_random_samples + j])
            next_qs.append(max(next_q_list))
        current_qs.append(max(q_list))
        actions.append(states_and_actions_arr[i * number_of_random_samples + q_list.index(max(q_list))][-1])

    for i in range(len(test_data)):
        current_adj_close = test_data.iloc[i]["Close"]
        prices.append(current_adj_close)
        current_q_all_states.append(q[i])

        action = actions[i]
        actions_history.append(action)

        number_of_units_before_trade = number_of_units_after_trade
        position_value_before_trade = number_of_units_before_trade * current_adj_close
        position_value_after_trade = action * net_worth
        number_of_units_after_trade = position_value_after_trade / current_adj_close
        balance = balance - (position_value_after_trade - position_value_before_trade)
        net_worth = balance + position_value_after_trade
        equity_curve.append(net_worth)

    portfolio_return = equity_curve[-1]/equity_curve[0]-1
    return plot_performance(test_data, prices, features, actions_history, equity_curve, save), portfolio_return
    # return 10, portfolio_return

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
        number_of_random_samples = 10
        metric = params["metric"]

        df = get_stock_data(ticker)
        df = add_features(df, features, state_lookback, 0.6)
        train_df = df.iloc[:int(0.6 * len(df)), :]
        val_df = df.iloc[int(0.6 * len(df)):int(0.8 * len(df)), :]
        all_actions = {0: 'percentage_invested_in_equity'}
        model = get_model(len(features) * state_lookback, len(all_actions), num_dense_layers,len(features) * state_lookback * num_dense_layers_by_num_features)
        model, returns_vs_episodes, weights = train_q_learning(train_df, state_lookback, model, alpha, epsilon, gamma,episodes, all_actions, metric, features,number_of_random_samples, plot=False)
        model.set_weights(weights)
        _, score = eval_q_learning(val_df, model, state_lookback,all_actions,number_of_random_samples, metric, features)
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
        run = neptune.init(project='pratiksaxena/V8-RL-ContinuousActionSpace', api_token=API_KEY)
    save = {}
    save_images = {}
    save["ExperimentName"] = f"Run {datetime.now().strftime('%H:%M:%S')}: Experiments with continuous action space"
    features = [
        {"feature": "Close_as_a_feature", "lookback": 0},
        {"feature": "diff_of_close", "lookback": 1},
    ]

    # Experiment params
    ticker = 'sinx'  # 'NIFc1'
    tune = False

    if tune:
        tuned_params = hyperparametric_tuning_optuna(run)
        state_lookback = save["state_lookback"] = int(tuned_params["state_lookback"])
        num_dense_layers = save["num_dense_layers"] = int(tuned_params["num_dense_layers"])
        num_dense_layers_by_num_features = save["num_dense_layers_by_num_features"] = int(tuned_params["num_dense_layers_by_num_features"])
        alpha = save["alpha"] = tuned_params["alpha"]
        epsilon = save["epsilon"] = tuned_params["epsilon"]
        gamma = save["gamma"] = tuned_params["gamma"]
        episodes = save["episodes"] = 100
        number_of_random_samples = save["number_of_random_samples"] = 10
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
        number_of_random_samples = save["number_of_random_samples"]  = 10
        metric = save["metric"] = "percent"
    ############################################################################################################################################

    # Get data
    df = get_stock_data(ticker)
    save["features"] = features

    df = add_features(df, features, state_lookback, 0.8)
    train_df = df.iloc[:int(0.8 * len(df)), :]
    test_df = df.iloc[int(0.8 * len(df)):, :]

    all_actions = {0: 'percentage_invested_in_equity'}

    model = get_model(len(features) * state_lookback, len(all_actions), num_dense_layers,
                      len(features) * state_lookback * num_dense_layers_by_num_features)
    # visualizer(model, format='png', view=True)

    model, returns_vs_episodes, weights = train_q_learning(train_df, state_lookback, model, alpha, epsilon, gamma,
                                                           episodes, all_actions, metric, features,number_of_random_samples, plot=True)

    model.set_weights(weights)
    fig, score = eval_q_learning(test_df, model, state_lookback,all_actions,number_of_random_samples, metric, features, save=True)
    save_images["TestResults"] = fig
    if not (RECORD_EXPERIMENT):
        plt.show()

    fig = plt.figure()
    plt.plot(returns_vs_episodes)
    save_images["TrainResultsvsEpisodes"] = fig
    if not (RECORD_EXPERIMENT):
        plt.show()

    if RECORD_EXPERIMENT:
        run = neptune.init(project='pratiksaxena/V8-RL-ContinuousActionSpace', api_token=API_KEY)
        for key in save_images.keys():
            run[key].upload(save_images[key])
        for key in save.keys():
            run[key] = save[key]
    else:
        pass