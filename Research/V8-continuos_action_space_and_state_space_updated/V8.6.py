# import pandas as pd
# %load_ext autoreload
# %autoreload 2
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
# from tqdm import tqdm
# import numpy as np
# from V8Suprabhash import get_stock_data, add_features,  plot_performance
from utils import *
from config import *
from keras.models import load_model
from keras import backend as K
import Feature_Selection

def get_model(num_features, num_actions, num_dense_layers, neurons_per_layer):
    from keras.models import Sequential
    from keras.layers import InputLayer
    from keras.layers import Dense

    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, num_features+num_actions)))
    for i in range(num_dense_layers):
        model.add(Dense(neurons_per_layer, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))#sigmoid didn't work well
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

def multiprocessing_wrapper_for_train_q_learning(ticker,state_lookback, alpha, epsilon, gamma, episodes, all_actions, metric, features,
                     number_of_random_samples,num_dense_layers,num_dense_layers_by_num_features, plot=True, train=None):
    inputs = []
    directory_path = os.getcwd()
    for i in range(number_of_nodes_for_multiprocessing):
        inputs.append([int(i),directory_path,ticker, state_lookback, alpha, epsilon, gamma, episodes, all_actions, metric, features,number_of_random_samples,num_dense_layers,num_dense_layers_by_num_features, plot ,train])

    outputs = []
    with open(f"output_params_{ticker}.pkl","wb") as file:
        pickle.dump(outputs,file)

    pool = ProcessingPool(nodes=number_of_nodes_for_multiprocessing)
    res = pool.map(train_q_learning, inputs)
    pool.clear()
    print("Done multiprocessing")

    best_returns = 0
    best_model_path = ""
    for i in range(number_of_nodes_for_multiprocessing):
        with open(f'{directory_path}\\selected_models\\returns_model_{ticker}_{int(i)}.pkl', 'rb') as file:
            best_returns_for_current_process = pickle.load(file)
        if best_returns_for_current_process>best_returns:
            best_model_path = f'{directory_path}\\selected_models\\model_{ticker}_{int(i)}'
            best_returns = best_returns_for_current_process

    print(f"Best returns across multiple starting points = {best_returns}")

    return best_model_path

def train_q_learning(args):
    [run_id,directory_path,ticker, state_lookback, alpha, epsilon, gamma, episodes, all_actions, metric, features,number_of_random_samples,num_dense_layers,num_dense_layers_by_num_features, plot ,train] = args

    model = get_model(len(features) * (state_lookback+1), len(all_actions), num_dense_layers,
                      len(features) * (state_lookback+1) * num_dense_layers_by_num_features)
    print("entered multiprocessing")
    if train == None:
        with open(f'train_data_{ticker}.pkl', 'rb') as file:
            train_data = pickle.load(file)
    else:
        train_data = train.copy()

    returns_vs_episodes = []
    best_episode_return = 0
    weights_best_performing = None

    states_and_actions_arr = np.empty(shape=(0, len(features) * (state_lookback+1) + len(all_actions)))
    states_arr_built_from_features = train_data[[feature['feature'] for feature in features] + [f"{col}_shift{i+1}" for col in
                                                                      [feature['feature'] for feature in features] for i
                                                                      in range(state_lookback)]].values
    for i in tqdm(range(len(states_arr_built_from_features))):
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
        equity_curve_dates = []
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
            action_to_take = states_and_actions_arr[i * number_of_random_samples + q_list.index(max(q_list))][-1]
            if action_to_take > 0.80:
                action_to_take = 1
            elif action_to_take <= 0.80 and action_to_take > 0.60:
                action_to_take = 0.75
            elif action_to_take <= 0.60 and action_to_take > 0.40:
                action_to_take = 0.50
            elif action_to_take <= 0.40 and action_to_take > 0.20:
                action_to_take = 0.25
            elif action_to_take <= 0.20 and action_to_take >= 0:
                action_to_take = 0
            # if action_to_take > 0.8:
            #     action_to_take = 1
            # else:
            #     action_to_take = 0
            actions.append(action_to_take)

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
            position_value_after_trade = action * net_worth
            number_of_units_after_trade = position_value_after_trade / current_adj_close
            balance = balance - (position_value_after_trade - position_value_before_trade)
            net_worth = balance + position_value_after_trade
            equity_curve.append(net_worth)
            equity_curve_dates.append(train_data.index[i])

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

        arr_fit_X = np.empty(shape=(0, (len(features) * (state_lookback+1)) + 1))
        arr_fit_Y = np.empty(shape=(0, len(all_actions)))
        for state, action, reward, cq, nq in zip(states, actions_history, rewards, current_q_all_states,next_q_all_states):
            target = ((1. - alpha) * cq) + alpha * (reward + gamma * nq)
            arr_fit_X = np.vstack((arr_fit_X, np.r_[state, action]))
            arr_fit_Y = np.vstack((arr_fit_Y, np.array([target]).reshape(-1, 1)))
        # K.set_session(K.tf.compat.v1.Session(config=K.tf.compat.v1.ConfigProto(device_count={"CPU": 1})))
        model.fit(arr_fit_X, arr_fit_Y, epochs=30, verbose=0)
        episode_return = equity_curve[-1]/equity_curve[0]-1
        print(f"Episode Number: {ii+1}, Total return of episode: {episode_return}")
        if plot and episode_return>5:
            plot_performance(train_data, prices, features, actions_history, equity_curve,equity_curve_dates=equity_curve_dates)

        if episode_return>best_episode_return:
            best_episode_return = episode_return
            model.save(f"{directory_path}\\selected_models\\model_{ticker}_{run_id}")
            with open(f'{directory_path}\\selected_models\\returns_model_{ticker}_{run_id}.pkl', 'wb') as file:
                pickle.dump(best_episode_return, file)

        returns_vs_episodes.append(episode_return)
    return

def eval_q_learning(test_data, model, state_lookback,all_actions,number_of_random_samples, metric, features, save=False, plot_rl_and_alpha_performance = False):

    states_and_actions_arr = np.empty(shape=(0, len(features) * (state_lookback+1) + len(all_actions)))
    states_arr_built_from_features = test_data[[feature['feature'] for feature in features] + [f"{col}_shift{i+1}" for col in [feature['feature'] for feature in features] for i in range(state_lookback)]].values

    for i in tqdm(range(len(states_arr_built_from_features))):
        random_actions = np.random.uniform(0, 1, number_of_random_samples)  # Try other types of distributions as well
        for j in range(number_of_random_samples):
            states_and_actions_arr = np.vstack(
                (states_and_actions_arr, np.r_[states_arr_built_from_features[i], random_actions[j]]))

    start_prediction_time = datetime.now()
    q = model.predict(states_and_actions_arr)
    end_prediction_time = datetime.now()
    print(f"Time taken for prediction = {end_prediction_time-start_prediction_time}")

    current_qs = []
    actions = []

    for i in range(len(states_arr_built_from_features)):
        q_list = []
        for j in range(number_of_random_samples):
            q_list.append(q[i * number_of_random_samples + j])
        current_qs.append(max(q_list))
        action_to_take = states_and_actions_arr[i * number_of_random_samples + q_list.index(max(q_list))][-1]
        if action_to_take > 0.80:
            action_to_take = 1
        elif action_to_take <= 0.80 and action_to_take > 0.60:
            action_to_take = 0.75
        elif action_to_take <= 0.60 and action_to_take > 0.40:
            action_to_take = 0.50
        elif action_to_take <= 0.40 and action_to_take > 0.20:
            action_to_take = 0.25
        elif action_to_take <= 0.20 and action_to_take >= 0:
            action_to_take = 0
        # if action_to_take > 0.8:
        #     action_to_take = 1
        # else:
        #     action_to_take = 0

        actions.append(action_to_take)

    #Backtester initialisation
    initial_capital = 10000
    balance = initial_capital
    net_worth = balance
    actions_history = []
    equity_curve = []
    prices = []
    equity_curve_dates = []
    number_of_units_after_trade = 0

    for i in range(len(test_data)):
        current_adj_close = test_data.iloc[i]["Close"]
        prices.append(current_adj_close)

        action = actions[i]
        actions_history.append(action)

        number_of_units_before_trade = number_of_units_after_trade
        position_value_before_trade = number_of_units_before_trade * current_adj_close
        position_value_after_trade = action * net_worth
        number_of_units_after_trade = position_value_after_trade / current_adj_close
        balance = balance - (position_value_after_trade - position_value_before_trade)
        net_worth = balance + position_value_after_trade
        equity_curve.append(net_worth)
        equity_curve_dates.append(test_data.index[i])

    portfolio_return = equity_curve[-1]/equity_curve[0]-1
    print(f"Testing returns = {portfolio_return}")

    if plot_rl_and_alpha_performance == True:
        plot_rl_performance_against_alpha_performance(test_data, prices, features, actions_history, equity_curve,
                         equity_curve_dates=equity_curve_dates)

    return plot_performance(test_data, prices, features, actions_history, equity_curve, save,equity_curve_dates=equity_curve_dates), portfolio_return

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
        best_model_path = multiprocessing_wrapper_for_train_q_learning(ticker, state_lookback, alpha, epsilon, gamma,episodes, all_actions, metric,features, number_of_random_samples,num_dense_layers,
                                                                       num_dense_layers_by_num_features)
        print(f"Selected model path = {best_model_path}")
        best_model = load_model(f"{best_model_path}")
        _, score = eval_q_learning(test_df, best_model, state_lookback, all_actions, number_of_random_samples, metric,features, save=True)
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
    RECORD_EXPERIMENT = False
    if RECORD_EXPERIMENT:
        run = neptune.init(project='pratiksaxena/V8-RL-ContinuousActionSpace', api_token=API_KEY)
    save = {}
    save_images = {}
    save["ExperimentName"] = f"Run {datetime.now().strftime('%H:%M:%S')}: Experiments with continuous action space"
    features = [
        {"feature": "Close_as_a_feature", "lookback": 0},
        {"feature": "Open_as_a_feature", "lookback": 0},
        {"feature": "High_as_a_feature", "lookback": 0},
        {"feature": "Low_as_a_feature", "lookback": 0},
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
        # {"feature": "PE_ZS_5", "lookback": 5},
        # {"feature": "PE_ZS_10", "lookback": 10},
        # {"feature": "PE_ZS_15", "lookback": 15},
        # {"feature": "PE_ZS_30", "lookback": 30},
        # {"feature": "PE_ZS_45", "lookback": 45},
        # {"feature": "PE_ZS_60", "lookback": 60},
        # {"feature": "PE_ZS_90", "lookback": 90},
        # {"feature": "PE_ZS_100", "lookback": 100},
        # {"feature": "PE_ZS_150", "lookback": 150},
        # {"feature": "PE_ZS_252", "lookback": 252},
        # {"feature": "Fisher1", "lookback": 1},
        # {"feature": "Fisher2", "lookback": 2},
        {"feature": "Fisher5", "lookback": 5},
        {"feature": "Fisher10", "lookback": 10},
        {"feature": "Fisher30", "lookback": 30},
        {"feature": "Fisher50", "lookback": 60},
        {"feature": "Fisher90", "lookback": 90},
        {"feature": "Fisher120", "lookback": 120},
        {"feature": "Fisher150", "lookback": 150},
        {"feature": "Fisher250", "lookback": 250},
        {"feature": "Fisher300", "lookback": 300},
        # {"feature": "Increasing_pivots_number_of_pivots", "lookback": 1},
        # {"feature": "Increasing_pivots_pivot_type", "lookback": 1},
        # {"feature": "CB50", "lookback": 50},
        # {"feature": "CB150", "lookback": 150},
        # {"feature": "CB300", "lookback": 300},
        # {"feature": "diff_of_close", "lookback": 1},
    ]

    add_uncorrelated_technical_indicators = False

    if add_uncorrelated_technical_indicators == True:
        uncorrelated_technical_indicators_df = pd.read_csv("C:\\Users\\pratiksaxena\\Desktop\\Pratik\\AcsysAlgo_github\\RL\\SuprabhashKT\\RL\\FeatureSelection\\FilteredFeatureSelectionResults.csv")
        uncorrelated_technical_indicators_df.drop(columns=['Unnamed: 0'], inplace=True)
        for row in tqdm(range(len(uncorrelated_technical_indicators_df))):
            g_name = uncorrelated_technical_indicators_df.iloc[row]["G"]
            f_name = uncorrelated_technical_indicators_df.iloc[row]["F"]
            if f"technical_indicator_{g_name}_{f_name}" not in ["technical_indicator_MININDEX_add_TrueRange","technical_indicator_FLOOR_add_TEMA","technical_indicator_MININDEX_add_constance_brown",
                                                            "technical_indicator_MAXINDEX_add_constance_brown"]:
                features.append({"feature":f"technical_indicator_{g_name}_{f_name}","lookback":[uncorrelated_technical_indicators_df.iloc[row]["G_Lookback"],uncorrelated_technical_indicators_df.iloc[row]["F_Lookback"]]})

    # Experiment params
    ticker = 'NIFc1' #'sinx'
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
        episodes = save["episodes"] = 2000
        number_of_random_samples = save["number_of_random_samples"]  = 10
        metric = save["metric"] = "percent"
    ############################################################################################################################################

    # Get data
    df = get_stock_data(ticker)
    save["features"] = features

    df = add_features(df, features, state_lookback, 0.8)
    with open(f"data_with_features_{ticker}.pkl", "wb") as file:
        pickle.dump(df,file)

    # with open(f"data_with_features_{ticker}.pkl", "rb") as file:
    #     df = pickle.load(file)

    train_df = df.iloc[:int(0.8 * len(df)), :]
    test_df = df.iloc[int(0.8 * len(df)):, :]

    with open(f"train_data_{ticker}.pkl","wb") as file:
        pickle.dump(train_df,file)

    all_actions = {0: 'percentage_invested_in_equity'}


    # visualizer(model, format='png', view=True)
    multiprocessing_across_starting_points = True
    directory_path = os.getcwd()

    if multiprocessing_across_starting_points == True:
        best_model_path = multiprocessing_wrapper_for_train_q_learning(ticker,state_lookback,alpha,epsilon,gamma,episodes,all_actions,metric,
                                                                    features,number_of_random_samples,num_dense_layers,num_dense_layers_by_num_features,plot=True)
        print(f"Selected model path = {best_model_path}")

    else:
        plot = True
        train = None
        inputs = [int(0), directory_path, ticker, state_lookback, alpha, epsilon, gamma, episodes, all_actions, metric, features,number_of_random_samples, num_dense_layers, num_dense_layers_by_num_features, plot, train]
        train_q_learning(inputs)
        best_model_path = f'{directory_path}\\selected_models\\model_{ticker}_{int(0)}'

    # best_model_path = f'{directory_path}\\selected_models\\model_{ticker}_{int(1)}'
    print(f"Best model path is {best_model_path}")
    best_model = load_model(f"{best_model_path}")

    number_of_random_samples_for_testing = save["number_of_random_samples_for_testing"] = number_of_random_samples

    print("The best training results - ")

    # fig, score = eval_q_learning(train_df, best_model, state_lookback, all_actions, number_of_random_samples_for_testing,
    #                              metric, features, save=True)
    # fig, score = eval_q_learning(test_df, best_model, state_lookback,all_actions,number_of_random_samples_for_testing, metric, features, save=True,plot_rl_and_alpha_performance=True)
    fig, score = eval_q_learning(df, best_model, state_lookback, all_actions, number_of_random_samples_for_testing,
                                 metric, features, save=True,plot_rl_and_alpha_performance=True)
    save_images["TestResults"] = fig
    if not (RECORD_EXPERIMENT):
        plt.show()

    fig = plt.figure()
    # plt.plot(returns_vs_episodes)
    save_images["TrainResultsVsEpisodes"] = fig
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