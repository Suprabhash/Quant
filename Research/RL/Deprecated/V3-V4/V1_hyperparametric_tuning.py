from V1 import *
import neptune.new.integrations.optuna as optuna_utils
import optuna

def objective(trial):
    params = {
        "state_lookback": trial.suggest_int("state_lookback", 1, 2, 1),
        "UseFisherinStateSpace": trial.suggest_categorical("UseFisherinStateSpace", [True, False]),
        "use_nn": trial.suggest_categorical("use_nn", [False]),
        "alpha": trial.suggest_uniform("alpha", 0, 1),
        "epsilon": trial.suggest_uniform("epsilon", 0, 1),
        "gamma": trial.suggest_uniform("gamma", 0, 1),
        "episodes": 10,
        "max_units": 1,
        "metric": trial.suggest_categorical("metric", ["percent", "absolute"]),
        "theta": trial.suggest_uniform("theta", 1, 2),
        "buy_reward": trial.suggest_uniform("buy_reward", 0.1, 5)
    }

    # Create Data
    train_df, test_df = get_stock_data('sinx', 0.8)

    # Define Actions
    all_actions = {0: 'hold', 1: 'buy', 2: 'sell'}

    # Fisher windows
    windows = [20]  # 20, 40

    # Create Features for RL and get info about states
    train_df = create_df(train_df, windows)
    price_states_value, fisher_states_value = get_states(train_df)
    train_df = create_state_df(train_df, price_states_value, fisher_states_value, params["UseFisherinStateSpace"])
    all_states = get_all_states(price_states_value, fisher_states_value, params["state_lookback"], params["UseFisherinStateSpace"])
    states_size = len(all_states)
    test_df = create_df(test_df, windows)
    test_df = create_state_df(test_df, price_states_value, fisher_states_value, params["UseFisherinStateSpace"])

    # Initialise qmatrix with random vals
    q = initialize_q_mat(all_states, all_actions) / 1e9

    # Train
    train_data = train_df[['Normalized_Close', 'state', 'Fisher_state']]

    q, train_actions_history, train_returns_since_entry, _, train_results_over_episodes, train_equity_curve = train_q_learning(
        train_data, params["state_lookback"], q, states_size, use_nn=params["use_nn"], alpha=params["alpha"], epsilon=params["epsilon"], gamma=params["gamma"],
        episodes=params["episodes"], buy_reward=params["buy_reward"], max_units=params["max_units"], metric=params["metric"], theta=params["theta"])

    score = get_invested_capital(train_actions_history, train_returns_since_entry)
    return score

if __name__ == '__main__':

    run = neptune.init(
        api_token=API_KEY, project='suprabhash/QLearning'
    )

    neptune_callback = optuna_utils.NeptuneCallback(run)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000, callbacks=[neptune_callback])

    # Log Optuna charts and study object after the sweep is complete
    optuna_utils.log_study_metadata(study, run, log_plot_contour=False)

    # Stop logging
    run.stop()

    print(study.best_trials[0].values)
    print(study.best_trials[0].params)

