from V3_Neptune import *
import neptune.new.integrations.optuna as optuna_utils
import optuna

def objective(trial):
    params = {
        "alpha": trial.suggest_uniform("alpha", 0, 1),
        "epsilon": trial.suggest_uniform("epsilon", 0, 1),
        "gamma": trial.suggest_uniform("gamma", 0, 1),
        "metric": trial.suggest_categorical("metric", ["percent", "absolute"]),
        "layers": trial.suggest_categorical("layers", [1,2,3]),
        "neurons_per_layer": trial.suggest_categorical("neurons_per_layer", [400,500,600]),
    }

    # Create Data
    df = get_stock_data('.NSEI')
    train_len = int(df.shape[0] * 0.8)

    features = [
        {"feature": "Fisher100", "lookback": 150, "discretize": 20},
        {"feature": "Fisher300", "lookback": 300, "discretize": 20},
    ]
    df, all_states = add_features(df, features)

    df["state"] = df[[f"{feature['feature']}_state" for feature in features]].agg(''.join, axis=1)
    states = [list(state['states']) for state in all_states]
    all_states = list(itertools.product(*states))
    all_states = [''.join(tuple([str(state) for state in all_state])) for all_state in all_states]

    all_states_dict = {}
    for i, state in enumerate(all_states):
        all_states_dict[state] = i

    df.replace({"state": all_states_dict}, inplace=True)

    train_df = df.iloc[:train_len, :]
    test_df = df.iloc[train_len:, :]

    all_actions = {0: 'neutral', 1: 'long'}
    model = get_model(len(all_states), len(all_actions), params["layers"], params["neurons_per_layer"])

    episodes = 100
    state_lookback = 1
    model, returns_vs_episodes = train_q_learning(train_df, state_lookback, model, params["alpha"], params["epsilon"], params["gamma"], episodes,
                                                  all_states, all_actions, params["metric"])
    fig, score = eval_q_learning(test_df, model, all_states, state_lookback, params["metric"])
    return score

if __name__ == '__main__':

    run = neptune.init(
        api_token=API_KEY, project='suprabhash/RL-MLP'
    )

    neptune_callback = optuna_utils.NeptuneCallback(run)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, callbacks=[neptune_callback])

    # Log Optuna charts and study object after the sweep is complete
    optuna_utils.log_study_metadata(study, run, log_plot_contour=False)

    # Stop logging
    run.stop()

    print(study.best_trials[0].values)
    print(study.best_trials[0].params)