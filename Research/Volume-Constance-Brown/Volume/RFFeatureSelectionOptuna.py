import warnings

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils
import optuna
from helper_functions_volume import *

API_KEY = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0MWY0OGUyMC1kOTlkLTRjZTItYjc4Ny00MmMyOTI1YTVmODIifQ=="

def objective(trial):
    params = {
        "criterion": "squared_error",  # "gini"
        "n_estimators": int(
            trial.suggest_discrete_uniform("n_estimators", 50, 1500, 50)
        ),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_uniform("min_samples_split", 0.1, 1.0),
        "max_features": trial.suggest_uniform("max_features", 0.1, 1.0),
        "min_samples_leaf": trial.suggest_uniform("min_samples_leaf", 0.1, 0.5),
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "oob_score": True,
        "m": trial.suggest_int("m", 5, 50, 5),
        "n": trial.suggest_int("n", 150,400,25)
    }

    rf_clf = RandomForestRegressor(criterion=params["criterion"], n_estimators=params["n_estimators"],
                                   max_depth=params["max_depth"], min_samples_split=params["min_samples_split"],
                                   max_features=params["max_features"], min_samples_leaf=params["min_samples_leaf"],
                                   n_jobs=params["n_jobs"], random_state=params["random_state"])
    print("fitting...")
    rf_clf.fit(vol_feats_train.to_numpy(),
               vol_metrics_train[f"{params['m']}MaxFReturn_percentile_over_{params['n']}"].to_numpy())
    y_pred = rf_clf.predict(vol_feats_val.to_numpy())
    score = metrics.mean_absolute_error(
        vol_metrics_val[f"{params['m']}MaxFReturn_percentile_over_{params['n']}"].to_numpy(), y_pred)
    print(f"internal score: {score:.4f}")
    return score


if __name__ == '__main__':

    ohlcv_df = get_daily_data(".NSEI")

    with open(f'NSEI_VolumeLevels.pkl', 'rb') as file:
        vol_df = pickle.load(file)

    print("Creating Features")
    m = [i for i in range(5, 55, 5)]
    n = [i for i in range(150,425,25)]
    vol_feats_train, vol_metrics_train, vol_feats_val, vol_metrics_val, vol_feats_test, vol_metrics_test = prepare_volume_features_for_rfregressor(
        ohlcv_df, vol_df, m, n, "2017", "2020")

    vol_feats_train.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'API'], inplace=True)
    vol_feats_val.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'API'], inplace=True)
    vol_feats_test.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'API'], inplace=True)

    print("Features Prepared: Running Experiment")

    run = neptune.init(
        api_token=API_KEY, project='suprabhash/VolumeFeaturesRandomForestRegressor'
    )  # you can pass your credentials here

    neptune_callback = optuna_utils.NeptuneCallback(run)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, callbacks=[neptune_callback])

    # Log Optuna charts and study object after the sweep is complete
    optuna_utils.log_study_metadata(study, run, log_plot_contour=False)

    # Stop logging
    run.stop()

    print(study.best_trials[0].values)
    print(study.best_trials[0].params)