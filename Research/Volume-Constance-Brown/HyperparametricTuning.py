import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils
import optuna


def get_Xy(df, y_col):
    X = df.drop(columns=[y_col])
    X = X[np.isfinite(X).all(1)]
    y = (df[y_col].to_frame().reindex(X.index).sort_index().squeeze()).dropna()
    X = X.loc[y.index]
    return X, y

data_inp = get_data("^NSEI", "yfinance")
data = data_inp.copy().drop(columns=["FReturn","Volume","CB_TypeCurrentPivot","CB_PivotValue","CB_TypePreviousPivot","SMACB_TypeCurrentPivot","SMACB_PivotValue","SMACB_TypePreviousPivot","FMACB_TypeCurrentPivot","FMACB_PivotValue","FMACB_TypePreviousPivot","CB_PreviousPivotValue","SMACB_PreviousPivotValue","FMACB_PreviousPivotValue"])
data = data.iloc[100:-1]
data.set_index("Date", inplace=True)
data.columns

df_train = data.loc[:"2016"]
df_validate = data.loc["2017":"2019"]
df_test = data.loc["2020":]

X_train, y_train = get_Xy(df_train, "BinaryOutcome")
X_validate, y_validate = get_Xy(df_validate, "BinaryOutcome")
agg_res = {}

neptune.init('suprabhash/ConstanceBrownClassifier', api_token=API_KEY)
neptune.create_experiment(
    "RandomForestForward1", upload_source_files=["*.py"]
)
neptune_callback = optuna_utils.NeptuneCallback(log_study=True, log_charts=True)

def objective(trial):
    params = {
        "criterion": ["gini", "entropy"],
        "n_estimators": int(
            trial.suggest_discrete_uniform("n_estimators", 50, 1500, 50)
        ),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_samples_split": trial.suggest_uniform("min_samples_split", 0.1, 1.0),
        "min_samples_leaf": trial.suggest_uniform("min_samples_leaf", 0.1, 1.0),
        "max_features": trial.suggest_uniform("max_features", 0.1, 1.0),
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "oob_score": True,
    }

#Points per leaf   #Number of features per split (sqrt(num of features))

    rf_clf = RandomForestClassifier(**params)
    print("fitting...")
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_validate)
    score = metrics.matthews_corrcoef(y_validate, y_pred)
    print(f"internal score: {score:.4f}")
    return score


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25, callbacks=[neptune_callback])
    optuna_utils.log_study(study)