import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
from V5 import *
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/suprabhashsahu/Desktop/StrategyResearch/venv/Graphviz/bin/'

if __name__ == "__main__":

    features_list = [
        # {"feature": "Close", "lookback": 0},
        {"feature": "IBS", "lookback": 0},
        {"feature": "CB", "lookback": 3},
        {"feature": "Momentum5", "lookback": 5},
        {"feature": "Momentum10", "lookback": 10},
        {"feature": "Momentum20", "lookback": 20},
        {"feature": "Momentum50", "lookback": 50},
        {"feature": "Momentum100", "lookback": 100},
        {"feature": "Momentum150", "lookback": 150},
        {"feature": "Momentum200", "lookback": 200},
        {"feature": "Momentum300", "lookback": 300},
        {"feature": "VolumePOC10", "lookback": 10},
        {"feature": "VolumePOC21", "lookback": 21},
        {"feature": "VolumePOC63", "lookback": 63},
        {"feature": "VolumePOC126", "lookback": 126},
        {"feature": "VolumePOC252", "lookback": 252},
        {"feature": "VolumeVAH10", "lookback": 10},
        {"feature": "VolumeVAH21", "lookback": 21},
        {"feature": "VolumeVAH63", "lookback": 63},
        {"feature": "VolumeVAH126", "lookback": 126},
        {"feature": "VolumeVAH252", "lookback": 252},
        {"feature": "VolumeVAL10", "lookback": 10},
        {"feature": "VolumeVAL21", "lookback": 21},
        {"feature": "VolumeVAL63", "lookback": 63},
        {"feature": "VolumeVAL126", "lookback": 126},
        {"feature": "VolumeVAL252", "lookback": 252},
        {"feature": "VolumePOLV10", "lookback": 10},
        {"feature": "VolumePOLV21", "lookback": 21},
        {"feature": "VolumePOLV63", "lookback": 63},
        {"feature": "VolumePOLV126", "lookback": 126},
        {"feature": "VolumePOLV252", "lookback": 252},
        {"feature": "Fisher5", "lookback": 5},
        {"feature": "Fisher10", "lookback": 10},
        {"feature": "Fisher50", "lookback": 50},
        {"feature": "Fisher100", "lookback": 100},
        {"feature": "Fisher150", "lookback": 150},
        {"feature": "Fisher300", "lookback": 300},
    ]

    #Experiment params
    ticker = 'NIFc1'
    train_percent = 0.8
    state_lookback = 1
    num_dense_layers = 2
    num_dense_layers_by_num_features = 2
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.2
    episodes = 100
    metric = "percent"
    results = []

    for features in features_list:
        features = [features]
        #Get data
        df = get_stock_data(ticker)
        df = add_features(df, features, state_lookback, 0.8)
        train_df = df.iloc[:int(0.8 * len(df)), :]
        test_df = df.iloc[int(0.8 * len(df)):, :]
        all_actions = {0: 'neutral', 1: 'long'}
        model = get_model(len(features)*state_lookback, len(all_actions), num_dense_layers, len(features)*state_lookback*num_dense_layers_by_num_features)
        model, returns_vs_episodes, weights = train_q_learning(train_df, state_lookback, model, alpha, epsilon, gamma, episodes, all_actions, metric, features, plot=False)
        model.set_weights(weights)
        fig, score = eval_q_learning(test_df, model, state_lookback, metric, features, save=f"FeatureAnalysis/{features[0]['feature']}.png")
        results.append({"Feature": features[0]["feature"], "BestTrainReturn": max(returns_vs_episodes), "TestReturn": score})
    results = pd.DataFrame(results)
    results.sort_values(by="TestReturn", ascending=False)
    results.to_csv("FeatureAnalysis/FeatureAnalysis.csv")



