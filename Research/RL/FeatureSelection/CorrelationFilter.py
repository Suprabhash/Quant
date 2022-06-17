import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid': False})

import pandas as pd
from Data.data_retrieval import get_data
import Feature_Selection
from tqdm import tqdm
import sys

def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def correlation_filter(features_df, number_of_filtered_selections, ohlcv):
    selected_features = []
    indices = []
    count = 0
    chunksize = 10
    while len(selected_features) < number_of_filtered_selections:
        selected_features_this_run = []
        df = pd.DataFrame()
        for row in range(chunksize):
            row = row + count * chunksize
            f = getattr(Feature_Selection, features_df.iloc[row]["F"])
            flb = features_df.iloc[row]["F_Lookback"]
            g = getattr(Feature_Selection, features_df.iloc[row]["G"])
            glb = features_df.iloc[row]["G_Lookback"]
            X = f([ohlcv, flb]).iloc[flb - 1:].reset_index(drop=True)
            col_name = X.columns[-1]
            X = g(X, col_name, glb).iloc[glb - 1:].reset_index(drop=True)
            X = X[[X.columns[-1]]].set_index(X["Datetime"])
            df = pd.concat([df, X], axis=1)

        df1 = df.copy()
        df.dropna(inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]
        features = [column for column in df]

        if len(indices)>0:
            row = indices[-1]
            # print(features_df.iloc[indices])
            f = getattr(Feature_Selection, features_df.iloc[row]["F"])
            flb = features_df.iloc[row]["F_Lookback"]
            g = getattr(Feature_Selection, features_df.iloc[row]["G"])
            glb = features_df.iloc[row]["G_Lookback"]
            X = f([ohlcv, flb]).iloc[flb - 1:].reset_index(drop=True)
            col_name = X.columns[-1]
            X = g(X, col_name, glb).iloc[glb - 1:].reset_index(drop=True)
            col_name = X.columns[-1]
            X = X[[X.columns[-1]]].set_index(X["Datetime"])

            if col_name not in df.columns:
                df = pd.concat([df, X], axis=1)

        corr_mat = df.corr()

        if len(selected_features) == 0:
            selected_features_this_run.append(features[0])
            selected_features.append(features[0])
            features.remove(features[0])
            last_selected_feature = features[0]
        else:
            last_selected_feature = selected_features[-1]

        while len(selected_features) < number_of_filtered_selections:
            corrs = corr_mat.loc[features][last_selected_feature]
            corrs = corrs.loc[corrs > 0.95]
            features = [f for f in features if f not in corrs.index.to_list()]
            if len(features) == 0:
                break
            feat = features[0]
            selected_features.append(feat)
            selected_features_this_run.append(feat)
            features.remove(feat)
            last_selected_feature = feat

        for feat in selected_features_this_run:
            indices.append(list(df1.columns).index(feat)+chunksize*count)
        count+=1

        update_progress(len(selected_features)/number_of_filtered_selections)

    selected_features = features_df.iloc[indices].reset_index(drop=True)
    return selected_features


df = pd.read_csv("FeatureSelectionResults.csv")
df.drop(columns=['Unnamed: 0'], inplace=True)
ohlcv = get_data('NIFc1', 'D')
number_of_filtered_selections = 100
filtered_strategies = correlation_filter(df, number_of_filtered_selections, ohlcv)
filtered_strategies.to_csv("FilteredFeatureSelectionResults.csv")
print(len(filtered_strategies))




