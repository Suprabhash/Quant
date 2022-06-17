import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

from tqdm import tqdm
from functools import partial
from Utils.add_features import add_fisher, add_inverse_fisher, add_constance_brown
from Utils.add_features import max_over_lookback, min_over_lookback, sma, x, shift
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from Optimisers.Optimiser import Optimiser
from Data.data_retrieval import get_data


if __name__ == "__main__":

    data = get_data('.NSEI', 'D')
    lookforwards = [3,5,10,21,45]

    for i in range(1,max(lookforwards)):
        data[f"FReturn{i}"] = data["Close"].shift(-i)/data["Close"]-1
    for lookforward in lookforwards:
        data[f"MaxFReturn{lookforward}"] = data[[f"FReturn{i}" for i in range(1,lookforward)]].max( axis=1)
        data[f"MinFReturn{lookforward}"] = data[[f"FReturn{i}" for i in range(1,lookforward)]].min( axis=1)
    data = data.drop(columns=[f"FReturn{i}" for i in range(1,max(lookforwards)) if i not in lookforwards])
    data.dropna(inplace=True)

    max_lookback = 400
    f_functions = [add_fisher, add_inverse_fisher, add_constance_brown]
    g_functions = [x, max_over_lookback, min_over_lookback, sma, shift]

    def prior(params):
        if (params[0] < 0) | (params[0] > len(f_functions)):
            return 0
        if (params[1] < 5) | (params[1] > max_lookback):
            return 0
        if (params[2] < 0) | (params[2] > len(g_functions)):
            return 0
        if (params[3] < 5) | (params[3] > max_lookback):
            return 0
        return 1

    def alpha(args):
        df = data.copy()
        f_function = f_functions[int(round(args[0]))]
        f_lookback = int(round(args[1]))
        g_function = g_functions[int(round(args[2]))]
        g_lookback = int(round(args[3]))

        df = f_function([df, f_lookback])
        f = df.columns[-1]
        df[f] = MinMaxScaler().fit_transform(df[[f]])
        df = df.iloc[max_lookback:].reset_index(drop=True)
        df = g_function(df, f, g_lookback)
        g = df.columns[-1]
        df = df.iloc[max_lookback:].reset_index(drop=True)
        if np.isnan(np.sum(df[g].to_numpy())):
            print(args)
        return df[g].to_numpy()

    res_all = pd.DataFrame(columns=["F_function", "F_lookback", "G_function", "G_lookback", "Return", "NMIS"])
    for target in tqdm([column for column in data.columns if "FReturn" in column]):
        if np.isnan(np.sum(data[target].iloc[2*max_lookback:].to_numpy())):
            print(target)
        guess_list = [[np.random.randint(0,len(f_functions)), np.random.randint(0,max_lookback+1),
                       np.random.randint(0,len(g_functions)), np.random.randint(0,max_lookback+1)] for i in range(10)]

        iters = 100
        params = ["F_function", "F_lookback", "G_function", "G_lookback"]
        guess_length = len(params)
        res = pd.DataFrame(columns=params+["NMIS"])
        for guess in guess_list:
            opt = Optimiser(method="MCMC")
            opt.define_alpha_function(alpha)
            opt.define_prior(prior)
            opt.define_guess(guess=guess)
            opt.define_iterations(iters)
            opt.define_optim_function(None)
            opt.define_target(data[target].iloc[2*max_lookback:].to_numpy())
            opt.define_lower_and_upper_limit(0, max_lookback+1)
            mc, rs = opt.optimise()
            res_iter = []
            for i in range(iters):
                d = {}
                for j in range(guess_length):
                    key = params[j]
                    val = mc.analyse_results(rs, top_n=iters)[0][i][j]
                    if key=="F_function":
                        d[key] = f_functions[int(round(val))].__name__
                    elif key=="G_function":
                        d[key] = g_functions[int(round(val))].__name__
                    else:
                        d[key] = int(round(val))
                    d.update({'NMIS': mc.analyse_results(rs, top_n=iters)[1][i]})
                res_iter.append(d)
            res_iter = pd.DataFrame(res_iter)
            res = pd.concat([res, res_iter], axis=0)
        res["Return"] = target
        res_all = pd.concat([res_all, res], axis=0)
    res_all = res_all.sort_values(by="NMIS", ascending=False)
    res_all = res_all.reset_index(drop=True)
    res_all.to_csv("Results_f(g(x)).csv")


