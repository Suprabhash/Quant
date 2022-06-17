import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense

from Utils.add_features import add_fisher
from Data.data_retrieval import get_data

np.random.seed(12)

def get_stock_data(symbol, train_size=0.8):
    if symbol == 'sinx':
        df = get_data(".NSEI", 'D')
        df.drop(columns=["Volume"], inplace=True)
        df["Close"] = df["Open"] = df["High"] = df["Low"] = np.sin(df.index / 10 ) +2
    else:
        df = get_data(symbol, 'D')
    df.set_index("Datetime", inplace=True)
    # df["Close_ROC"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    train_len = int(df.shape[0] * train_size)

    if train_len > 0:
        train_df = df.iloc[:train_len, :]
        test_df = df.iloc[train_len:, :]
        return train_df, test_df
    else:
        return df


def create_df(df, windows):
    for window in windows:
        df = add_fisher([df, window])
        df[f'Fisher{window}'].iloc[:window - 1] = np.nan

    df.dropna(inplace=True)
    for window in windows:
        df[f'Normalized_Fisher{window}'] = df[f'Fisher{window}']  # /df.iloc[0,:][f'Fisher{window}']

    df['Normalized_Close'] = df['Close']  # / df.iloc[0, :]['Close']
    # df['Normalized_Close_ROC'] = df["Close_ROC"]
    return df


def get_states(df):
    price_states_value = discretize(df['Normalized_Close'])
    # price_roc_states_value = discretize(df["Normalized_Close_ROC"])
    fisher_states_value = []
    for column in df.columns:
        if column.startswith('Normalized_Fisher'):
            fisher_states_value.append(discretize(df[column]))

    return price_states_value, fisher_states_value  # price_roc_states_value,

def discretize(values, num_states=10):
    states_value = dict()
    step_size = 1./num_states
    for i in range(num_states):
        if i == num_states - 1:
            states_value[i] = values.max()
        else:
            states_value[i] = values.quantile((i+1)*step_size)
    return states_value

def value_to_state(value, states_value):
    if np.isnan(value):
        return np.nan
    else:
        for state, v in states_value.items():
            if value <= v:
                return str(state)
        return str(state)                       #If value in test is more than highest value in train, return max state

def create_state_df(df, price_states_value, fisher_states_value): #price_roc_states_value
    df['Normalized_Close_state'] = df['Normalized_Close'].apply(lambda x : value_to_state(x, price_states_value))
    # df['Normalized_Close_ROC_state'] = df['Normalized_Close_ROC'].apply(lambda x: value_to_state(x, price_roc_states_value))
    fisher_cols = []
    for column in df.columns:
        if column.startswith('Normalized_Fisher'):
            fisher_cols.append(column)
    for i in range(len(fisher_cols)):
        df[f'{fisher_cols[i]}_state'] = df[fisher_cols[i]].apply(lambda x: value_to_state(x, fisher_states_value[i]))

    df['Fisher_state'] = df[[f'{col}_state' for col in fisher_cols]].agg(''.join, axis=1)
    df['state'] = df['Normalized_Close_state'] + df['Fisher_state']  #+ df['Normalized_Close_ROC_state']
    df.dropna(inplace=True)
    return df

def get_all_states(price_states_value,  fisher_states_value):  #, max_num_units   #price_roc_states_value,
    states = []
    # for units in range(max_num_units - 10, max_num_units + 9):
    for p, _ in price_states_value.items():
        # for proc, _ in price_roc_states_value.items():
        fisher_states = []
        for i in range(len(fisher_states_value)):
            fisher_states.append(list(fisher_states_value[i].items()))
        fisher_states = list(itertools.product(*fisher_states))
        for fs in fisher_states:
            fisher_state = ''
            for f in fs:
                c, _ = f
                fisher_state =  fisher_state + str(c)
            states.append(str(p)+fisher_state)  #str(units)+     #+str(proc)
    return states

def initialize_q_mat(all_states, all_actions):
    states_size = len(all_states)
    actions_size = len(all_actions)

    q_mat = np.random.rand(states_size, actions_size) / 1e9
    q_mat = pd.DataFrame(q_mat, columns=all_actions.keys())

    q_mat['states'] = all_states
    q_mat.set_index('states', inplace=True)

    return q_mat


def get_return_since_entry(bought_history, current_adj_close):
    return_since_entry = 0.

    for b in bought_history:
        return_since_entry += (current_adj_close - b)
    return return_since_entry


def visualize_results(actions_history, returns_since_entry):
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    ax1.plot(returns_since_entry,  linewidth=0.2)

    days, prices, actions = [], [], []
    for d, p, a in actions_history:
        days.append(d)
        prices.append(p)
        actions.append(a)

    # ax2.figure(figsize=(20,10))
    ax2.plot(days, prices, label='normalized adj close price',  linewidth=0.2)
    hold_d, hold_p, buy_d, buy_p, sell_d, sell_p = [], [], [], [], [], []
    for d, p, a in actions_history:
        if a == 0:
            hold_d.append(d)
            hold_p.append(p)
        if a == 1:
            buy_d.append(d)
            buy_p.append(p)
        if a == 2:
            sell_d.append(d)
            sell_p.append(p)
        # ax2.annotate(all_actions[a], xy=(d,p), xytext=(d-.2, p+0.001), color=color, arrowprops=dict(arrowstyle='->',connectionstyle='arc3'))
    # ax2.scatter(hold_d, hold_p, color='blue', label='hold', s=2)
    ax2.scatter(buy_d, buy_p, color='green', label='buy', s=10)
    ax2.scatter(sell_d, sell_p, color='red', label='sell', s=10)
    ax2.legend()

def get_invested_capital(actions_history, returns_since_entry):
    invest = []
    total = 0
    return_invest_ratio = None
    for i in range(len(actions_history)):
        a = actions_history[i][2]
        p = actions_history[i][1]
        try:
            next_a = actions_history[i + 1][2]
        except:
            # print('end')
            break
        if a == 1:
            total += p
            # print(total)
            if next_a != 1 or (i == len(actions_history) - 2 and next_a == 1):
                invest.append(total)
                total = 0
    if invest:
        return_invest_ratio = returns_since_entry[-1] / max(invest)
        print('invested capital {}, return/invest ratio {}'.format(max(invest), return_invest_ratio))
    else:
        print('no buy transactions, invalid training')
    return return_invest_ratio


def get_base_return(data):
    start_price, _ = data[0]
    end_price, _ = data[-1]
    return (end_price - start_price) / start_price

def get_sell_reward(returns, theta):
    # percent returns
    # for i, ret in enumerate(returns):
    #     if i == 0:
    #         returns[i] = 0
    #     if ret<0:
    #         returns[i] = theta*ret
    # tot_ret = 1
    # for ret in returns:
    #     tot_ret = tot_ret*(1+ret)
    # tot_ret = tot_ret-1
    # return tot_ret

    # absolute returns
    for i, ret in enumerate(returns):
        if i==0:
            returns[i] = 0
        if ret<0:
            returns[i] = theta*ret
    tot_ret = sum(returns)
    return tot_ret

def act(state, q_mat, threshold=0.2, actions_size=3):
    if np.random.uniform(0, 1) < threshold:  # go random
        action = np.random.randint(low=0, high=actions_size)
    else:
        action = np.argmax(q_mat.loc[state].values)
    return action

def get_model(num_states, actions_size):
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, num_states)))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(actions_size, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

def act_nn(state, model, num_states, threshold=0.2, actions_size=3):
    if np.random.uniform(0, 1) < threshold:  # go random
        action = np.random.randint(low=0, high=actions_size)
    else:
        action = np.argmax(model.predict(np.identity(num_states)[int(state):int(state) + 1]))
    return action

def train_q_learning(train, q, num_states, use_nn, alpha, gamma, episodes, theta=2):   #, max_num_units
    if use_nn:
        model = get_model(num_states, actions_size=3)
    else:
        model = None
    train_df = train.copy()
    train_df["PctReturns"] = train_df["Normalized_Close"]/train_df["Normalized_Close"].shift(1)-1
    train_df["AbsReturns"] = train_df["Normalized_Close"].diff(1)
    train_df.dropna(inplace=True)
    dates = list(train_df.index)
    train_data = np.array(train[1:])
    actions_history = []
    returns_since_entry = [0]
    train_returns = []

    for ii in tqdm(range(episodes)):
        actions_history = []
        num_shares = 0
        bought_history = []
        bought_history_dates = []
        returns_since_entry = [0]

        for i, val in enumerate(train_data):
            current_adj_close, state = val
            # state = str(num_shares) + state
            current_date = dates[i]


            if len(bought_history) > 0:
                returns_since_entry.append(get_return_since_entry(bought_history, current_adj_close))
            else:
                returns_since_entry.append(returns_since_entry[-1])

            # decide action
            if alpha > 0.1:
                alpha = alpha / (i + 1)

            if use_nn:
                action = act_nn(state, model, num_states, threshold=alpha, actions_size=3)
            else:
                action = act(state, q, threshold=alpha, actions_size=3)

            # get reward
            if action == 0:  # hold
                if num_shares > 0:
                    prev_adj_close, _ = train_data[i - 1]
                    past = current_adj_close - prev_adj_close
                    # reward = past/prev_adj_close
                    reward = past
                else:
                    reward = 0

            if action == 1:  # buy
                # Constraining max number of units bought
                # if num_shares<10:
                #     reward = 0
                #     num_shares += 1
                #     bought_history.append((current_adj_close))
                # else:
                #     reward = -100

                # Another unit cannot be bought within 5 days of previous buy
                # try:
                #     last_bought_date = bought_history_dates[-1]
                #     days_since_last_buy = (current_date - last_bought_date).total_seconds() / (3600 * 24)
                #     if days_since_last_buy<5:
                #         reward = -100
                #     else:
                #         reward = 0
                # except:
                #     reward = 0

                #Vanilla reward
                reward = 0
                num_shares += 1

                bought_history.append((current_adj_close))
                bought_history_dates.append((current_date))

            if action == 2:  # sell
                if num_shares > 0:
                    bought_price = bought_history[0]
                    bought_date = bought_history_dates[0]
                    days_held = (current_date - bought_date).total_seconds()/(3600*24)

                    #penalize negative moves
                    # reward = get_sell_reward(list(train_df.loc[bought_date:current_date]["PctReturns"]), theta)
                    reward = get_sell_reward(list(train_df.loc[bought_date:current_date]["AbsReturns"]), theta)

                    #Scaling rewards based on number of days held
                    # reward = reward*days_held/30

                    # Penalizing trades if they are held for more than 60 days
                    # if days_held>60:
                    #     reward = (current_adj_close - bought_price)
                    # else:
                    #     reward = -100

                    bought_history.pop(0)
                    bought_history_dates.pop(0)
                    num_shares -= 1
                else:
                    reward = -100

            try:
                next_adj_close, next_state = train_data[i + 1]
                # if action == 0:
                #     next_state = str(num_shares) + next_state
                # elif action==1:
                #     next_state = str(num_shares+1) + next_state
                # else:
                #     next_state = str(num_shares-1) + next_state
            except:
                break

            actions_history.append((i, current_adj_close, action))

            if use_nn:
                target = reward + gamma *np.max(model.predict(np.identity(num_states)[int(next_state):int(next_state) + 1]))
                target_vector = model.predict(np.identity(num_states)[int(state):int(state) + 1])[0]
                target_vector[action] = target
                model.fit(np.identity(num_states)[int(state):int(state) + 1], target_vector.reshape(-1, 3),epochs=1, verbose=0)
            else:
                # update q table
                # print(f"Action: {action}, reward:{reward}")
                q.loc[state, action] = (1. - alpha) * q.loc[state, action] + alpha * (
                            reward + gamma * (q.loc[next_state].max()))

        train_returns.append(get_invested_capital(actions_history, returns_since_entry))
        # plt.plot(train_returns)
        # plt.show()
        # sns.heatmap(data=q, xticklabels=False, yticklabels=False)
        # plt.show()

    print('End of Training!')
    return q, actions_history, returns_since_entry, model


def eval_q_learning(test_data, q):
    actions_history = []
    num_shares = 0
    returns_since_entry = [0]
    bought_history = []

    for i, val in enumerate(test_data):
        current_adj_close, state = val
        # state = str(num_shares) + state
        # try:
        #     next_adj_close, next_state = test_data[i + 1]
        # except:
        #     print('End of data! Done!')
        #     break

        if len(bought_history) > 0:
            returns_since_entry.append(get_return_since_entry(bought_history, current_adj_close))
        else:
            returns_since_entry.append(returns_since_entry[-1])

        # decide action
        action = act(state, q, threshold=0, actions_size=3)

        if action == 1:  # buy
            num_shares += 1
            bought_history.append((current_adj_close))
        if action == 2:  # sell
            if num_shares > 0:
                bought_price = bought_history[0]
                bought_history.pop(0)
                num_shares -= 1

        actions_history.append((i, current_adj_close, action))

    return actions_history, returns_since_entry

if __name__ == "__main__":

    train_df, test_df = get_stock_data('sinx', 0.8)
    all_actions = {0:'hold', 1:'buy', 2:'sell'}
    windows = [] #20, 40
    # max_num_units = 1
    train_df = create_df(train_df, windows)
    # price_states_value, price_roc_states_value, fisher_states_value = get_states(train_df)
    price_states_value, fisher_states_value = get_states(train_df)
    # train_df = create_state_df(train_df, price_states_value, price_roc_states_value, fisher_states_value)
    # all_states = get_all_states(price_states_value, price_roc_states_value, fisher_states_value)
    train_df = create_state_df(train_df, price_states_value, fisher_states_value)
    all_states = get_all_states(price_states_value, fisher_states_value)  #, max_num_units
    states_size = len(all_states)
    test_df = create_df(test_df, windows)
    test_df = create_state_df(test_df, price_states_value, fisher_states_value)

    q = initialize_q_mat(all_states, all_actions)/1e9
    print('Initializing q')
    print(q[:3])

    train_data = train_df[['Normalized_Close', 'state']]
    q, train_actions_history, train_returns_since_entry, _ = train_q_learning(train_data, q, states_size,  use_nn=False, alpha=0.4, gamma=0.1, episodes=10, theta=1)   #max_num_units,

    sns.heatmap(data=q, xticklabels=False, yticklabels=False)
    plt.show()

    visualize_results(train_actions_history, train_returns_since_entry)
    get_invested_capital(train_actions_history, train_returns_since_entry)
    print('base return/invest ratio {}'.format(get_base_return(np.array(train_data))))

    #%% md

    ## Test evaluation

    #%%

    test_data = np.array(test_df[['Normalized_Close', 'state']])
    test_actions_history, test_returns_since_entry = eval_q_learning(test_data, q)

    #%%

    visualize_results(test_actions_history, test_returns_since_entry)
    get_invested_capital(test_actions_history, test_returns_since_entry)
    # print('invested capital {}, return/invest ratio {}'.format(invested_capital, return_invest_ratio))
    print('base return/invest ratio {}'.format(get_base_return(test_data)))

