import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from datetime import datetime
import gym
from gym import spaces
from Data.data_retrieval import get_data
from Utils.add_features import add_fisher
import stable_baselines3
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import os
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import neptune.new as neptune
from Utils.neptune_ai_api_key import API_KEY
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3.common.callbacks import EvalCallback


class StockTradingEnv(gym.Env):

    def __init__(self, df, features, init_account_balance, window_shape):
        # initialize environment
        super(StockTradingEnv, self).__init__()
        # raw dataset
        self.df = df
        # how much $ do we have?
        self.init_account_balance = init_account_balance
        self.balance = self.init_account_balance
        self.net_worth = self.init_account_balance

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.window_shape = window_shape
        self.action_space = spaces.Discrete(2)
        self.current_step = self.window_shape-1
        self.signal_features = features
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.window_shape, self.signal_features.shape[-1]), dtype=np.float16)

        # share costs
        self.in_position = False
        self.position_value = 0.0
        self.price_bought = 0.0
        self.bet_bought = 0.0

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = self.init_account_balance
        self.net_worth = self.init_account_balance
        self.current_step = self.window_shape-1
        self.in_position = False
        self.position_value = 0.0
        self.price_bought = 0.0
        self.bet_bought = 0.0
        return self.get_observation()

    def get_observation(self):
        market_state = self.signal_features.iloc[self.current_step - self.window_shape+1:self.current_step+1]
        # print(f"Market state: {market_state}")
        return market_state

    def step(self, action):
        done = False
        if self.current_step==(len(self.df)):
            self.current_step=self.window_shape-1
            done=True
        self.take_action(action)
        reward = self.net_worth
        obs = self.get_observation()
        # print(done)
        self.current_step += 1
        return obs, reward, done, {}

    def take_action(self, action):

        # Set the current price to a random price within the time step
        current_price = self.df.iloc[self.current_step]["Close"]
        self.current_price = current_price
        self.datetime = self.df.index[self.current_step]


        # print("*"*100)
        # print(f"Current step: {self.current_step}")
        # print(f"df.iloc[self.current_step]: {df.iloc[self.current_step]}")
        # print(f"Datetime: {self.datetime}")


        if not self.in_position:
            if action == 1:  # OPEN LONG
                self.in_position = True
                self.price_bought = current_price
                self.bet_bought = self.balance
                self.balance -= self.bet_bought
                self.position_value = self.bet_bought
            else:  # KEEP LOOKING
                pass
        else:
            market_return = ((current_price - self.price_bought) / self.price_bought)
            if action == 1:  # HOLD LONG
                self.position_value = self.bet_bought * (1.0 + market_return)
            else:  # CLOSE LONG
                self.balance += self.bet_bought * (1.0 + market_return)
                self.in_position = False
                self.price_bought = 0.0
                self.bet_bought = 0.0
                self.position_value = 0.0

        self.net_worth = self.balance + self.position_value

    def render(self, mode='human'):
        return {
            'step': self.current_step,
            'price': self.current_price,
            'balance': self.balance,
            'position': self.position_value,
            'net_worth': self.net_worth,
            'profit': self.net_worth - self.init_account_balance,
            'datetime': self.datetime
        }

def evaluate_agent(env, df, model):
    obs = env.reset()
    history = {
        'balance': [],
        'action': [],
        'position': [],
        'net_worth': [],
        'price': [],
        'Datetime': []
    }

    for i in range(len(df) - HISTORICAL_STATES+1):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        state = env.render()
        history['action'].append(action[0])
        history['balance'].append(state['balance'])
        history['net_worth'].append(state['net_worth'])
        history['position'].append(state['position'])
        history['price'].append(state['price'])
        history['Datetime'].append(state['datetime'])
        if done[0] == True:
            break

    actions_over_time = pd.DataFrame({"Datetime": history['Datetime'], "action": history['action']}).set_index(("Datetime"))
    return history, actions_over_time

if __name__ == "__main__":

    RECORD_EXPERIMENT = False
    save = {}
    save_images = {}
    save["Experiment"] = f"Run {datetime.now().strftime('%H:%M:%S')}: .NSEI"
    algorithm = "A2C"
    save["Algorithm"] = algorithm
    scale = False
    save["scale"] = scale
    TICKER = ".NSEI"
    save["TICKER"] = TICKER
    INIT_NET_WORTH = 10000
    save["INIT_NET_WORTH"] = INIT_NET_WORTH
    HISTORICAL_STATES = 5
    save["HISTORICAL_STATES"] = HISTORICAL_STATES
    features = [
        # {"feature": "Close", "lookback": 0},
        {"feature": "Fisher5", "lookback": 5},
        {"feature": "Fisher20", "lookback": 20},
        # {"feature": "Fisher50", "lookback": 50},
        # {"feature": "Fisher100", "lookback": 100},
        # {"feature": "Fisher150", "lookback": 150},
        # {"feature": "Fisher300", "lookback": 300},
    ]
    save["features"] = features
    LR = 0.001
    save["LR"] = LR
    RANDOM_SEED = 11111
    save["RANDOM_SEED"] = RANDOM_SEED
    train_percent = 0.8
    save["train_percent"] = train_percent


    if TICKER=="COSINE":
        length = get_data('.NSEI', 'D').shape[0]
        df = pd.DataFrame({
            'High': np.sin(np.arange(length) / 10.0)+2,
            'Open': np.sin(np.arange(length) / 10.0)+2,
            'Close': np.sin(np.arange(length) / 10.0)+2,
            'Low': np.sin(np.arange(length) / 10.0)+2,
            'Volume': np.abs(np.sin(np.arange(length) / 10.0)),
        }, index=get_data('.NSEI', 'D').Datetime
        )
    else:
        df = get_data(TICKER, 'D').set_index("Datetime")

    signal_features = pd.DataFrame(index=df.index)
    for feature, lookback in [(feature["feature"], feature["lookback"]) for feature in features]:
        if feature == "Close":
            signal_features[feature] = df[["Close"]]
        if feature.startswith("Fisher"):
            signal_features[feature] = add_fisher([df, lookback])[[f"Fisher{lookback}"]]

    train_df = df.iloc[:int(train_percent * len(df))]
    test_df = df.iloc[int(train_percent * len(df)):]
    train_signal_features = signal_features.iloc[:int(train_percent * len(signal_features))]
    test_signal_features = signal_features.iloc[int(train_percent * len(signal_features)):]
    NUM_EPISODES = 10
    save["NUM_EPISODES"] = NUM_EPISODES
    N_TIME_STEPS = NUM_EPISODES*(len(train_df) - HISTORICAL_STATES + 1)
    save["N_TIME_STEPS"] = N_TIME_STEPS

    env_train = DummyVecEnv(
        [lambda: StockTradingEnv(train_df, train_signal_features, INIT_NET_WORTH, HISTORICAL_STATES)])

    if scale:
        env_train = VecNormalize(env_train, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=100000)

    algorithm = getattr(stable_baselines3, algorithm)
    model = algorithm('MlpPolicy', env_train, verbose=1, learning_rate=LR, seed=RANDOM_SEED)
    # eval_callback = EvalCallback(env_train, best_model_save_path='./logs/',log_path='./logs/', eval_freq=1, deterministic=True, render=False)
    model.learn(total_timesteps=N_TIME_STEPS, log_interval=(len(train_df) - HISTORICAL_STATES + 1))#, callback=eval_callback

    env_test = DummyVecEnv(
        [lambda: StockTradingEnv(test_df, test_signal_features, INIT_NET_WORTH, HISTORICAL_STATES)])
    history_test, actions_over_time_test = evaluate_agent(env_test, test_df, model)
    history_test = pd.DataFrame(history_test).set_index("Datetime")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 7))
    ax1.plot(history_test['price'], label='Close')
    ax1_copy = ax1.twinx()
    ax1_copy.plot(actions_over_time_test, label='Actions')
    ax2.plot(actions_over_time_test, label='Actions')
    ax2_copy = ax2.twinx()
    for feature in [feature["feature"] for feature in features]:
        ax2_copy.plot(test_signal_features[feature], label=feature, color='green', ls='dotted')
    ax2_copy.axhline(0.0, ls='--', color='grey')
    ax3.plot(history_test['net_worth'], label='Net worth')
    ax3.plot(history_test['price'] * INIT_NET_WORTH / history_test['price'].iloc[0], label='Benchmark')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    save["Final_net_worth"] = history_test.iloc[-1]['net_worth']
    save_images["TestResults"] = fig
    if not (RECORD_EXPERIMENT):
        plt.show()

    if RECORD_EXPERIMENT:
        run = neptune.init(project="suprabhash/RL-StableBaselines", api_token=API_KEY)
        for key in save_images.keys():
            run[key].upload(save_images[key])
        for key in save.keys():
            run[key] = save[key]
    else:
        pass




