#%%

from helper_functions_volume import *
from celluloid import Camera
from IPython.display import HTML

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,8)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

import eikon as ek
import plotly.graph_objects as go
ek.set_app_key('9a249e0411184cf49e553b61a6e76c52d295ec17')

temp_og, temp_og1 = get_data("BJFN.NS")

import pickle
with open(f'Test.pkl', 'rb') as file:
    temp = pickle.load(file)

temp = pd.concat([temp.set_index("Date"), temp_og.set_index("Date")], axis=1, join="inner").reset_index()

fig = plt.figure()
camera = Camera(fig)
for i in tqdm(range(504,len(temp))):
    ax1 = temp.iloc[i-504:i]["Close"].plot()
    for lookback in [2,5,10,21,42,63,126,252,504]:
        dict = temp.iloc[i][f"PriceLevels_{lookback}"]
        poc = dict["poc"]
        profile_high = dict["profile_high"]
        profile_low = dict["profile_low"]
        vah = dict["vah"]
        val = dict["val"]
        # plt.plot((temp.index[i-lookback], poc), (temp.index[i], poc))
        ax1.hlines(y=poc, xmin=[i-lookback], xmax=[i],  color="orange")
        ax1.hlines(y=profile_high, xmin=[i-lookback], xmax=[i],   color="darkgreen")
        ax1.hlines(y=profile_low, xmin=[i-lookback], xmax=[i],   color="maroon")
        ax1.hlines(y=vah, xmin=[i-lookback], xmax=[i],   color="lime")
        ax1.hlines(y=val, xmin=[i-lookback], xmax=[i],   color="red")
    plt.legend(["Close","Point of Control", "Profile High", "Profile Low", "Value Area High", "Value Area Low"])
    #plt.show()
    camera.snap()
    plt.clf()

animation = camera.animate(blit=False, interval=1)

animation.save('Test.mp4',
               dpi=300,
               savefig_kwargs={
                   'frameon': False,
                   'pad_inches': 'tight'
               }
              )
#%%


