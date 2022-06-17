import datetime
import numpy as np
import pandas  as pd
import pandas_datareader.data as web
from Data.data_retrieval import get_data
from Utils.add_features import add_fisher
import itertools
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})


















