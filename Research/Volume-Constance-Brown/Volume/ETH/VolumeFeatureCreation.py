import sys
import pandas as pd
import pickle

sys.path.append('C:/Users/suprabhashsahu/Desktop/Volume-Constance-Brown/Volume')
from helper_functions_volume import *

if __name__ == "__main__":

    temp_og1 = get_data_ETH_minute()
    temp_og = resample_data(temp_og1, minutes=60)
    temp = return_volume_features_minute_hourly(temp_og, temp_og1)

    with open(f'ETH_VolumeLevels_Hourly.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(temp), file)