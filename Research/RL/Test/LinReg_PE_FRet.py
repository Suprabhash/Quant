k_best = 504
l_best = 495
n_best = 5
with open(f'df_PE_ZS{l_best}_offset{n_best}_FRet{k_best}.pkl', 'rb') as file:
    best_c = pickle.load(file)

# Readying the data --best_c here refers to the dataframe from the pickle file I sent over
test_start_dt = best_c.loc[best_c.FRet.isna()].index[0]
test_end_dt = best_c.loc[best_c.FRet.isna()].index[-1]
train_start_dt = best_c.loc[best_c.PE_ZS_offset.isna()].index[-1]

# Creating Train-Test set
train_set = best_c.loc[(best_c.index>train_start_dt) & (best_c.index<test_start_dt)]
test_set = best_c.loc[(best_c.index>=test_start_dt) & (best_c.index<=test_end_dt)]

# Creating arrays
X_train = np.array(train_set['PE_ZS_offset']).reshape(-1,1)
Y_train = np.array(train_set['FRet'])
X_test = np.array(test_set['PE_ZS_offset']).reshape(-1,1)

# Ready a linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
print(f"coefficient of determination: {model.score(X_train, Y_train)}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")

# Predict and replace the prediction inside NaN values of test set
y_pred = model.predict(X_test)
test_df = test_set.copy()
test_df['FRet'] = y_pred

# Plot 
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(test_df[f'PE_ZS_offset'].values, label='PE_ZS')
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.plot(test_df[f'FRet'].values, color='red', label='FRet')
ax2.legend()
ax2.title.set_text(f'PE_ZS{l_best}_offset{n_best}/FRet{k_best}')