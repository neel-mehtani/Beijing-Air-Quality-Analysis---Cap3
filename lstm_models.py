
#imports
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM 
from tensorflow.keras.layers import Dropout
import os
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
get_ipython().system('pip install pandas')
get_ipython().system('pip install sklearn')
import pandas as pd
from lstm_supervised_conv import series_to_supervised

#Load in Data

df = pd.read_csv('beijing_master.csv')
df_gr = df.groupby('date').mean()
df_gr = df_gr.drop(['year', 'hour'], axis = 1)



train = df_gr.iloc[:-365]
test = df_gr.iloc[-365:]

feats = ['month', 'day', 'weekday','SO2', 'NO2', 'CO', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM','E', 'ENE', 'ESE', 'N', 'NE', 'NNE',
       'NNW', 'NW', 'S', 'SE', 'SSE', 'SSW', 'SW', 'W', 'WNW', 'WSW', 'AQI']


# LSTM: Univariate 2 Univariate

# # Baseline Avg

baseline_avg = [140.62]*len(df_gr['AQI'].iloc[-400:-365])

baseline_ms = mean_squared_error(baseline_avg, df_gr['AQI'].iloc[-400:-365])
rms_baseline = np.sqrt(ms)

t_lag = [1, 3, 5, 7, 10, 15, 20, 30] 

# # Prediction 15 days ahead

pred_15 = {}

#2 lstms, 1 hidden 
rms_15 = []
for t in t_lag:
    reframed = series_to_supervised(df_gr['AQI'], t, 1)
    values = reframed.values
    train = values[:-365, :]
    test = values[-365:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.fit_transform(test)

    train_X, train_y = scaled_train[:, :-1], scaled_train[:, -1]
    test_X, test_y = scaled_test[:, :-1], scaled_test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    model = Sequential()
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
    model.add(Dense(50, activation = 'tanh'))
#     model.add(Dense(25, activation ='tanh'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    
    history = model.fit(train_X, train_y, epochs= 100, batch_size=40,  verbose=1, shuffle=False
                                   , validation_data = (test_X, test_y))

    yhat = model.predict(test_X[:15])
    yhat = yhat.reshape(yhat.shape[0], yhat.shape[1])
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[2])
    inv_yhat = np.concatenate((yhat, test_X[:15]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]


    inv_y = np.concatenate((test_y[:15].reshape(15,1), test_X[:15]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    rms_15.append(np.sqrt(mean_squared_error(inv_yhat, inv_y)))

pred_15[(2,2)] = rms_15


#single LSTM layer, single hidden
rms_15 = []
for t in t_lag:
    reframed = series_to_supervised(df_gr['AQI'], t, 1)
    values = reframed.values
    train = values[:-365, :]
    test = values[-365:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.fit_transform(test)

    train_X, train_y = scaled_train[:, :-1], scaled_train[:, -1]
    test_X, test_y = scaled_test[:, :-1], scaled_test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    model = Sequential()
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
#     model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
    model.add(Dense(50, activation = 'tanh'))
#     model.add(Dense(25, activation ='tanh'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    
    history = model.fit(train_X, train_y, epochs= 100, batch_size=40,  verbose=1, shuffle=False
                                   , validation_data = (test_X, test_y))

    yhat = model.predict(test_X[:15])
    yhat = yhat.reshape(yhat.shape[0], yhat.shape[1])
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[2])
    inv_yhat = np.concatenate((yhat, test_X[:15]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]


    inv_y = np.concatenate((test_y[:15].reshape(15,1), test_X[:15]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    rms_15.append(np.sqrt(mean_squared_error(inv_yhat, inv_y)))

pred_15[(1,2)] = rms_15



# In[275]:


#single LSTM layer, single hidden
rms_15 = []
for t in t_lag:
    reframed = series_to_supervised(df_gr['AQI'], t, 1)
    values = reframed.values
    train = values[:-365, :]
    test = values[-365:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.fit_transform(test)

    train_X, train_y = scaled_train[:, :-1], scaled_train[:, -1]
    test_X, test_y = scaled_test[:, :-1], scaled_test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    model = Sequential()
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
#     model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
    model.add(Dense(50, activation = 'tanh'))
#     model.add(Dense(25, activation ='tanh'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    
    history = model.fit(train_X, train_y, epochs= 100, batch_size=40,  verbose=1, shuffle=False
                                   , validation_data = (test_X, test_y))

    yhat = model.predict(test_X[:15])
    yhat = yhat.reshape(yhat.shape[0], yhat.shape[1])
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[2])
    inv_yhat = np.concatenate((yhat, test_X[:15]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]


    inv_y = np.concatenate((test_y[:15].reshape(15,1), test_X[:15]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    rms_15.append(np.sqrt(mean_squared_error(inv_yhat, inv_y)))

pred_15[(1,2)] = rms_15

#2 lstm 2 hidden

t_lag = [1, 3, 5, 7, 10, 15, 20, 30] 
rms_15 = []
for t in t_lag:
    reframed = series_to_supervised(df_gr['AQI'], t, 1)
    values = reframed.values
    train = values[:-365, :]
    test = values[-365:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.fit_transform(test)

    train_X, train_y = scaled_train[:, :-1], scaled_train[:, -1]
    test_X, test_y = scaled_test[:, :-1], scaled_test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    model = Sequential()
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
    model.add(Dense(50, activation = 'tanh'))
    model.add(Dense(25, activation ='tanh'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    
    history = model.fit(train_X, train_y, epochs= 100, batch_size=40,  verbose=1, shuffle=False
                                   , validation_data = (test_X, test_y))

    yhat = model.predict(test_X[:15])
    yhat = yhat.reshape(yhat.shape[0], yhat.shape[1])
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[2])
    inv_yhat = np.concatenate((yhat, test_X[:15]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]


    inv_y = np.concatenate((test_y[:15].reshape(15,1), test_X[:15]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    rms_15.append(np.sqrt(mean_squared_error(inv_yhat, inv_y)))

pred_15[(2,3)] = rms_15


# # Predict 30 days ahead

pred_30 = {}

#2 LSTM 1 hidden
t_lag = [1, 3, 5, 7, 10, 15, 20, 30] 
rms_30 = []
for t in t_lag:
    reframed = series_to_supervised(df_gr['AQI'], t, 1)
    values = reframed.values
    train = values[:-365, :]
    test = values[-365:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.fit_transform(test)

    train_X, train_y = scaled_train[:, :-1], scaled_train[:, -1]
    test_X, test_y = scaled_test[:, :-1], scaled_test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    model = Sequential()
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
    model.add(Dense(50, activation = 'tanh'))
#     model.add(Dense(25, activation ='tanh'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    
    history = model.fit(train_X, train_y, epochs= 100, batch_size=40,  verbose=1, shuffle=False
                                   , validation_data = (test_X, test_y))

    yhat = model.predict(test_X[:30])
    yhat = yhat.reshape(yhat.shape[0], yhat.shape[1])
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[2])
    inv_yhat = np.concatenate((yhat, test_X[:30]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]


    inv_y = np.concatenate((test_y[:30].reshape(30,1), test_X[:30]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    rms_30.append(np.sqrt(mean_squared_error(inv_yhat, inv_y)))

pred_30[(2,2)] = rms_30

#1 LSTM 1 hidden
rms_30 = []
for t in t_lag:
    reframed = series_to_supervised(df_gr['AQI'], t, 1)
    values = reframed.values
    train = values[:-365, :]
    test = values[-365:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.fit_transform(test)

    train_X, train_y = scaled_train[:, :-1], scaled_train[:, -1]
    test_X, test_y = scaled_test[:, :-1], scaled_test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    model = Sequential()
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
#     model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
    model.add(Dense(50, activation = 'tanh'))
#     model.add(Dense(25, activation ='tanh'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    
    history = model.fit(train_X, train_y, epochs= 100, batch_size=40,  verbose=1, shuffle=False
                                   , validation_data = (test_X, test_y))

    yhat = model.predict(test_X[:30])
    yhat = yhat.reshape(yhat.shape[0], yhat.shape[1])
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[2])
    inv_yhat = np.concatenate((yhat, test_X[:30]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]


    inv_y = np.concatenate((test_y[:30].reshape(30,1), test_X[:30]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    rms_30.append(np.sqrt(mean_squared_error(inv_yhat, inv_y)))

pred_30[(1,2)] = rms_30


#1 LSTM 2 hidden
rms_30 = []
for t in t_lag:
    reframed = series_to_supervised(df_gr['AQI'], t, 1)
    values = reframed.values
    train = values[:-365, :]
    test = values[-365:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.fit_transform(test)

    train_X, train_y = scaled_train[:, :-1], scaled_train[:, -1]
    test_X, test_y = scaled_test[:, :-1], scaled_test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    model = Sequential()
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
#     model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
    model.add(Dense(50, activation = 'tanh'))
    model.add(Dense(25, activation ='tanh'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    
    history = model.fit(train_X, train_y, epochs= 100, batch_size=40,  verbose=1, shuffle=False
                                   , validation_data = (test_X, test_y))

    yhat = model.predict(test_X[:30])
    yhat = yhat.reshape(yhat.shape[0], yhat.shape[1])
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[2])
    inv_yhat = np.concatenate((yhat, test_X[:30]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]


    inv_y = np.concatenate((test_y[:30].reshape(30,1), test_X[:30]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    rms_30.append(np.sqrt(mean_squared_error(inv_yhat, inv_y)))

pred_30[(1,2)] = rms_30

#2 LSTM 2 hidden
rms_30 = []
for t in t_lag:
    reframed = series_to_supervised(df_gr['AQI'], t, 1)
    values = reframed.values
    train = values[:-365, :]
    test = values[-365:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.fit_transform(test)

    train_X, train_y = scaled_train[:, :-1], scaled_train[:, -1]
    test_X, test_y = scaled_test[:, :-1], scaled_test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    model = Sequential()
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
    model.add(Dense(50, activation = 'tanh'))
    model.add(Dense(25, activation ='tanh'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    
    history = model.fit(train_X, train_y, epochs= 100, batch_size=40,  verbose=1, shuffle=False
                                   , validation_data = (test_X, test_y))

    yhat = model.predict(test_X[:30])
    yhat = yhat.reshape(yhat.shape[0], yhat.shape[1])
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[2])
    inv_yhat = np.concatenate((yhat, test_X[:30]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]


    inv_y = np.concatenate((test_y[:30].reshape(30,1), test_X[:30]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    rms_30.append(np.sqrt(mean_squared_error(inv_yhat, inv_y)))

pred_30[(2,3)] = rms_30

#view rms_15 and rms_30 for best network config (#lstm layers, #hidden)

