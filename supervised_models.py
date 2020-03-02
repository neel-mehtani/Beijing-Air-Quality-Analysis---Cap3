
#run on docker container
import sys
get_ipython().system('{sys.executable} -m pip install --pre -f https://sklearn-nightly.scdn8.secure.raxcdn.com scikit-learn')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_random_state
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt


# # Load in Data

df = pd.read_csv('beijing_master.csv')
df_gr = df.groupby('date').mean()
df_gr = df_gr.drop(['year', 'hour'], axis = 1)

# # Segregate Train and Test 

train = df_gr.iloc[:-365]
test = df_gr.iloc[-365:]


aqi_names = {'PM2.5': 'pm2', 'PM10': 'pm10', 'O3': 'o3'}


# # Multivariate Output Target: Pollutant Outcome Information 
#ExtraTreeRegressor Model
#no feature selection

X_train = train[['month', 'day', 'TEMP',
       'PRES', 'DEWP', 'RAIN', 'WSPM', 'fall', 'spring', 'summer', 'winter',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'weekday', 'CO', 'NO2', 'SO2']]

X_test = test[['month', 'day', 'TEMP',
       'PRES', 'DEWP', 'RAIN', 'WSPM', 'fall', 'spring', 'summer', 'winter',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'weekday', 'CO', 'NO2', 'SO2']]

y_train = train[['PM2.5', 'PM10','O3']]
y_test = test[['PM2.5', 'PM10','O3']]

# # Set up Model for Multivariate Output Target
model = ExtraTreesRegressor() #regular model fit
model.fit(X_train, y_train)

#plot feature importances
ft = pd.DataFrame({'fi': model.feature_importances_, 'feature': X_train.columns})

ax = ft.plot.barh(x='feature')
plt.show()

results = model.predict(X_test)
results = pd.DataFrame(results, columns = y_test.columns, index = test.index)


#convert pollutant predictions to corresponding AQI preds (df_aqi) and pollutant info from y_test to AQIs (y_test_aqi)
#aqi_target, aqi_preds = max of the respective AQI columns.

df_aqi = pd.DataFrame()
y_test_aqi = pd.DataFrame()

for i in aqi_names.keys(): 
    insert_col = 'AQI_' + i
    df_aqi[insert_col] = results[i].apply(lambda x: compute_aqi(x, aqi_names[i]))
    y_test_aqi[insert_col] = y_test[i].apply(lambda x: compute_aqi(x, aqi_names[i]))

df_aqi['AQI'] = df_aqi.max(axis = 1)
y_test_aqi['AQI'] = y_test_aqi.max(axis = 1)

aqi_target = y_test_aqi['AQI']
aqi_preds = df_aqi['AQI']


x = pd.DataFrame({'col1': aqi_preds, 'col2': aqi_target})

#compute the RMSE of the model


rms = sqrt(mean_squared_error(x['col1'], x['col2'])) 

baseline_target = [np.mean(aqi_target)]*len(aqi_target)
rms_b = sqrt(mean_squared_error(aqi_target, baseline_target))

rms, rms_b


# Feature Selection MultiVariate Output Model

# In[32]:


#run the model with some predictors dropped to test out the score

X_train = train[['month', 'TEMP',
       'PRES', 'RAIN', 
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'CO', 'NO2', 'SO2']]

X_test = test[['month', 'TEMP',
       'PRES','RAIN', 
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'CO', 'NO2', 'SO2']]

y_train = train[['PM2.5', 'PM10','O3']]

y_test = test[['PM2.5', 'PM10','O3']]




# In[33]:


model_ = ExtraTreesRegressor()
model_.fit(X_train, y_train)
results_ = model_.predict(X_test)
results_ = pd.DataFrame(results_, columns = y_test.columns, index = test.index)
results_


# In[34]:


#convert pollutant predictions to corresponding AQI preds (df_aqi) and pollutant info from y_test to AQIs (y_test_aqi)
#aqi_target, aqi_preds = max of the respective AQI columns.



aqi_names = {'PM2.5': 'pm2', 'PM10': 'pm10', 'O3': 'o3'}

df_aqi = pd.DataFrame()
y_test_aqi = pd.DataFrame()

for i in aqi_names.keys(): 
    insert_col = 'AQI_' + i
    df_aqi[insert_col] = results_[i].apply(lambda x: compute_aqi(x, aqi_names[i]))
    y_test_aqi[insert_col] = y_test[i].apply(lambda x: compute_aqi(x, aqi_names[i]))

df_aqi['AQI'] = df_aqi.max(axis = 1)
y_test_aqi['AQI'] = y_test_aqi.max(axis = 1)

aqi_target = y_test_aqi['AQI']
aqi_preds = df_aqi['AQI']


# In[35]:


x_ = pd.DataFrame({'col1': aqi_preds, 'col2': aqi_target})
x_


# In[36]:


rms = sqrt(mean_squared_error(aqi_target, aqi_preds)) 

baseline_target = [np.mean(aqi_target)]*len(aqi_target)
rms_b = sqrt(mean_squared_error(aqi_target, baseline_target))

rms, rms_b


# In[136]:


#choose feature selected train model for tuning hyperparameters


# In[37]:


param_grid = {
    'bootstrap': [True, False],
    'max_depth': [80, 90, 100, 110, 120, 130, 140, 150],
    'max_features': [3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_leaf': [3, 4, 5],
    'n_estimators': [100, 120, 140, 160, 180, 200]
}

grid_search = GridSearchCV(estimator = model_,
                          param_grid = param_grid, 
                          cv = 3, 
                           n_jobs = -1,
                          verbose = 2)


# In[39]:


grid_search.fit(X_train, y_train)


# In[61]:


grid_search.best_params_


# In[40]:


grid_search.best_estimator_.fit(X_train, y_train)
preds = grid_search.best_estimator_.predict(X_test)


# In[41]:


preds


# In[45]:


results_cv = pd.DataFrame(preds, columns = y_test.columns)
results_cv


# In[52]:


#convert pollutant predictions to corresponding AQI preds (df_aqi) and pollutant info from y_test to AQIs (y_test_aqi)
#aqi_target, aqi_preds = max of the respective AQI columns.

df_aqi = pd.DataFrame()
y_test_aqi = pd.DataFrame()

for i in aqi_names.keys(): 
    insert_col = 'AQI_' + i
    df_aqi[insert_col] = results_cv[i].apply(lambda x: compute_aqi(x, aqi_names[i]))
    y_test_aqi[insert_col] = y_test[i].apply(lambda x: compute_aqi(x, aqi_names[i]))

df_aqi['AQI'] = df_aqi.max(axis = 1)
y_test_aqi['AQI'] = y_test_aqi.max(axis = 1)

aqi_target = y_test_aqi['AQI']
aqi_preds = df_aqi['AQI']
aqi_preds.index = aqi_target.index


# In[59]:


x_ = pd.DataFrame({'preds': aqi_preds, 'target': aqi_target})
x_


# In[60]:


rms = sqrt(mean_squared_error(aqi_target, aqi_preds)) 
baseline_target = [np.mean(aqi_target)]*len(aqi_target)
rms_b = sqrt(mean_squared_error(aqi_target, baseline_target))
rms, rms_b


# # Univariate Output Target: AQI Outcome Information 

# RandomForest models

X_train = train[['month', 'day', 'TEMP',
       'PRES', 'DEWP', 'RAIN', 'WSPM', 'fall', 'spring', 'summer', 'winter',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'weekday', 'CO', 'NO2', 'SO2']]

X_test = test[['month', 'day', 'TEMP',
       'PRES', 'DEWP', 'RAIN', 'WSPM', 'fall', 'spring', 'summer', 'winter',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'weekday', 'CO', 'NO2', 'SO2']]

y_train = train['AQI']
y_test = test['AQI']

#all features included for simple RF model


rf = RandomForestRegressor()
rf.fit(X_train, y_train)
preds = rf.predict(X_test)

baseline_target = [np.mean(y_test)]*len(y_test)
rms_b = sqrt(mean_squared_error(y_test, baseline_target))
rms = sqrt(mean_squared_error(preds, y_test))
rms, rms_b

rf_df = pd.DataFrame({'preds': preds, 'target': y_test})
rf_df.to_csv('random_forest_ts.csv') #best performing model

#attain 
plt.barh(X_train.columns, rf.feature_importances_)
plt.show()





#feature selection on simple RF model

X_train = train[['month',  'TEMP', 'weekday', 'day',
       'PRES', 'WSPM', 'RAIN', 'winter', 
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'CO', 'NO2', 'SO2']]

X_test = test[['month', 'TEMP', 'weekday', 'day',
       'PRES', 'WSPM', 'RAIN', 'winter',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'CO', 'NO2', 'SO2']]

rf_f = RandomForestRegressor()
rf_f.fit(X_train, y_train)
preds = rf_f.predict(X_test)

baseline_target = [np.mean(y_train[-365:])]*len(y_train[-365:])
rms_b = sqrt(mean_squared_error(y_test, baseline_target))
rms = sqrt(mean_squared_error(preds, y_test))
rms, rms_b

#GridSearchCV on RF model without features selected
X_train = train[['month', 'day', 'TEMP',
       'PRES', 'DEWP', 'RAIN', 'WSPM', 'fall', 'spring', 'summer', 'winter',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'weekday', 'CO', 'NO2', 'SO2']]

X_test = test[['month', 'day', 'TEMP',
       'PRES', 'DEWP', 'RAIN', 'WSPM', 'fall', 'spring', 'summer', 'winter',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'weekday', 'CO', 'NO2', 'SO2']]

param_grid = {
    'bootstrap': [True, False],
    'max_depth': [80, 90, 100, 110, 120, 130, 140, 150],
    'max_features': [3, 4, 5],
    'min_samples_leaf': [3, 4, 5],
    'n_estimators': [100, 120, 140, 160, 180, 200]
}

grid_search = GridSearchCV(estimator = rf,
                          param_grid = param_grid, 
                          cv = 3, 
                           n_jobs = -1,
                          verbose = 2)

grid_search.fit(X_train, y_train)


grid_search.best_estimator_.fit(X_train, y_train)
preds = grid_search.best_estimator_.predict(X_test)


baseline_target = [np.mean(y_train[-365:])]*len(y_train[-365:])
rms_b = sqrt(mean_squared_error(y_test, baseline_target))
rms = sqrt(mean_squared_error(preds, y_test))
rms, rms_b

print(rms, rms_b, grid_search.best_params_)


# In[90]:


rms, rms_b


# In[94]:


rf = RandomForestRegressor()
rf.fit(X_train, y_train)
preds = rf.predict(X_test)

baseline_target = [np.mean(y_test)]*len(y_test)
rms_b = sqrt(mean_squared_error(y_test, baseline_target))
rms = sqrt(mean_squared_error(preds, y_test))
rms, rms_b


# In[68]:


model.feature_importances_

plt.barh(X_train.columns, model.feature_importances_)
plt.show()


# In[69]:


X_train = train[['month',  'TEMP', 'weekday', 'day',
       'PRES', 'WSPM', 'RAIN', 'winter', 
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'CO', 'NO2', 'SO2']]

X_test = test[['month', 'TEMP', 'weekday', 'day',
       'PRES', 'WSPM', 'RAIN', 'winter',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'CO', 'NO2', 'SO2']]


# In[70]:


model = RandomForestRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)



# In[71]:


baseline_target = [np.mean(y_test)]*len(y_test)
rms_b = sqrt(mean_squared_error(y_test, baseline_target))
rms = sqrt(mean_squared_error(preds, y_test))
rms, rms_b


# Gradient Boosting Regressor

# In[96]:


#no feature selection. Simple gradient boosting regressor.


# In[97]:


X_train = train[['month', 'day', 'TEMP',
       'PRES', 'DEWP', 'RAIN', 'WSPM', 'fall', 'spring', 'summer', 'winter',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'weekday', 'CO', 'NO2', 'SO2']]

X_test = test[['month', 'day', 'TEMP',
       'PRES', 'DEWP', 'RAIN', 'WSPM', 'fall', 'spring', 'summer', 'winter',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'weekday', 'CO', 'NO2', 'SO2']]

gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
preds = gb.predict(X_test)


# In[98]:


baseline_target = [np.mean(y_test)]*len(y_test)
rms_b = sqrt(mean_squared_error(y_test, baseline_target))
rms = sqrt(mean_squared_error(preds, y_test))
rms, rms_b


# In[897]:


plt.barh(X_train.columns, model.feature_importances_)
plt.show()


# In[99]:


#feature selection for GradientBoosting

X_train = train[['month',  'TEMP','weekday',
       'PRES', 'WSPM', 'RAIN', 'winter', 'DEWP',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'CO', 'NO2', 'SO2']]

X_test = test[['month', 'TEMP','weekday',
       'PRES', 'WSPM', 'RAIN', 'winter', 'DEWP',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'CO', 'NO2', 'SO2']]

gb_f = GradientBoostingRegressor()
gb_f.fit(X_train, y_train)
preds = gb_f.predict(X_test)


# In[100]:


baseline_target = [np.mean(y_test)]*len(y_test)
rms_b = sqrt(mean_squared_error(y_test, baseline_target))
rms = sqrt(mean_squared_error(preds, y_test))
rms, rms_b

#GridSearch CV for feature selected GradientBoost. 
X_train = train[['month',  'TEMP','weekday',
       'PRES', 'WSPM', 'RAIN', 'winter', 'DEWP',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'CO', 'NO2', 'SO2']]

X_test = test[['month', 'TEMP','weekday',
       'PRES', 'WSPM', 'RAIN', 'winter', 'DEWP',
       'E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
       'SSW', 'SW', 'W', 'WNW', 'WSW', 'CO', 'NO2', 'SO2']]


param_grid = {
    'max_depth': [80, 90, 100, 110, 120, 130, 140, 150],
    'max_features': [3, 4, 5],
    'min_samples_leaf': [3, 4, 5],
    'n_estimators': [100, 120, 140, 160, 180, 200]
}

grid_search = GridSearchCV(estimator = gb_f,
                          param_grid = param_grid, 
                          cv = 3, 
                           n_jobs = -1,
                          verbose = 2)

grid_search.fit(X_train, y_train)

grid_search.best_params_

grid_search.best_estimator_.fit(X_train, y_train)
preds = grid_search.best_estimator_.predict(X_test)

baseline_target = [np.mean(y_train[-365:])]*len(y_train[-365:])
rms_b = sqrt(mean_squared_error(y_test, baseline_target))
rms = sqrt(mean_squared_error(preds, y_test))
rms, rms_b

print(rms, rms_b, grid_search.best_params_)

