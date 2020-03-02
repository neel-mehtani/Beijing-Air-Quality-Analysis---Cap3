# Beijing Air Quality Time Series Analysis

## Background and Motivation

Air pollution in urban environments has risen steadily in the last several decades. As a growing and urgent public health concern, cities and environmental agencies have been exploring methods to forecast future air pollution, hoping to enact policies and provide incentives and services to benefit their citizenry. Much research is being conducted in environmental science to generate deterministic models of air pollutant behavior; however, this is both complex, as the underlying molecular interactions in the atmosphere need to be simulated, and often inaccurate. As a result, with greater computing power in the twenty-first century, using machine learning methods for forecasting air pollution has become more popular. This study investigates the use of the LSTM recurrent neural network (LSTM-RNN) as a framework for forecasting in the future, based on time series data of pollution and meteorological information in Beijing. Due to the sequence dependencies associated with large-scale and longer time series datasets, RNNs, and in particular LSTM models, are well-suited. To support the investigation of LSTM models for time series analysis, various supervised learning models like Random Forests/ExtraTreeRegressors are also implemented to test their predictive ability. 

Our results show that the LSTM framework produces equivalent accuracy when predicting future timesteps compared to the baseline support vector regression for a single timestep. Using our LSTM framework, we can now extend the prediction from a single timestep out to 5 to 10 hours into the future. This is promising in the quest for forecasting urban air quality and leveraging that insight to enact beneficial policy

## Dataset

The data for this investigation was procured from UCI's open Machine Learning Repository at: https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data. 

Within the data directory, there is both meterological and pollutant information recorded on an hourly basis across 12 nationally-controlled air-quality monitoring sites from 03/01/2013 - 02/28/2017. Given the granular nature of air quality and weather data records, the dataset is appropriate for time series analysis predictions, where information relating to weather patterns or pollutant concentrations can be used to compute/predict air quality, which is converted into the AQI metric (Air Quality Index). 

In its raw form, the data contains information on:
- 6 atmospheric pollutants: CO, NO2, SO2, O3, PM2.5, and PM10 (recorded as ug/m3).
- 6 meterological factors: TEMP, PRES (pressure), DEWP (dewpoint), RAIN, WSPM (Wind Speed m/s), and Wind Direction (16 angular types). 

In preprocessing the data, attention must be given to converting pollutant concentrations to appropriate units with which they can be handled (O3 & CO: ppm, NO2 & SO2: ppb). The necessary conversions can be implemented using the following formula. 

<img src = "unit_conv.png">
                                                  
Additionally, feature engineering for the str-type wind direction is required to enable its predictive value in AQI. For this, we implement dummy columns that take all 16 wind directions into account. 

Finally, the time steps of data to be used in the analysis must be determined for the time-series prediction. Here, mean aggregation is performed on the meterological and pollutant data points across 24 hours for each day to condense the dataset into a daily format. Since AQI is predominantly determined by the concentrations of O3, PM2.5, and PM10 (which are influenced by alternative pollutants and weather), the following AQI formula is used to compute each pollutant's AQI value, the maximum of which is the deterministic AQI for that time point. 

<img src = "aqi_eqn.png">

For this investigation, the intial 3 years of data are used for training (2013-16), while data from the last year (2017) is used as a hold out set to examine how different time series predictor models perform. 

## EDA

<img src = "aqi_cat_table.png" height="300">

<img src = "beijing_ts.png" width="500" height="200">

<img src = "Screen Shot 2020-01-09 at 4.17.59 PM.png">


## Supervised Learning Time Series Analysis



## LSTM Neural Network Time Series Prediction 



## Future Scope 









