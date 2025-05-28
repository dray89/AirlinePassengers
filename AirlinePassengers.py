#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')
from AirlinePassengers import UsefulFunctions


# In[2]:


#Time series analysis in Python
dataset = pd.read_csv('/Users/coding/Documents/GitHub/TimeSeriesAnalysisWithPython/data/AirPassengers.csv')
#Parse strings to datetime type
dataset['Month'] = pd.to_datetime(dataset['Month'], infer_datetime_format=True)
indexedDataset = dataset.set_index(['Month'])


# In[3]:


indexedDataset


# In[4]:


# Our data is not stationary as seen from the graph below. 
# Due to high growth in airline traffic, the mean is changing year over year.
plt.xlabel('Date')
plt.ylabel('Number of air passengers')
plt.title('Number of Airline Passengers')
plt.plot(indexedDataset)


# In[5]:


rolmean = indexedDataset.rolling(window = 12).mean()
rolstd = indexedDataset.rolling(window = 12).std()


# In[6]:


# The mean number of the airline passengers increases year over year
rolmean[12:]


# In[7]:


plt.xlabel('Date')
plt.ylabel('Mean Number of air passengers')
plt.title('Rolling 12-Month Mean of Airline Passengers')
plt.plot(rolmean)


# In[8]:


# The standard deviation is also increasing over all over time and 
# fluctuates up and down between years. 
rolstd[12:]


# In[9]:


# As you can see, the standard deviation of air passengers increases overall 
# The standard deviation fluctuates between years in an up and down pattern.
plt.xlabel('Date')
plt.ylabel('Standard Deviation of air passengers')
plt.title('Rolling 12-Month Standard Deviation of Airline Passengers')
plt.plot(rolstd)


# In[10]:


orig = plt.plot(indexedDataset, color = 'blue', label = 'Number of airline passengers')
mean = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
std = plt.plot(rolstd, color = 'black', label = 'Rolling Standard Deviation')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation of Number of Airline Passengers')
plt.show(block=False)


# In[11]:


Dickey_Fuller_Test = adfuller(indexedDataset['#Passengers'], autolag='AIC')
pd.Series(Dickey_Fuller_Test)


# In[12]:


# The results of the Dickey Fuller Test show that the data is not stationary
Dickey_Fuller_Output = pd.Series(Dickey_Fuller_Test, index = 
          ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used', 'Test Statistics', 'Maximized Information Criterion'])

for key, value in Dickey_Fuller_Test[4].items():
    Dickey_Fuller_Output['Critical Value (%s)'%key] = value #Unpacking embeded critical value dictionary
    
Dickey_Fuller_Output = Dickey_Fuller_Output.drop('Test Statistics')
Dickey_Fuller_Output


# In[13]:


#KPSS also shows that the data is not stationary.
#KPSS tests for stationarity around a deterministic trend
kpsstest = kpss(indexedDataset['#Passengers'], regression='c', nlags="auto")
kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','#Lags Used'])
for key,value in kpsstest[3].items():
    kpss_output['Critical Value (%s)'%key] = value
kpss_output


# In[14]:


#Estimating Trend: The pattern has not changed
indexedDataset_logScale = np.log(indexedDataset)
plt.plot(indexedDataset_logScale)
plt.xlabel('Date')
plt.ylabel('Log Scale of Number of Air Passengers')
plt.title('Estimating Trend: The pattern has not changed')


# In[15]:


# The mean is still not stationary, however it is more stationary than before!
movingAverage = indexedDataset_logScale.rolling(window=12).mean()
movingStd = indexedDataset_logScale.rolling(window=12).std()
plt.plot(indexedDataset_logScale, label='Log Scale Data')
plt.xlabel('Date')
plt.ylabel('Log Scale of Number of Air Passengers & Moving Average')
plt.plot(movingAverage, color='red', label = 'Moving Average')
plt.title('Log Scale of Number of Airline Passengers & Moving Average')
plt.legend()


# In[16]:


# Make the time series stationary
LogScaleMinusMovingAverage = indexedDataset_logScale - movingAverage
LogScaleMinusMovingAverage.dropna(inplace=True)
LogScaleMinusMovingAverage.tail(10)


# In[17]:


    


# In[18]:


# Our data looks more stationary
UsefulFunctions.plot_stationarity(LogScaleMinusMovingAverage)


# In[19]:


#Augmented Dickey Fuller test
#Our p-value has decreased substantially. 
#We reject the null hypothesis that the data is non-stationary
test_stationarity(LogScaleMinusMovingAverage)


# In[20]:


#KPSS Test
#Cannot be used interchangeably with Dickey Fuller
#Tests for stationarity around a deterministic trend
kpss_test(LogScaleMinusMovingAverage)
#the data is basically stationary. We do not reject the null. 


# #### Exponential Decay Weighted Average

# In[21]:


#Calculate the trend of the original time series
exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife = 12, min_periods=0, adjust = True).mean()
plt.plot(indexedDataset_logScale, label='Log Scale Data')
plt.plot(exponentialDecayWeightedAverage, label='Expon Decay Weighted Mean', color='red')
plt.xlabel('Date')
plt.ylabel('Exponential Decay Weighted Average / Log Scale Data')
plt.legend()
plt.title('Exponential Decay Weighted Average vs Log Scale Data')


# In[22]:


# Standard Deviation More Flat
# Rolling Mean is fairly stable between 0 and .2
# No trend in Number of Airline Passengers
LogScaleMinusExponentialDecayWeightedAverage = indexedDataset_logScale - exponentialDecayWeightedAverage
plot_stationarity(LogScaleMinusExponentialDecayWeightedAverage)


# In[23]:


#P-value is lower than previous when we subtracted moving average
#The data is stationary
#We reject the null hypothesis that the data is non-stationary
test_stationarity(LogScaleMinusExponentialDecayWeightedAverage)


# In[24]:


#We do not reject the null that the data is stationary
kpss_test(LogScaleMinusExponentialDecayWeightedAverage)


# #### Shift Data

# In[25]:


#Differentiating the time series by 1 
dataShifted = indexedDataset_logScale - indexedDataset_logScale.shift()
plt.plot(dataShifted)


# In[26]:


#Our data looks stationary but let's run the test in the next cell
dataShifted.dropna(inplace=True)
plot_stationarity(dataShifted)


# In[27]:


#Our time series is basically stationary. 
#We reject the null hypothesis that the data is non-stationary
test_stationarity(dataShifted)


# In[28]:


kpss_test(dataShifted) #We do not reject the null that the data is non-stationary


# In[29]:


decomposition = seasonal_decompose(indexedDataset_logScale)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logScale, label= 'Number of Airline Passengers')
plt.legend(loc = 'best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# In[30]:


decomposedLogData = residual 
decomposedLogData.dropna(inplace=True) 


# In[31]:


# High variation in residual between 1950 - 1954 & 1957 - 1960
# Lower variability between 1954 - 1957 which is interesting.
plot_stationarity(decomposedLogData)


# In[32]:


plot_acf(indexedDataset_logScale)
#Blue area is confidence interval
#There is seasonality in the data
# The order of the Moving Average part can be inferred from the ACF plot (q or Q)
# The blue area is the confidence interval
# y-axis is the autocorrelation level
# x-axis Lags
# Autocorrelation is high at least around .5
# This is an ACF plot with 20 lags.


# In[33]:


#Partial Auto-Correlation Function (PACF)
d = indexedDataset_logScale.diff().dropna()
plot_pacf(d)
#Partial Auto-Correlation is the relationship between a lagged value and its succeeding value.
#Here there is both positive and negative Autocorrelation 
# The order of AR can be inferred from this plot


# #### SARIMA
# ##### The SARIMA model works much better on this data than ARIMA or ARMA due to seasonality. I am not splitting the data into train and test sets for this demonstration because it is a very small dataset.

# In[91]:


# fit model
#train = indexedDataset[:len(indexedDataset)*.7]
#test = indexedDataset[len(indexedDataset)*7:]
#model = SARIMAX(train, order= (2,1,1), seasonal_order = (1,1,2,12))
model = SARIMAX(indexedDataset, order=(2,1,1), seasonal_order=(1,1,2,12))
model_fit = model.fit(disp=False)
# make prediction
preds = model_fit.forecast(steps= 20)


# In[92]:


predictions = pd.Series(indexedDataset['#Passengers'], 
                           index = indexedDataset.index)
predictions = predictions.add(preds, fill_value = 0)


# In[93]:


rmse = sqrt(mean_squared_error(model_fit.fittedvalues, indexedDataset))


# In[94]:


plt.plot(predictions[:-19], label='Original Data')
plt.plot(predictions[-20:], label='Predictions') #That's a huge improvement!
plt.legend()
plt.xlabel('Date')
plt.ylabel('# of Airline Passengers')
plt.title('# of Airline Passengers and Projections')
plt.annotate(f'RMSE is {rmse}', xy =(datetime.strptime("1950-01-01", "%Y-%m-%d"), 600))


# #### SARIMA Grid Search Coding Mental Challenge
# The ARIMA has a grid search for the best (p, d, q) values. We could do the same thing with SARIMA to find the best combination of parameters. I have done this with other statistical methods including logistic regression in Python using multithreading for speed. We would also need to use combinatorics to populate all the possible combinations without repetition where order matters: permutations. I have the code to populate all the permutations:
# 
# ```
# from itertools import permutations
# output = []
# for c in permutations([i for i in range(6)], 3):
#     output.append(c)
# 
# ```
# 
# The next step would be to construct a for loop and submit all of them to the 'submit' method and use maximum likelihood estimation method, conditional sum of squares or another method to rate the models with varying parameters to choose the best one.
# 
# For example, here is some similar code that I wrote for fun to choose the best parameters for a logistic regression model. I was inspired by the grid search functionality in sklearn. Using multithreading, I was able to create my own grid search that ran much faster than sklearn's grid search function.
# 
# ```
# 
# def thread_training(X_train, y_train, X_test, y_test, solver, C):
#     threads = min(100, len(solver)*len(C)*2)
#     output = {}
#     with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
#         for s in solver:
#             for c in C:
#                 if s in ['newton-cg','lbfgs', 'sag']:
#                     output[executor.submit(run_regression, X_train, y_train, solver=s, penalty='l2', C=c)] = (s, 'l2', c)
#                 elif s =='saga':
#                     for p in ['l1', 'l2', 'elasticnet']:
#                         if p == 'elasticnet':
#                             for l1 in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
#                                 output[executor.submit(run_regression, X_train, y_train, solver=s, penalty=p, C=c, l1_ratio=l1)] =(s,p,c,l1)
#                         else:
#                             output[executor.submit(run_regression, X_train, y_train, solver=s, penalty=p, C=c)] = (s, p, c)
#                 else:
#                     for p in ['l1', 'l2']:
#                         output[executor.submit(run_regression, X_train, y_train, solver=s, penalty=p, C=c)] = (s, p, c)
#     return output
# 
# def thread_save(results, X_train, y_train, X_test, y_test):
#     threads = min(100, len(results))
#     output = {}
#     with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
#         for each in results:
#             output[each] = executor.submit(save_results, each, X_train, y_train, X_test, y_test)
#     return output
# 
# def run_regression(X_train, y_train, solver, penalty, C, l1_ratio=None):
# 
#     reg = LogisticRegression(C=C, penalty=penalty, 
#                              random_state=0, solver=solver, 
#                              max_iter=100,l1_ratio=l1_ratio).fit(X_train, y_train)
#     return reg
# 
# def save_results(results, X_train, y_train, X_test, y_test):
#         y_pred = results.result().predict(X_test)
#         tn, fp, fn, tp  = confusion_matrix(y_test, y_pred).ravel()
#         return {'penalty':results.result().penalty, 
#                            'solver':results.result().solver, 
#                            'train_score':results.result().score(X_train, y_train), 
#                            'test_score': results.result().score(X_test, y_test),
#                            'f1_score': f1_score(y_test, y_pred),
#                            'true_positive_rate':tp/(tp + fp),
#                            'true_negative_rate':tn/(tn+fn),
#                            'l1_ratio':results.result().l1_ratio,
#                            'C': results.result().C
#                           }
# 
# 
# ```

# #### ARIMA With Log Scale Data

# In[38]:


# Auto ARIMA takes into account the AIC and BIC values generated 
#     to determine the best combination of parameters: 
#     AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion).
# AIC uses the number of independent variables and 
#     the maximum likelihood estimate of the model to determine best fit.
#      The best-fit model is the one that explains the greatest amount of variation 
#      using the fewest possible independent variables.
# BIC has a larger penalty for more independent variables and involves
#      the number of observations in the formula
results = auto_arima(indexedDataset_logScale, trace=True, suppress_warnings=True)
results.summary()


# In[39]:


model = ARIMA(indexedDataset_logScale, order = (4 , 1, 3)) #ARIMA
results = model.fit()
# linear function of the differenced observations and residual errors at prior time steps.
# Our model must have a trend as we can see from graphs. 
# Suitable if there are no seasonal components which there probably are in this data. 


# In[40]:


error = results.fittedvalues - indexedDataset_logScale['#Passengers']
errorsq = error**2
rss= errorsq.sum()
rss
#can also use from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(results.fittedvalues, indexedDataset_logScale))
print(rss, rmse)


# In[41]:


plt.plot(indexedDataset_logScale[2:], label='Log Scale Data')
plt.plot(results.fittedvalues[2:], label='Fitted Values',color='red')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Log Scale Number of Airline Passengers')
plt.annotate(f'RMSE is {rmse}', xy =(datetime.strptime("1950-01-01", "%Y-%m-%d"), 500))
plt.title('Log Scale Data & Fitted Values')


# In[42]:


results.summary()
#Jarque-Bera test is a test for normality. 
# A goodness-of-fit test of whether sample data have the skewness and kurtosis 
# matching a normal distribution.
# A normal distribution has a skew of zero and a kurtosis of three.
# The skew is close to zero but the kurtosis is a little less than three.
# Prob(JB) = .57 means that we cannot reject the null the data is normally distributed.
# Ljung-Box: Test to see if any of a group of autocorrelations of a time series 
#            are different from zero.
# With a Prob(Q) of .4 we say that we cannot reject the null that there is NOT a lack of fit. 


# In[43]:


indexedDataset_logScale[2:].head()


# In[44]:


plt.plot(indexedDataset_logScale[2:], label='# of Airline Passengers')
plt.plot(results.fittedvalues[2:], color='red', label='ARIMA Fitted Values')
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Log Scale Number of Airline Passengers')
plt.annotate(f'RMSE is {rmse}', xy =(datetime.strptime("1950-01-01", "%Y-%m-%d"), 6))
plt.title('Fitted versus Original Values')


# In[45]:


preds = results.forecast(steps=20)
preds


# In[46]:


#how to append one series on another
predictionsLog = pd.Series(indexedDataset_logScale['#Passengers'], index = indexedDataset_logScale.index)
predictionsLog = predictionsLog.add(preds, fill_value = 0)
predictionsLog


# In[47]:


predictions = np.exp(predictionsLog)
predictions #The data is now in its original form with predictions appended


# In[48]:


# Due to seasonality. SARIMA would be a better model
plt.plot(predictions[:-19], label='indexedDataset_logScale')
plt.plot(predictions[-20:], label='Predictions')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Number of Air Passengers')
plt.title('ARIMA Log Scale Data')
plt.annotate(f'RMSE is {rmse}', xy =(datetime.strptime("1950-01-01", "%Y-%m-%d"), 500))


# #### ARMA with log scale minus moving average

# In[49]:


results = auto_arima(LogScaleMinusMovingAverage, trace=True, suppress_warnings=True)
results.summary()


# In[50]:


model = ARIMA(LogScaleMinusMovingAverage, order = (3 , 0, 2)) #ARMA with no integration.
results = model.fit()
#ARMA: linear function of the observations and residual errors at prior time steps.
#Suitable for data without trend or seasonal components.
# Subtracting the moving average changed the model choice to ARMA


# In[51]:


results.summary()


# In[52]:


results.fittedvalues


# In[53]:


rmse = sqrt(mean_squared_error(results.fittedvalues, LogScaleMinusMovingAverage))
rmse


# In[54]:


LogScaleMinusMovingAverage.head()


# In[55]:


plt.plot(LogScaleMinusMovingAverage, label='# of Airline Passengers')
plt.plot(results.fittedvalues, color='red', label='ARIMA Fitted Values')
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Log Scale Minus Moving Average Number of Passengers')
plt.annotate(f'RMSE is {rmse}', xy =(datetime.strptime("1950-01-01", "%Y-%m-%d"), -.2))
plt.title('ARMA Log Scale Minus Moving Average: Fitted versus Original Values')


# In[56]:


preds = results.forecast(steps=20)
preds #Preds from 1961 - 1962


# In[57]:


#how to append one series on another
predictionsLog = pd.Series(LogScaleMinusMovingAverage['#Passengers'], 
                           index = LogScaleMinusMovingAverage.index)
predictionsLog = predictionsLog.add(preds, fill_value = 0)
predDf = pd.DataFrame(predictionsLog, columns = ['preds'])


# In[58]:


predDf['movingAverage'] = movingAverage


# In[59]:


predDf['movingAverage'].fillna(method='ffill', inplace=True) 
#Forward filling the last movingAverage for addition into predictions


# In[60]:


predDf.fillna(method='ffill', inplace=True) 
#Forward filling the last movingAverage for addition into predictions


# In[61]:


predDf['undifferencing'] = predDf['preds'] + predDf['movingAverage']
predDf['expon'] = np.exp(predDf['undifferencing'])
predDf #The data is now in its original form with predictions appended


# In[62]:


#plt.plot(indexedDataset)
plt.plot(predDf['expon'][:-19], label='Log Scale Minus Moving Average')
plt.plot(predDf['expon'][-20:], label='Predictions')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Log Scale Number of Air Passengers')
plt.title('ARMA With Log Scale Minus Moving Average')
plt.annotate(f'RMSE is {rmse}', xy =(datetime.strptime("1950-01-01", "%Y-%m-%d"), 500))


# #### ARIMA With Original Data

# In[63]:


results = auto_arima(indexedDataset, trace=True, suppress_warnings=True)
results.summary()


# In[64]:


model = ARIMA(indexedDataset, order = (4 , 1, 3)) #ARIMA model
results = model.fit()
#Our data has a trend and looks like seasonal components as well.
# SARIMA may be best


# In[65]:


results.summary()


# In[66]:


results.fittedvalues


# In[67]:


rmse = sqrt(mean_squared_error(results.fittedvalues, indexedDataset))
rmse


# In[68]:


indexedDataset.head()


# In[69]:


plt.plot(indexedDataset[1:], label='# of Airline Passengers')
plt.plot(results.fittedvalues[1:], color='red', label='ARIMA Fitted Values')
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('# of Air Passengers')
plt.annotate(f'RMSE is {rmse}', xy =(datetime.strptime("1950-01-01", "%Y-%m-%d"), 500))
plt.title('ARIMA Original Data: Fitted versus Original Values')


# In[70]:


preds = results.forecast(steps=20)
preds #Preds from 1961 - 1962


# In[71]:


data = pd.Series(indexedDataset['#Passengers'], index = indexedDataset.index)
predictions = data.add(preds, fill_value = 0)
predictions


# In[72]:


plt.plot(predictions[:-19], label='Original Data')
plt.plot(predictions[-20:], label='Predictions')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Number of Air Passengers')
plt.title('ARIMA With Original Data')
plt.annotate(f'RMSE is {rmse}', xy =(datetime.strptime("1950-01-01", "%Y-%m-%d"), 500))


# #### Seasonal Differencing

# In[73]:


d = indexedDataset_logScale.diff(12).dropna()
plot_acf(d)


# In[74]:


d = indexedDataset_logScale.diff(12).dropna()
plot_pacf(d)


# In[ ]:




