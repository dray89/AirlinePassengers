#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 14:40:04 2025

@author: raydebra89
"""
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

# Previously, I wanted to show each step in the process. 
# But, I can also put the code in functions like so.
# I will use these going forward to avoid repetitive code
 
def plot_stationarity(timeseries):
    movingAverage = timeseries.rolling(window=12).mean()
    movingStd = timeseries.rolling(window=12).std()
    
    #plot rolling statistics
    orig = plt.plot(timeseries, color = 'blue', label = 'Number of Airline Passengers')
    mean = plt.plot(movingAverage, color = 'red', label = 'Rolling Mean')
    std = plt.plot(movingStd, color = 'black', label = 'Rolling Standard Deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
def test_stationarity(timeseries):
    Dickey_Fuller_Test = adfuller(timeseries['#Passengers'], autolag='AIC')

    Dickey_Fuller_Output = pd.Series(Dickey_Fuller_Test, index = 
          ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used', 'Test Statistics', 'Maximized Information Criterion'])

    for key, value in Dickey_Fuller_Test[4].items():
        Dickey_Fuller_Output['Critical Value (%s)'%key] = value #Unpacking embeded critical value dictionary

    Dickey_Fuller_Output = Dickey_Fuller_Output.drop('Test Statistics')
    return Dickey_Fuller_Output

def kpss_test(timeseries):
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','#Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    return kpss_output