# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Import the library
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt

# Data Manupulation
import numpy as np
import pandas as pd
import talib as tb
from datetime import date

# Plotting graphs
import matplotlib.pyplot as plt

# Machine learning
from sklearn.linear_model import LogisticRegression




#Load the stock price from yahoo.finance 
def load_data(code,start_date,end_date):
    sp=yf.download(code,start_date,end_date)
    return sp
#Input the ticker, start date, and end date
sp=load_data('AMAZ','2015-02-01','2018-04-01')
sp = sp.dropna()
sp = sp.iloc[:,:4]
print(sp)

#define the moveing average 
def MA(timeperiod):
    MA=tb.SMA(sp.Close,timeperiod)
    return MA

#get the moving average 
MA50=MA(50)
MA200=MA(200)

# plot the 50-day moveing average and plot the 200-day moveing average
plt.plot(sp.Close)
plt.plot(MA50,label='MA50')


plt.plot(MA200,label='MA200')
plt.grid()
plt.legend()
plt.show()

# Define dependent variables 
def spclosecorr(timeperiod):
    closecorr=sp["Close"].rolling(timeperiod).corr(MA(timeperiod))
    return closecorr
def spRSI(timeperiod):
    spRSI= tb.RSI(np.array(sp['Close']),timeperiod)
    return spRSI

#
MA10=MA(10)
cc=spclosecorr(10)
rsi=spRSI(10)

#define the independent variables
sp["10-day Moving average"]=MA10
sp["Correlation"]=cc
sp["RSI"]=rsi
sp['Open-Close'] = sp['Open'] - sp['Close'].shift(1)
sp['Open-Open'] = sp['Open'] - sp['Open'].shift(1)

#get the date
sp = sp.dropna()
X = sp.iloc[:,:9]
print(X)


#define the target variable
y = np.where (sp['Close'].shift(-1) > sp['Close'],1,-1)

#Split the dataset(define the function)
split = int(0.7*len(sp))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

#build the model 
model=LogisticRegression()
model=model.fit(X_train,y_train)

#
sp['Predicted_Signal'] = model.predict(X)
sp['AMAZ_returns'] = np.log(sp['Close']/sp['Close'].shift(1))
Cumulative_AMAZ_returns = np.cumsum(sp[split:]['AMAZ_returns'])
sp['Startegy_returns'] = sp['AMAZ_returns']* sp['Predicted_Signal'].shift(1)
Cumulative_Strategy_returns = np.cumsum(sp[split:]['Startegy_returns'])
plt.figure(figsize=(10,5))
plt.plot(Cumulative_AMAZ_returns, color='r',label = 'AMAZ Returns')
plt.plot(Cumulative_Strategy_returns, color='g', label = 'Strategy Returns')
plt.legend()
plt.show()

