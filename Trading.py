# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 19:31:52 2020

@author: dkhoo
"""
import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
#import tensorflow as tf
#import keras
#from sklearn.preprocessing import MinMaxScaler

class startUp:
    
    def __init__(self, folderPath = 'stock'):
        self.folderPath = folderPath
        self.dataPath = os.path.join(folderPath, 'stockdata')
        self.tickerCols = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.sectorData = pd.read_csv(os.path.join(self.folderPath, 'sector.csv'), names = ['Name', 'Ticker', 'Sector'])
        self.listDataFrames = []
    
        for subfolders in os.listdir(self.dataPath):
            for files in os.listdir(os.path.join(self.dataPath, subfolders)):
                currentData = pd.read_csv(os.path.join(self.dataPath, subfolders, files), names = self.tickerCols)
                currentData['Date'] = pd.to_datetime(currentData['Date'].astype(str), format='%Y%m%d')
                self.listDataFrames.append(currentData)
    
        self.tickerData = pd.concat(self.listDataFrames, axis=0, ignore_index=True)
    
    def getSectorData(self):
        return self.sectorData

    def getTickerData(self):
        return self.tickerData
    
    def getName(self, ticker):
        temp = self.getSectorData().loc[self.getSectorData()['Ticker'] == ticker]
        temp.set_index('Name', inplace=True)
        return temp.index[0]
        
    def getSectors(self):
        sectors = []
        
        for i in self.getSectorData().get('Sector'):
            if i not in sectors:
                sectors.append(i)
                
        return sectors

    def viewSectors(self):
        for i in self.divideList(self.getSectors(), 3):
            print(i)

    def sectorListLower(self):
        sectors = []
        for i in self.getSectors():
            if i.lower() not in sectors:
                sectors.append(i.lower())
        return sectors

    def divideList(self, myList, n):
        for i in range(0, len(myList), n):
            yield myList[i:i + n]

class StockPrice:
    
    def __init__(self, ticker, startUp=startUp(), days=None):
        self.ticker = ticker.upper()
        self.base = startUp
        self.days = days
        self.allData = startUp.getTickerData()
        self.priceData = self.allData.loc[self.allData['Ticker'] == self.ticker]
        if self.days is not None:
            self.changeData(self.days)
        
    def getTicker(self):
        return self.getTicker
    
    def getDays(self):
        return self.days
    
    def getData(self):
        return self.priceData
    
    def changeData(self, numberOfDays):
        newDate = self.getData()['Date'].max() - timedelta(days=numberOfDays)
        self.priceData = self.priceData.loc[self.priceData['Date'] >= newDate]
        
    def plotY(self, y, priceType = 'Close', colour = None, lbl=None):
        x = matplotlib.dates.date2num(self.getData().loc[:, 'Date'])
        plt.plot(x, y, color = colour, label = lbl)
        
    def linear(self, priceType = 'Close'):
        x = matplotlib.dates.date2num(self.getData().loc[:, 'Date'])
        s, i, r, p, e = stats.linregress(x, self.getData().loc[:, priceType])
        print("R^2:", round(r, 2))
        print("Linear Minumum: $", round(s*max(x)+i, 2))
        print("Linear Median: $", round(s*x[round(len(x)/2)]+i, 2))
        print("Linear Maximum: $", round(s*min(x)+i, 2))
        self.plotY(s*x+i, colour='red', lbl = 'Linear')
        
    def sma(self, priceType = 'Close'):
        if priceType.lower() == 'open':
            return self.getData().Open.rolling(window=round(len(self.getData())/4)).mean()
        elif priceType.lower() == 'low':
            return self.getData().Low.rolling(window=round(len(self.getData())/4)).mean()
        elif priceType.lower() == 'high':
            return self.getData().High.rolling(window=round(len(self.getData())/4)).mean()
        return self.getData().Close.rolling(window=round(len(self.getData())/4)).mean()
        
    def summary(self, priceType='Close'):
        start = datetime.strftime(self.getData()['Date'].min(), '%d/%m/%Y')
        end = datetime.strftime(self.getData()['Date'].max(), '%d/%m/%Y')
        print(self.ticker + " | " + self.base.getName(self.ticker))
        print("Dates: " + start + " - " + end)
        print("No. of days:", (self.getData()['Date'].max() - self.getData()['Date'].min()).days)
        print("SMA Days Ahead: " + str(round(len(self.getData()))/4) + " " + datetime.strftime(self.getData()['Date'].min() \
                                  + timedelta(days=round(len(self.getData()))/4), '%d/%m/%Y'))
        print("Minimum: $", self.getData()[priceType].min())
        print("Maximum: $", self.getData()[priceType].max())
        plt.plot_date(self.getData().loc[:, 'Date'], self.getData().loc[:, priceType], \
                      color = 'purple', markersize=3)
        self.linear(priceType)
        print("SMA Minimum: $", round(self.sma(priceType).min(), 2))
        print("SMA Maximum: $", round(self.sma(priceType).max(), 2))
        self.plotY(self.sma(priceType), colour = 'green', lbl = 'SMA')
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.show()
        
        """
    def forecast(self):
        closeData = self.getData()['Close']
        data = closeData.to_numpy()

        trainData = data[:len(test)/2]
        testData = data[len(test)/2:]        
        
        scaler = MinMaxScaler()
        trainData = trainData.reshape(-1, 1)
        testData = testData.reshape(-1, 1)
        
        smoothing = 4
        for di in range(0, len(closeData), 4):
            scaler.fit(trainData[di:di+smoothing,:])
            trainData[di:di+smoothing,:] = scaler.transform(trainData[di:di+smoothing,:])
            
        scaler.fit(trainData[di+smoothing:,:])
        trainData[di+smoothing:,:] = scaler.transform(trainData[di+smoothing:,:])
        
        trainData = trainData.reshape(-1)
        testData = scaler.transform(testData).reshape(-1)
        
        EMA = 0.0
        gamma = 0.1
        for ti in range(len(data)):
            EMA = gamma*trainData[ti] + (1-gamma)*EMA
            trainData[ti] = EMA
            
        newData = np.concatenate([trainData, testData], axis = 0)
        
        D = 1
        rolling = 30
        batch = 10
        nodes = [100,100,50]
        layers = len(nodes)
        dropout = 0.2
        
        tf.reset_default_graph()
        
        trainInputs, trainOuputs = [], []
        
        for u in range(rolling):
            trainInputs.append(tf.placeholder(tf.float32, shape=[batch], D, name = 'trainInputs_%d'%ui))
            trainOuputs.append(tf.placeholder(tf.float32, shape = [batch, 1], name = 'trainOuputs_%d'%ui))
        
        lstm_cells = [
            tf.contrib.rnn.LSTMCell(num_units=nodes[li],
                                    state_is_tuple = True,
                                    initializer = tf.contrib.layer.xavier_initializer()
                                    )
            for li in range(layers)]
        
        dropLSTM = [tf.contrib.rnn.DropoutWrapper(
            lstm, input_keep_prob = 1.0, output_keep_prob = 1.0-dropout, state_keep_prob = 1-dropout)
            for lstm in lstm_cells
            ]
        
        drop_multi = tf.contrib.rnn.MultiRnnCell(dropLSTM)
        multi = tf.contrib.rnn.MultiRNNCell(LSTM)
        w = tf.get_variable('w', shape = [nodes[-1], 1], initializer = tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', initializer=tf.random_uniform([1], -0.1, 0.1))        
        
        c, h = [], []
        initial = []
        for li in range(layers):
            c.append(tf.Variable(tf.zeros([batch, nodes[li]]), trainable=False))
            h.append(tf.Variable(tf.zeros([batch, nodes[li]]), trainable=False))
            initial.append(tf.contrib.rnn.LSTMStateTuple(c[li], h[li]))
            
        allInputs = tf.concat([tf.expand_dims(t,0) for t in train_inputs],axis=0)
        
        allLSTMOutputs, state = tf.nn.dynamic_rnn(
            drop_multi_cell, allInputs, initial_state=tuple(initial),
            time_major = True, dtype=tf.float32)

        allLSTMOutputs = = tf.reshape(allLSTMOutputs, [batch*rolling, nodes[-1]])
        
        allOutputs = tf.nn.xw_plus_b(allLSTMOutputs, w, b)
        
        splitOutputs = tf.split(allOutputs, rollings, axis=0)
        """
        
class SectorProfile:
    
    def __init__(self, sector, startUp = startUp(), numberOfDays = None):
        self.sector = sector
        self.days = numberOfDays
        self.base = startUp
        
    def getSector(self):
        return self.sector
    
    def getDays(self):
        return self.days
    
    def getBase(self):
        return self.base
    
    def getTickers(self):
        data = self.getBase().getSectorData()
        tickers = data.loc[data['Sector'].str.lower() == self.getSector().lower()]
        return list(tickers['Ticker'])
        
    def summary(self):
        for i in self.getTickers():
            if self.getDays() is None:
                StockPrice(i).summary()
            else:
                StockPrice(i, self.getBase(), self.getDays()).summary()
                

def main():
    base = startUp()
    print("Welcome, press q at anytime to go back or quit")
    while True:
        mode = input("Ticker or Sector (word or t or s):")
        if mode.lower() == 'q':
            break
        elif mode.lower() == 'test':
            StockPrice("STO", base).summary()
            
        elif mode.lower() == 't' or mode.lower() == 'ticker':
            while True:
                ticker = input("Comnpany Ticker:").upper()
                if ticker.lower() == 'q':
                    break
                elif ticker in base.getSectorData().get('Ticker').tolist():
                    while True:
                        is_int = None
                        days = input("How many days of data do you want to use (Any non-integer input will result in using all data):")
                        if days.lower() == 'q':
                            break
                        else:
                            try:
                                int(days)
                                is_int = True
                            except ValueError:
                                is_int = False
                            
                            if is_int:
                                StockPrice(ticker, base, days=int(days)).summary()
                            else:
                                StockPrice(ticker, base).summary()
                else:
                    print("Invalid ticker, try again..")
                    continue
                                
        elif mode.lower() == 's' or mode.lower() == 'sector':
            base.viewSectors()
            while True:
                sector = input("Desired sector:")
                if sector.lower() == 'q':
                    break
                
                elif sector.lower() in base.sectorListLower():
                    while True:
                        is_int = None
                        days = input("How many days of data do you want to use (Any non-integer input will result in using all data):")
                        if days.lower() == 'q':
                            break
                        else:
                            try:
                                int(days)
                                is_int = True
                            except ValueError:
                                is_int = False
                                
                            if is_int:
                                SectorProfile(sector, base, int(days)).summary()
                            else:
                                SectorProfile(sector, base).summary()
                else:
                    print("Invalid sector, try again..")
                    continue
        else:
            print("Invalid input, try again...")
            continue
        
if __name__ == "__main__":
  main()