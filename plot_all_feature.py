# code to plot all feature

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
x = pd.read_csv("feature_VBL-VA001.csv", header=None)

# extract normal condition, 1st 1000 rows
x_norm = x.iloc[:1000, :]

# extract misalignment condition, next 1000 rows
x_mis = x.iloc[1000:2000, :]

# extract unbalance condition, next 1000 rows
x_unb = x.iloc[2000:3000, :]

# extract bearing condition, last 1000 rows
x_bear = x.iloc[3000:4000, :]

# plot skewness for z-axis
y1 = x_norm.iloc[:, -1]
y2 = x_mis.iloc[:, -1]
y3 = x_unb.iloc[:, -1]
y4 = x_bear.iloc[:, -1]

y1 = y1.values.flatten()
y2 = y2.values.flatten()
y3 = y3.values.flatten()
y4 = y4.values.flatten()

# helper for x axis
x = np.arange(0,len(y1),1)

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

y11 = movingaverage(y1, 30)
y22 = movingaverage(y2, 30)
y33 = movingaverage(y3, 30)
y44 = movingaverage(y4, 30)

#y = plt.scatter(x,y1,s=5,color='cyan',label ='Normal')
plt.plot(x, y11, x, y22, x, y33, x, y44)



plt.rcParams.update({'figure.figsize':(6,4), 'figure.dpi':100, 'legend.loc':'upper right'})

# Decorate
plt.title('Skewness')
plt.xlabel('Data')
plt.ylabel('skew (z-axis)')
plt.legend(['Normal', 'Misalignment', 'Unbalance', 'Bearing'], loc='right')
plt.xlim(20,980)
plt.show()

