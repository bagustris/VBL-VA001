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

# plot all nine features
feat_name = ['Shape Factor', 'RMS', 'Impulse Factor', 'Peak to Peak', 'Kurtosis', 'Crest Factor', 'Mean', 'Standard Deviation', 'Skewness']
feat_n = np.arange(2, 27, 3)
fig, ax = plt.subplots(3, 3)
fig.set_size_inches(8, 6)

axs = ax.ravel()
for i, n in enumerate(feat_n):
    y1 = x_norm.iloc[:, n]
    y2 = x_mis.iloc[:, n]
    y3 = x_unb.iloc[:, n]
    y4 = x_bear.iloc[:, n]

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

    # print(f"i = {i}, n = {n}")
    axs[i].plot(x, y11, x, y22, x, y33, x, y44)
    # Decorate
    axs[i].set_title(feat_name[i])
    axs[i].set_xlim(20,980)
    if i <= 5:
        # axs[i].get_xaxis().set_visible(False)
        axs[i].axes.xaxis.set_ticklabels([])

plt.show()
# plt.savefig('feature_z_axis.pdf')
# save manually for better quality, then crop