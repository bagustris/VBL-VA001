from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pickle
import pandas as pd
import numpy as np
import statistics
import os
from numpy.fft import rfft, rfftfreq

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input csv file")
args = parser.parse_args()
data_csv = pd.read_csv(args.input, header=None)

# data_csv = argparse.ArgumentParser

def FFT(data):
    ''' Function to convert waveform to spectrum '''
    data = np.asarray(data)
    n=len(data)
    dt=1/20000 #time increment in each data
    data=rfft(data)*dt
    freq=rfftfreq(n,dt)
    data=abs(data)
    return data


import scipy
from scipy.stats import kurtosis
from scipy.stats import skew

def std(data):
    data = np.asarray(data)
    stdev=pd.DataFrame(np.std(data, axis=1))
    return stdev
def mean(data):
    data = np.asarray(data)
    M=pd.DataFrame(np.mean(data, axis=1))
    return M
def pp(data):
    data = np.asarray(data)
    PP=pd.DataFrame(np.max(data, axis=1) - np.min(data, axis=1))
    return PP
def Variance(data):
    data = np.asarray(data)
    Var=pd.DataFrame(np.var(data, axis=1))
    return Var
def rms(data):
    data = np.asarray(data)
    Rms=pd.DataFrame(np.sqrt(np.mean(data**2, axis=1)))
    return Rms
def Ab_mean(data):
    data = np.asarray(data)
    Abm=pd.DataFrame(np.mean(np.absolute(data),axis=1))
    return Abm
def Shapef(data):
    data = np.asarray(data)
    shapef=pd.DataFrame(rms(data)/Ab_mean(data))
    return shapef
def Impulsef(data):
    data = np.asarray(data)
    impulse=pd.DataFrame(np.max(data)/Ab_mean(data))
    return impulse
def crestf(data):
    data = np.asarray(data)
    crest=pd.DataFrame(np.max(data)/rms(data))
    return crest
def SQRT_AMPL(data):
    data = np.asarray(data)
    SQRTA=pd.DataFrame((np.mean(np.sqrt(np.absolute(data, axis=1))))**2)
    return SQRTA
def clearancef(data):
    data = np.asarray(data)
    clrf=pd.DataFrame(np.max(data, axis=1)/SQRT_AMPL(data))
    return clrf
def kurtosis(data):
    data = pd.DataFrame(data);
    kurt = data.kurt(axis=1);
    return kurt
def skew(data):
    data = pd.DataFrame(data)
    skw = data.skew(axis=1)
    return skw



#test sumbu x
test_x = pd.DataFrame(data_csv)
test_x.drop(test_x.columns[[0,2,3]], axis=1, inplace=True) #hapus kolom 0,2,3
test_x = test_x.T
test_x = test_x.dropna(axis=1)

#test sumbu y
test_y = pd.DataFrame(data_csv)
test_y.drop(test_y.columns[[0,1,3]], axis=1, inplace=True) #hapus kolom 0,1,3
test_y = test_y.T
test_y = test_y.dropna(axis=1)

#test sumbu z
test_z = pd.DataFrame(data_csv)
test_z.drop(test_z.columns[[0,1,2]], axis=1, inplace=True) #hapus kolom 0,1,2
test_z = test_z.T
test_z = test_z.dropna(axis=1)


# FFT
fft_test_x = FFT(test_x)
fft_test_y = FFT(test_y)
fft_test_z = FFT(test_z)


# EKSTRAKSI FITUR
Shapef_x = Shapef(fft_test_x)
Shapef_y = Shapef(fft_test_y)
Shapef_z = Shapef(fft_test_z)
Shapef_test = pd.concat([Shapef_x,Shapef_y,Shapef_z], axis=1,ignore_index=True)

rms_x = rms(fft_test_x)
rms_y = rms(fft_test_y)
rms_z = rms(fft_test_z)
rms_test = pd.concat([rms_x,rms_y,rms_z], axis=1,ignore_index=True)

Impulsef_x = Impulsef(fft_test_x)
Impulsef_y = Impulsef(fft_test_y)
Impulsef_z = Impulsef(fft_test_z)
Impulsef_test = pd.concat([Impulsef_x,Impulsef_y,Impulsef_z], axis=1,ignore_index=True)

pp_x = pp(fft_test_x)
pp_y = pp(fft_test_y)
pp_z = pp(fft_test_z)
pp_test = pd.concat([pp_x,pp_y,pp_z], axis=1,ignore_index=True)

kurtosis_x = kurtosis(fft_test_x)
kurtosis_y = kurtosis(fft_test_y)
kurtosis_z = kurtosis(fft_test_z)
kurtosis_test = pd.concat([kurtosis_x,kurtosis_y,kurtosis_z], axis=1,ignore_index=True)

crestf_x = crestf(fft_test_x)
crestf_y = crestf(fft_test_y)
crestf_z = crestf(fft_test_z)
crestf_test = pd.concat([crestf_x,crestf_y,crestf_z], axis=1,ignore_index=True)

mean_x = mean(fft_test_x)
mean_y = mean(fft_test_y)
mean_z = mean(fft_test_z)
mean_test = pd.concat([mean_x,mean_y,mean_z], axis=1,ignore_index=True)

std_x = std(fft_test_x)
std_y = std(fft_test_y)
std_z = std(fft_test_z)
std_test = pd.concat([std_x,std_y,std_z], axis=1,ignore_index=True)

skew_x = skew(fft_test_x)
skew_y = skew(fft_test_y)
skew_z = skew(fft_test_z)
skew_test = pd.concat([skew_x,skew_y,skew_z], axis=1,ignore_index=True)

data_test = pd.concat([mean_test,std_test,Shapef_test,rms_test,Impulsef_test,pp_test,kurtosis_test,crestf_test,skew_test], axis=1,ignore_index=True)

# print(f"Mean feature shape: {mean_test.shape}")
print(f"Total feature shape (x, y, z): {data_test.shape}")
