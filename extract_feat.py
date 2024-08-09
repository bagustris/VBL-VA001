# Script to train

# importation
from numpy.fft import rfft, rfftfreq
from sklearn import preprocessing
import numpy as np
import os
import pandas as pd
import glob

data_path = '/data/VBL-VA001/'

totalFiles = 0
totalDir = 0

for base, dirs, files in os.walk(data_path):
    print('Searching in : ', base)
    for directories in dirs:
        totalDir += 1
    for Files in files:
        totalFiles += 1

print('Total number of files', totalFiles)
print('Total number of directories', totalDir)

# Collecting number data
dir_path1 = data_path + '/normal/'
print('Total data Normal :', len([entry for entry in os.listdir(
    dir_path1) if os.path.isfile(os.path.join(dir_path1, entry))]))
dir_path2 = data_path + '/misalignment/'
print('Total data misalignment :', len([entry for entry in os.listdir(
    dir_path2) if os.path.isfile(os.path.join(dir_path2, entry))]))
dir_path3 = data_path + '/unbalance'
print('Total data unbalance :', len([entry for entry in os.listdir(
    dir_path3) if os.path.isfile(os.path.join(dir_path3, entry))]))
dir_path4 = data_path + '/bearing'
print('Total data bearing fault:', len([entry for entry in os.listdir(
    dir_path4) if os.path.isfile(os.path.join(dir_path4, entry))]))

# Collecting file names
normal = glob.glob(data_path + '/normal/*.csv')
misalignment = glob.glob(data_path + '/misalignment/*.csv')
unbalance = glob.glob(data_path + '/unbalance/*.csv')
bearing = glob.glob(data_path + '/bearing/*.csv')


def FFT(data):
    '''FFT process, take real values only'''
    data = np.asarray(data)
    n = len(data)
    dt = 1/20000  # time increment in each data
    data = rfft(data)*dt
    freq = rfftfreq(n, dt)
    data = abs(data)
    return data

# Feature Extraction function
def std(data):
    '''Standard Deviation features'''
    data = np.asarray(data)
    stdev = pd.DataFrame(np.std(data, axis=1))
    return stdev


def mean(data):
    '''Mean features'''
    data = np.asarray(data)
    M = pd.DataFrame(np.mean(data, axis=1))
    return M


def pp(data):
    '''Peak-to-Peak features'''
    data = np.asarray(data)
    PP = pd.DataFrame(np.max(data, axis=1) - np.min(data, axis=1))
    return PP


def Variance(data):
    '''Variance features'''
    data = np.asarray(data)
    Var = pd.DataFrame(np.var(data, axis=1))
    return Var


def rms(data):
    '''RMS features'''
    data = np.asarray(data)
    Rms = pd.DataFrame(np.sqrt(np.mean(data**2, axis=1)))
    return Rms


def Shapef(data):
    '''Shape factor features'''
    data = np.asarray(data)
    shapef = pd.DataFrame(rms(data)/Ab_mean(data))
    return shapef


def Impulsef(data):
    '''Impulse factor features'''
    data = np.asarray(data)
    impulse = pd.DataFrame(np.max(data)/Ab_mean(data))
    return impulse


def crestf(data):
    '''Crest factor features'''
    data = np.asarray(data)
    crest = pd.DataFrame(np.max(data)/rms(data))
    return crest


def kurtosis(data):
    '''Kurtosis features'''
    data = pd.DataFrame(data)
    kurt = data.kurt(axis=1)
    return kurt


def skew(data):
    '''Skewness features'''
    data = pd.DataFrame(data)
    skw = data.skew(axis=1)
    return skw


# Helper functions to calculate features
def Ab_mean(data):
    data = np.asarray(data)
    Abm = pd.DataFrame(np.mean(np.absolute(data), axis=1))
    return Abm


def SQRT_AMPL(data):
    data = np.asarray(data)
    SQRTA = pd.DataFrame((np.mean(np.sqrt(np.absolute(data, axis=1))))**2)
    return SQRTA


def clearancef(data):
    data = np.asarray(data)
    clrf = pd.DataFrame(np.max(data, axis=1)/SQRT_AMPL(data))
    return clrf


# Extract features from X, Y, Z axis
def read_data(filenames):
    data = pd.DataFrame()
    for filename in filenames:
        df = pd.read_csv(filename, usecols=[1], header=None)
        data = pd.concat([data, df], axis=1, ignore_index=True)
    return data

# read data from csv files
all_cond = [normal, misalignment, unbalance, bearing]
cond_names = ['normal', 'misalignment', 'unbalance', 'bearing']
data = {}
fft = {}

for cond, cond_name in zip(all_cond, cond_names):
    for ax in ['x', 'y', 'z']:
        name = f"{cond_name}_{ax}"
        data[name] = read_data(cond).T.dropna(axis=1)
        fft[name] = FFT(data[name])

# fft_merged = pd.concat(fft, axis=1)

# Find max and min value of fft
max_value = max(fft.values(), key=lambda item: max(max(sub_array) for sub_array in item))
MAX_FFT = max(max(sub_array) for sub_array in max_value)
min_value = min(fft.values(), key=lambda item: min(min(sub_array) for sub_array in item))
MIN_FFT = min(min(sub_array) for sub_array in min_value)

def NormalizeData(**kwargs):  # Normalisasi (0-1)
    return (data - MIN_FFT) / (MAX_FFT - MIN_FFT)

