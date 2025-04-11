#!python -m pip install matplotlib
#!python -m pip install pandas
#!python -m pip install scikit-learn

from os import listdir
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy import stats
from sklearn.decomposition import PCA

from scipy.signal import butter,filtfilt
# Filter requirements.
T = 1         # Sample Period
fs = 200       # sample rate, Hz
cutoff = 20      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 4       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Enable interactive matplotlib plots
# %matplotlib notebook
# %matplotlib inline

# Print versions
!python --version
print('Numpy ' + np.__version__)
print('Pandas ' + pd.__version__)

# Settings
dataset_path = 'foz_dataset'  # Directory where raw accelerometer data is stored
normal_op_list = ['normal-idle'] # can add 'normal2' to include more normal data
anomaly_op_list = ['misfire-idle','misfire2']
sample_rate = 200       # Hz
sample_time = 0.5       # Time (sec) length of each sample
samples_per_file = 200 # Expected number of measurements in each file
max_measurements = int(sample_time * sample_rate)

print('Max measurements per file:', max_measurements)

# Create list of filenames
def createFilenameList(op_list):
    
    # Extract paths and filenames in each directory
    op_filenames = []
    num_samples = 0
    for index, target in enumerate(op_list):
        samples_in_dir = listdir(join(dataset_path, target))
        samples_in_dir = [join(dataset_path, target, sample) for sample in samples_in_dir]
        op_filenames.append(samples_in_dir)
    
    # Flatten list
    return [item for sublist in op_filenames for item in sublist]

# Create normal and anomaly filename lists
normal_op_filenames = createFilenameList(normal_op_list)
anomaly_op_filenames = createFilenameList(anomaly_op_list)
print('Number of normal samples:', len(normal_op_filenames))
print('Number of anomaly samples:', len(anomaly_op_filenames))

# Function to plot normal vs anomaly samples side-by-side
def plotTimeSeriesSample(normal_sample, anomaly_sample):
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    fig.tight_layout(pad=3.0)
    axs[0].plot(np.abs(butter_lowpass_filter(normal_sample.T[0], cutoff, fs, order)), label='x')
    axs[0].plot(np.abs(butter_lowpass_filter(normal_sample.T[1], cutoff, fs, order)), label='y')
    axs[0].plot(np.abs(butter_lowpass_filter(normal_sample.T[2], cutoff, fs, order)), label='z')
    axs[0].set_title('Normal sample')
    axs[0].set_xlabel('sample')
    axs[0].set_ylabel('G-force')
    axs[0].legend()
    axs[1].plot(np.abs(butter_lowpass_filter(anomaly_sample.T[0], cutoff, fs, order)), label='x')
    axs[1].plot(np.abs(butter_lowpass_filter(anomaly_sample.T[1], cutoff, fs, order)), label='y')
    axs[1].plot(np.abs(butter_lowpass_filter(anomaly_sample.T[2], cutoff, fs, order)), label='z')
    axs[1].set_title('Anomaly sample')
    axs[1].set_xlabel('sample')
    axs[1].set_ylabel('G-force')
    axs[1].legend()
    
# Function to plot 3D scatterplot of normal and anomaly smaples
def plotScatterSamples(normal_samples, anomaly_samples, num_samples, title=''):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(num_samples):
        ax.scatter(normal_samples[i].T[0], normal_samples[i].T[1], normal_samples[i].T[2], c='b')
        ax.scatter(anomaly_samples[i].T[0], anomaly_samples[i].T[1], anomaly_samples[i].T[2], c='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    
# Examine a normal sample vs anomalous sample
normal_sample = np.genfromtxt(normal_op_filenames[0], delimiter=',')
anomaly_sample = np.genfromtxt(anomaly_op_filenames[0], delimiter=',')


#normal_sample = butter_lowpass_filter(normal_sample0, cutoff, fs, order)
#anomaly_sample = butter_lowpass_filter(anomaly_sample0, cutoff, fs, order)

# Plot time series of accelerometer data
plotTimeSeriesSample(normal_sample, anomaly_sample)

# Shuffle samples for further analysis
random.shuffle(normal_op_filenames)
random.shuffle(anomaly_op_filenames)

# Make a 3D scatterplot
num_samples = 20
normal_samples = []
anomaly_samples = []
for i in range(num_samples):
    normal_samples.append(np.genfromtxt(normal_op_filenames[i], delimiter=','))
    anomaly_samples.append(np.genfromtxt(anomaly_op_filenames[i], delimiter=','))

#np.abs(butter_lowpass_filter(normal_sample.T[0], cutoff, fs, order))

plotScatterSamples(normal_samples, anomaly_samples, num_samples)

# Let's remove DC to see what it looks like in time domain
normal_sample = np.genfromtxt(normal_op_filenames[0], delimiter=',')
anomaly_sample = np.genfromtxt(anomaly_op_filenames[0], delimiter=',')
normal_sample = normal_sample - np.mean(normal_sample, axis=0)
anomaly_sample = anomaly_sample - np.mean(anomaly_sample, axis=0)

# Plot time series of accelerometer data
plotTimeSeriesSample(normal_sample, anomaly_sample)


# Make a 3D scatterplot with DC removed
num_samples = 10
normal_samples = []
anomaly_samples = []
for i in range(num_samples):
    normal_sample = np.genfromtxt(normal_op_filenames[i], delimiter=',')
    anomaly_sample = np.genfromtxt(anomaly_op_filenames[i], delimiter=',')
    normal_sample = normal_sample - np.mean(normal_sample, axis=0)
    anomaly_sample = anomaly_sample - np.mean(anomaly_sample, axis=0)
    normal_sample.T[0] = np.abs(butter_lowpass_filter(normal_sample.T[0], cutoff, fs, order))
    anomaly_sample.T[0] = np.abs(butter_lowpass_filter(anomaly_sample.T[0], cutoff, fs, order))
    normal_sample.T[1] = np.abs(butter_lowpass_filter(normal_sample.T[1], cutoff, fs, order))
    anomaly_sample.T[1] = np.abs(butter_lowpass_filter(anomaly_sample.T[1], cutoff, fs, order))
    normal_sample.T[2] = np.abs(butter_lowpass_filter(normal_sample.T[2], cutoff, fs, order))
    anomaly_sample.T[2] = np.abs(butter_lowpass_filter(anomaly_sample.T[2], cutoff, fs, order))
    normal_samples.append(normal_sample)
    anomaly_samples.append(anomaly_sample)

print(anomaly_samples[0].shape)
plotScatterSamples(normal_samples, anomaly_samples, num_samples)

# Let's look at various statistics of 1 sample (with DC removed)
idx = 0
normal_sample = np.genfromtxt(normal_op_filenames[idx], delimiter=',')
normal_sample = normal_sample - np.mean(normal_sample, axis=0)

print('Sample shape:', normal_sample.shape)
print('Mean:', np.mean(normal_sample, axis=0))
print('Variance:', np.var(normal_sample, axis=0))
print('Kurtosis:', stats.kurtosis(normal_sample))
print('Skew:', stats.skew(normal_sample))
print('MAD:', stats.median_abs_deviation(normal_sample))
print('Correlation:')
print(np.corrcoef(normal_sample.T))

# Make a 3D scatterplot of means (with DC removed)
num_samples = 70
normal_samples = []
anomaly_samples = []
for i in range(num_samples):
    normal_sample = np.genfromtxt(normal_op_filenames[i], delimiter=',')
    anomaly_sample = np.genfromtxt(anomaly_op_filenames[i], delimiter=',')
    normal_sample = normal_sample - np.mean(normal_sample, axis=0)
    anomaly_sample = anomaly_sample - np.mean(anomaly_sample, axis=0)
    normal_sample.T[0] = np.abs(butter_lowpass_filter(normal_sample.T[0], cutoff, fs, order))
    anomaly_sample.T[0] = np.abs(butter_lowpass_filter(anomaly_sample.T[0], cutoff, fs, order))
    normal_sample.T[1] = np.abs(butter_lowpass_filter(normal_sample.T[1], cutoff, fs, order))
    anomaly_sample.T[1] = np.abs(butter_lowpass_filter(anomaly_sample.T[1], cutoff, fs, order))
    normal_sample.T[2] = np.abs(butter_lowpass_filter(normal_sample.T[2], cutoff, fs, order))
    anomaly_sample.T[2] = np.abs(butter_lowpass_filter(anomaly_sample.T[2], cutoff, fs, order))
    normal_samples.append(np.mean(normal_sample, axis=0))
    anomaly_samples.append(np.mean(anomaly_sample, axis=0))
plotScatterSamples(normal_samples, anomaly_samples, num_samples, title='Means')

# Make a 3D scatterplot of variances
num_samples = 70
normal_samples = []
anomaly_samples = []
for i in range(num_samples):
    normal_sample = np.genfromtxt(normal_op_filenames[i], delimiter=',')
    anomaly_sample = np.genfromtxt(anomaly_op_filenames[i], delimiter=',')
    normal_sample.T[0] = np.abs(butter_lowpass_filter(normal_sample.T[0], cutoff, fs, order))
    anomaly_sample.T[0] = np.abs(butter_lowpass_filter(anomaly_sample.T[0], cutoff, fs, order))
    normal_sample.T[1] = np.abs(butter_lowpass_filter(normal_sample.T[1], cutoff, fs, order))
    anomaly_sample.T[1] = np.abs(butter_lowpass_filter(anomaly_sample.T[1], cutoff, fs, order))
    normal_sample.T[2] = np.abs(butter_lowpass_filter(normal_sample.T[2], cutoff, fs, order))
    anomaly_sample.T[2] = np.abs(butter_lowpass_filter(anomaly_sample.T[2], cutoff, fs, order))
    normal_samples.append(np.var(normal_sample, axis=0))
    anomaly_samples.append(np.var(anomaly_sample, axis=0))
plotScatterSamples(normal_samples, anomaly_samples, num_samples, title='Variances')

# Make a 3D scatterplot of kurtosis
num_samples = 70
normal_samples = []
anomaly_samples = []
for i in range(num_samples):
    normal_sample = np.genfromtxt(normal_op_filenames[i], delimiter=',')
    anomaly_sample = np.genfromtxt(anomaly_op_filenames[i], delimiter=',')
    normal_sample.T[0] = np.abs(butter_lowpass_filter(normal_sample.T[0], cutoff, fs, order))
    anomaly_sample.T[0] = np.abs(butter_lowpass_filter(anomaly_sample.T[0], cutoff, fs, order))
    normal_sample.T[1] = np.abs(butter_lowpass_filter(normal_sample.T[1], cutoff, fs, order))
    anomaly_sample.T[1] = np.abs(butter_lowpass_filter(anomaly_sample.T[1], cutoff, fs, order))
    normal_sample.T[2] = np.abs(butter_lowpass_filter(normal_sample.T[2], cutoff, fs, order))
    anomaly_sample.T[2] = np.abs(butter_lowpass_filter(anomaly_sample.T[2], cutoff, fs, order))
    normal_samples.append(stats.kurtosis(normal_sample))
    anomaly_samples.append(stats.kurtosis(anomaly_sample))
plotScatterSamples(normal_samples, anomaly_samples, num_samples, title='Kurtosis')

# Make a 3D scatterplot of skew
num_samples = 70
normal_samples = []
anomaly_samples = []
for i in range(num_samples):
    normal_sample = np.genfromtxt(normal_op_filenames[i], delimiter=',')
    anomaly_sample = np.genfromtxt(anomaly_op_filenames[i], delimiter=',')
    normal_sample.T[0] = np.abs(butter_lowpass_filter(normal_sample.T[0], cutoff, fs, order))
    anomaly_sample.T[0] = np.abs(butter_lowpass_filter(anomaly_sample.T[0], cutoff, fs, order))
    normal_sample.T[1] = np.abs(butter_lowpass_filter(normal_sample.T[1], cutoff, fs, order))
    anomaly_sample.T[1] = np.abs(butter_lowpass_filter(anomaly_sample.T[1], cutoff, fs, order))
    normal_sample.T[2] = np.abs(butter_lowpass_filter(normal_sample.T[2], cutoff, fs, order))
    anomaly_sample.T[2] = np.abs(butter_lowpass_filter(anomaly_sample.T[2], cutoff, fs, order))
    normal_samples.append(stats.skew(normal_sample))
    anomaly_samples.append(stats.skew(anomaly_sample))
plotScatterSamples(normal_samples, anomaly_samples, num_samples, title='Skew')

# Make a 3D scatterplot of MAD
num_samples = 70
normal_samples = []
anomaly_samples = []
for i in range(num_samples):
    normal_sample = np.genfromtxt(normal_op_filenames[i], delimiter=',')
    anomaly_sample = np.genfromtxt(anomaly_op_filenames[i], delimiter=',')
    normal_sample.T[0] = np.abs(butter_lowpass_filter(normal_sample.T[0], cutoff, fs, order))
    anomaly_sample.T[0] = np.abs(butter_lowpass_filter(anomaly_sample.T[0], cutoff, fs, order))
    normal_sample.T[1] = np.abs(butter_lowpass_filter(normal_sample.T[1], cutoff, fs, order))
    anomaly_sample.T[1] = np.abs(butter_lowpass_filter(anomaly_sample.T[1], cutoff, fs, order))
    normal_sample.T[2] = np.abs(butter_lowpass_filter(normal_sample.T[2], cutoff, fs, order))
    anomaly_sample.T[2] = np.abs(butter_lowpass_filter(anomaly_sample.T[2], cutoff, fs, order))
    normal_samples.append(stats.median_abs_deviation(normal_sample))
    anomaly_samples.append(stats.median_abs_deviation(anomaly_sample))
plotScatterSamples(normal_samples, anomaly_samples, num_samples, title='MAD')

# Plot histograms of correlation matricies
num_samples = 70
n_bins = 20
normal_samples = []
anomaly_samples = []
for i in range(num_samples):
    normal_sample = np.genfromtxt(normal_op_filenames[i], delimiter=',')
    anomaly_sample = np.genfromtxt(anomaly_op_filenames[i], delimiter=',')
    normal_sample = normal_sample - np.mean(normal_sample, axis=0)
    anomaly_sample = anomaly_sample - np.mean(anomaly_sample, axis=0)
    normal_sample.T[0] = np.abs(butter_lowpass_filter(normal_sample.T[0], cutoff, fs, order))
    anomaly_sample.T[0] = np.abs(butter_lowpass_filter(anomaly_sample.T[0], cutoff, fs, order))
    normal_sample.T[1] = np.abs(butter_lowpass_filter(normal_sample.T[1], cutoff, fs, order))
    anomaly_sample.T[1] = np.abs(butter_lowpass_filter(anomaly_sample.T[1], cutoff, fs, order))
    normal_sample.T[2] = np.abs(butter_lowpass_filter(normal_sample.T[2], cutoff, fs, order))
    anomaly_sample.T[2] = np.abs(butter_lowpass_filter(anomaly_sample.T[2], cutoff, fs, order))
    normal_samples.append(np.corrcoef(normal_sample.T))
    anomaly_samples.append(np.corrcoef(anomaly_sample.T))
normal_samples = np.array(normal_samples)
anomaly_samples = np.array(anomaly_samples)
print('Correlation coefficients of normal sample:')
print(np.corrcoef(normal_sample.T))

# Draw plots
fig, axs = plt.subplots(3, 3)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
axs[0, 1].hist(normal_samples[:,0,1], bins=n_bins, color='blue')
axs[0, 1].hist(anomaly_samples[:,0,1], bins=n_bins, color='red')
axs[0, 2].hist(normal_samples[:,0,2], bins=n_bins, color='blue')
axs[0, 2].hist(anomaly_samples[:,0,2], bins=n_bins, color='red')
axs[1, 2].hist(normal_samples[:,1,2], bins=n_bins, color='blue')
axs[1, 2].hist(anomaly_samples[:,1,2], bins=n_bins, color='red')
fig.suptitle('Histograms of Correlation Coefficients')

# Function: Calculate FFT for each axis in a given sample
def extract_fft_features(sample):

    # Truncate sample size
    #sample = sample[0:max_measurements]

    # Crate a window
    hann_window = np.hanning(sample.shape[0])

    # Compute a windowed FFT of each axis in the sample (leave off DC)
    out_sample = np.zeros((int(sample.shape[0] / 2), sample.shape[1]))
    for i, axis in enumerate(sample.T):
        fft = abs(np.fft.rfft(axis * hann_window))
        out_sample[:, i] = fft[1:]

    return out_sample

# Test: Compute FFTs (without DC) for samples and average them together
num_samples = 70
normal_ffts = []
anomaly_ffts = []
for i in range(num_samples):
    normal_sample = np.genfromtxt(normal_op_filenames[i], delimiter=',')
    anomaly_sample = np.genfromtxt(anomaly_op_filenames[i], delimiter=',')
    normal_fft = extract_fft_features(normal_sample)
    anomaly_fft = extract_fft_features(anomaly_sample)
    normal_ffts.append(normal_fft)
    anomaly_ffts.append(anomaly_fft)
normal_ffts = np.array(normal_ffts)
anomaly_ffts = np.array(anomaly_ffts)
normal_fft_avg = np.average(normal_ffts, axis=0)
anomaly_fft_avg = np.average(anomaly_ffts, axis=0)

# Plot FFTs
start_bin = 1
fig, axs = plt.subplots(3, 1, figsize=(8, 6))
fig.tight_layout(pad=3.0)

axs[0].plot(normal_fft_avg[start_bin:, 0], label='normal', color='blue')
axs[0].plot(anomaly_fft_avg[start_bin:, 0], label='anomaly', color='red')
axs[0].set_title('X')
axs[0].set_xlabel('bin')
axs[0].set_ylabel('G-force')
axs[0].legend()

axs[1].plot(normal_fft_avg[start_bin:, 1], label='normal', color='blue')
axs[1].plot(anomaly_fft_avg[start_bin:, 1], label='anomaly', color='red')
axs[1].set_title('Y')
axs[1].set_xlabel('bin')
axs[1].set_ylabel('G-force')
axs[1].legend()

axs[2].plot(normal_fft_avg[start_bin:, 2], label='normal', color='blue')
axs[2].plot(anomaly_fft_avg[start_bin:, 2], label='anomaly', color='red')
axs[2].set_title('Z')
axs[2].set_xlabel('bin')
axs[2].set_ylabel('G-force')
axs[2].legend()