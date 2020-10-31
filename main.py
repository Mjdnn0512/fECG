import numpy as np
import numpy.fft as fft
import mne
import pandas
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import signal
import scipy.signal


window = 3000
fs = 1000
cutoff_high = 8
cutoff_low = 100
filter_order = 2
notch_freq = 60
quality = 30

def butter_lowpass_filter(data, cutoff_frequency, sampling_rate, order):
    nyqs = 0.5 * sampling_rate
    normal_cutoff_freq = cutoff_frequency / nyqs
    b,a = butter(order, normal_cutoff_freq, 'low', False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data
def butter_highpass_filter(data, cutoff_frequency, sampling_rate, order):
    nyqs = 0.5 * sampling_rate
    normal_cutoff_freq = cutoff_frequency / nyqs
    b,a = butter(order, normal_cutoff_freq, 'high', False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def notch_filter(data, notch_frequency, sampling_rate, quality):
    nyqs = 0.5*sampling_rate
    normal_notch_frequency = notch_frequency/nyqs
    b,a = signal.iirnotch(normal_notch_frequency,quality,sampling_rate)
    filtered_data = filtfilt(b,a,data)
    return filtered_data


# Converting to a CSV File
# If you dont have the CSV data, then uncomment this block of code below
# edf = mne.io.read_raw_edf('r04.edf')
# header = ','.join(edf.ch_names)
# np.savetxt('output.csv', edf.get_data().T, delimiter=',', header=header)

# -----------------------------------------------------------------
# Loading data and all 5 columns in a, b, c, d, e arrays
a,b,c,d,e= np.loadtxt('output.csv', delimiter=',', unpack=True).tolist()
# Creating a time array since the CSV file doesn't have any time column

# check out the frequency spectrum of the data
# spectrum = fft.fft(b)
# freq = fft.fftfreq(len(spectrum))
#
# plt.plot(freq,abs(spectrum))
# plt.show()

time =np.arange(window)
notch_filter_data = notch_filter(b,notch_freq,fs,quality)
low_filter_data =butter_lowpass_filter(notch_filter_data, cutoff_frequency=cutoff_low, sampling_rate=fs, order=filter_order)
high_filter_data =butter_highpass_filter(low_filter_data, cutoff_frequency=cutoff_high, sampling_rate=fs, order=filter_order)


temp_direct_fetal_data =butter_highpass_filter(a, cutoff_frequency=cutoff_high, sampling_rate=fs, order=filter_order)
direct_fetal_data =butter_highpass_filter(temp_direct_fetal_data, cutoff_frequency=cutoff_high, sampling_rate=fs, order=filter_order)



# clustering algorithm for detecting QRS

# create test array containing the first 5000 samples
test = high_filter_data[0:window]

maximums_t = scipy.signal.argrelextrema(test, np.greater)
minimums_t = scipy.signal.argrelextrema(test, np.less)
maximums_y = np.copy(test[maximums_t])
minimums_y = np.copy(test[minimums_t])

#plot the data
plt.subplots(figsize=(10,6))
plt.plot(time[0:window], test[0:window])
plt.plot(time[0:window], direct_fetal_data[0:window])
plt.plot(time[maximums_t], test[maximums_t], 'o')
plt.plot(time[minimums_t], test[minimums_t], 'o')
plt.xlabel('time')
plt.ylabel('signal')
plt.title('Filtered Data')
plt.show()


start_index_min = 0 # this is the index value into the array of signal data for min array
start_index_max = 0 # this is the index value into the array of signal data for max array
# # calculate the deltaX*deltaY values
if(minimums_t[0][start_index_min] < maximums_t[0][start_index_max]):
    start_index_min = start_index_min + 1

delta_x_y = np.array([]) # declare a blank array for storing the dx*dy values
sample_location = np.array([]) # array to store the sample location for dx*dy value

for i in range(start_index_max,min(len(minimums_t[0]) - start_index_min,len(maximums_t[0]))):
    delta_x = abs(minimums_t[0][i + start_index_min] - maximums_t[0][i]) #delta x is the sample # difference
    delta_y = abs(test[minimums_t[0][i + start_index_min]] - test[maximums_t[0][i]]) # delta y is the y difference between max and min
    x_y = delta_x*delta_y # multiply together to improve detection
    delta_x_y = np.append(delta_x_y,x_y) # append all the values to the array
    sample_location = np.append(sample_location,maximums_t[0][i])

print(maximums_t)
print(minimums_t)

print(delta_x_y)
print(sample_location)

print(test[584])
print(test[598])
print(test[566])
print(test[580])
print(test[530])
print(test[537])
print(test[505])
print(test[525])
#plot the data
plt.subplots(figsize=(10,6))
plt.scatter(sample_location,delta_x_y)
plt.plot(time[0:window], direct_fetal_data[0:window])
plt.xlabel('delta x*y')
plt.ylabel('signal')
plt.title('Filtered Data')
plt.show()