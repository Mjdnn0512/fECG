import numpy as np
import numpy.fft as fft
import mne
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import kmeans as km
import filters as fl


# These are the window vars that define the window location and size of the data being looked at
window_high = 10000
window_low = 0


# Parameters for filter section
fs = 1000
cutoff_high = 8
cutoff_low = 100
filter_order = 2
notch_freq = 50
quality = 30

# Converting to a CSV File
# If you dont have the CSV data, then uncomment this block of code below
# edf = mne.io.read_raw_edf('r04.edf')
# header = ','.join(edf.ch_names)
# np.savetxt('output.csv', edf.get_data().T, delimiter=',', header=header)

# -----------------------------------------------------------------
# Loading data and all 5 columns in a, b, c, d, e arrays
a,b,c,d,e= np.loadtxt('output.csv', delimiter=',', unpack=True).tolist()

# check out the frequency spectrum of the data
# spectrum = fft.fft(b)
# freq = fft.fftfreq(len(spectrum))
# plt.plot(freq,abs(spectrum))
# plt.show()


# Creating an array for sample locations since the data arrays only have voltage signal
samples =np.arange(window_low, window_high)

# Apply the filters to the data in array b
# notch_filter_data = fl.notch_filter(b,notch_freq,fs,quality)
low_filter_data = fl.butter_lowpass_filter(b, cutoff_frequency=cutoff_low, sampling_rate=fs, order=filter_order)
high_filter_data = fl.butter_highpass_filter(low_filter_data, cutoff_frequency=cutoff_high, sampling_rate=fs, order=filter_order)


# Apply the filter to the direct fetal data to remove baseline and high freq noise
temp_direct_fetal_data = fl.butter_highpass_filter(a, cutoff_frequency=cutoff_high, sampling_rate=fs, order=filter_order)
direct_fetal_data = fl.butter_highpass_filter(temp_direct_fetal_data, cutoff_frequency=cutoff_high, sampling_rate=fs, order=filter_order)



# clustering algorithm for detecting QRS


# Find the max and min points of the data within the window
maximums_t = km.get_max_points(high_filter_data, window_low, window_high)
minimums_t = km.get_min_points(high_filter_data, window_low, window_high)


#plot the data showing the min max pairs
plt.subplots(figsize=(50,10))
plt.plot(samples[window_low:window_high], high_filter_data[window_low:window_high])
plt.plot(samples[window_low:window_high], direct_fetal_data[window_low:window_high])
plt.plot(samples[maximums_t], high_filter_data[maximums_t], 'o')
plt.plot(samples[minimums_t], high_filter_data[minimums_t], 'o')
plt.xlabel('Samples')
plt.ylabel('Signal')
plt.title('Filtered Data')
plt.show()

# Apply the clustering algorithm using the SkiKit learn method
##############################################################
color_map = {1: 'r', 2: 'b', 3: 'g'} # Map the points groups to different colors


# Get a dataframe containing the x and y values for the max min pairs
max_min_pairs_skkit, sample_location = km.get_x_y_pairs_2(minimums_t, maximums_t, high_filter_data, window_low, window_high)


#define k = 3 so that there will be three groupings shown
kmeans = KMeans(n_clusters=3)
kmeans.fit(max_min_pairs_skkit)

labels = kmeans.predict(max_min_pairs_skkit)
centers_skkit = kmeans.cluster_centers_
colors = map(lambda x: color_map[x + 1], labels)


# plot the data after algorithm is applied
plt.subplots(figsize=(50,6))
plt.scatter(sample_location, max_min_pairs_skkit['y'], color=list(colors), alpha=0.5, edgecolors='k')
plt.plot(samples[window_low:window_high], direct_fetal_data[window_low:window_high])
plt.xlabel('Using SciKit Learn')
plt.ylabel('Signal')
plt.title('Filtered Data')
plt.show()


# Apply the second k means clustering algorithm (not using SkiKit)
##################################################################

# Initialization section
max_min_pairs = km.get_x_y_pairs_1(minimums_t, maximums_t, high_filter_data, window_low, window_high)
k = 3
np.random.seed(200)
centers = {
    i + 1: [np.random.randint(0,30),np.random.randint(0,1)]
    for i in range(k)
}

# Get firs set of centers
max_min_pairs = km.assign_points(max_min_pairs, centers, color_map)
centers = km.recalculate_centers(centers, max_min_pairs)

# copy and update, if the old value center assignments equals the new, then the k means process is complete
while True:
    old_assignments = max_min_pairs['minimum'].copy(deep=True)
    centers = km.recalculate_centers(centers, max_min_pairs)
    max_min_pairs = km.assign_points(max_min_pairs,centers, color_map)
    if old_assignments.equals(max_min_pairs['minimum']):
        break


# plot the data
plt.subplots(figsize=(100,6))
plt.scatter(max_min_pairs['sample_location'], max_min_pairs['y'], color =max_min_pairs['color'], alpha=0.5, edgecolors='k')
plt.plot(samples[window_low:window_high], direct_fetal_data[window_low:window_high])
plt.xlabel('Using my algorithm')
plt.ylabel('signal')
plt.title('Filtered Data')
plt.show()
