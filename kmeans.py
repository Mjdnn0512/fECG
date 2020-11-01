import numpy as np
import pandas as pd
import scipy.signal



def get_max_points(data, window_start, window_end):
    temp = data[window_start:window_end]
    maximums_t = scipy.signal.argrelextrema(temp, np.greater)
    return maximums_t


def get_min_points(data, window_start, window_end):
    temp = data[window_start:window_end]
    minimums_t = scipy.signal.argrelextrema(temp, np.less)
    return minimums_t


def get_x_y_pairs_1(minimums, maximums, data, window_start, window_end):
    temp = data[window_start:window_end]
    start_index_min = 0  # this is the index value into the array of signal data for min array
    start_index_max = 0  # this is the index value into the array of signal data for max array
    # ensure that the first value is a maximum
    if (minimums[0][start_index_min] < maximums[0][start_index_max]):
        start_index_min = start_index_min + 1

    delta_x_y = np.array([])  # declare a blank array for storing the dx*dy values
    sample_location = np.array([])  # array to store the sample location for dx*dy value

    delta_y_array = np.array([])
    delta_x_array = np.array([])

    for i in range(start_index_max, min(len(minimums[0]) - start_index_min, len(maximums[0]))):
        delta_x = abs(minimums[0][i + start_index_min] - maximums[0][i])  # delta x is the sample # difference
        delta_y = abs(data[minimums[0][i + start_index_min]] - data[
            maximums[0][i]])  # delta y is the y difference between max and min
        delta_y_array = np.append(delta_y_array, delta_y)
        delta_x_array = np.append(delta_x_array, delta_x)
        x_y = delta_x * delta_y  # multiply together to improve detection
        # delta_y_array = np.append(delta_y_array,x_y)
        delta_x_y = np.append(delta_x_y, x_y)  # append all the values to the array
        sample_location = np.append(sample_location, maximums[0][i])

    max_min_pairs = pd.DataFrame()

    max_min_pairs['sample_location'] = sample_location.T
    max_min_pairs["x"] = delta_x_array.T
    max_min_pairs["y"] = delta_y_array.T

    return max_min_pairs


def get_x_y_pairs_2(minimums, maximums, data, window_start, window_end):
    temp = data[window_start:window_end]
    start_index_min = 0  # this is the index value into the array of signal data for min array
    start_index_max = 0  # this is the index value into the array of signal data for max array
    # ensure that the first value is a maximum
    if (minimums[0][start_index_min] < maximums[0][start_index_max]):
        start_index_min = start_index_min + 1

    delta_x_y = np.array([])  # declare a blank array for storing the dx*dy values
    sample_location = np.array([])  # array to store the sample location for dx*dy value

    delta_y_array = np.array([])
    delta_x_array = np.array([])

    for i in range(start_index_max, min(len(minimums[0]) - start_index_min, len(maximums[0]))):
        delta_x = abs(minimums[0][i + start_index_min] - maximums[0][i])  # delta x is the sample # difference
        delta_y = abs(data[minimums[0][i + start_index_min]] - data[
            maximums[0][i]])  # delta y is the y difference between max and min
        delta_y_array = np.append(delta_y_array, delta_y)
        delta_x_array = np.append(delta_x_array, delta_x)
        x_y = delta_x * delta_y  # multiply together to improve detection
        # delta_y_array = np.append(delta_y_array,x_y)
        delta_x_y = np.append(delta_x_y, x_y)  # append all the values to the array
        sample_location = np.append(sample_location, maximums[0][i])

    max_min_pairs = pd.DataFrame()

    max_min_pairs["x"] = delta_x_array.T
    max_min_pairs["y"] = delta_y_array.T

    return max_min_pairs, sample_location


# takes the dataframe and calculates the distance from each point to each of the centers. Leaves a column "minimum"
# that contains the center that is closest to the point and returns a dataframe
def assign_points(max_min_pairs, centers, color_map):
    for i in centers.keys():
        max_min_pairs['distance_from_{}'.format(i)] = ((np.sqrt((max_min_pairs['x'] -
                                                centers[i][0])**2 + (max_min_pairs['y']
                                                 - centers[i][1])**2))) # this is to calculate cartesian distance
    centers_dist_col = ['distance_from_{}'.format(i) for i in centers.keys()]
    max_min_pairs['minimum'] = max_min_pairs.loc[:,centers_dist_col].idxmin(axis=1)
    max_min_pairs['minimum'] = max_min_pairs['minimum'].map(lambda x: int(x.lstrip('distance_from_')))
    max_min_pairs['color'] = max_min_pairs['minimum'].map(lambda x: color_map[x])
    return max_min_pairs


# update the centers based on the points that were closest to them
def recalculate_centers(centers, max_min_pairs):
    for i in centers.keys():
        centers[i][0] = np.mean(max_min_pairs[max_min_pairs['minimum'] == i]['x'])
        centers[i][1] = np.mean(max_min_pairs[max_min_pairs['minimum'] == i]['y'])
    return centers