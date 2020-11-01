from scipy.signal import butter, filtfilt
from scipy import signal




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
    b,a = signal.iirnotch(normal_notch_frequency, quality, sampling_rate)

    filtered_data = filtfilt(b,a,data)
    return filtered_data