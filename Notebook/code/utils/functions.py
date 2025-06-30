"""
    functions for cleaning & preprocessing & feature extraction
"""
import pandas as pd
import numpy as np
import scipy as sp
from scipy import signal
from scipy import fftpack
from statsmodels.robust import mad as median_deviation
from scipy.stats import iqr, pearsonr
import yaml
import pickle
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

# -------------------load dataset---------------------
def load_dataset(filename):
    with open(filename, 'rb') as f:
        data_x, data_y, data_group = pickle.load(f)
    print(f"from file {filename}")
    print(f"reading instances: {data_x.shape}")
    data_x = data_x.astype(np.float32)
    data_y = data_y.astype(np.uint8)
    data_group = data_group.astype(np.uint8)
    return data_x, data_y, data_group


""" model training data reading
    Reads the pickle file data to obtain a format suitable for RF training.
"""
def generate_pkldata_RF(features_pickle_path):
    # --------------load cfg & data file--------------
    data_x, data_y, data_group = load_dataset(features_pickle_path)
    # --------------sample feature select--------------
    sel = VarianceThreshold(threshold=0.01)
    data_x = sel.fit_transform(data_x)
    data_y = data_y.ravel()
    return data_x, data_y, data_group

"""
    Convert pickle-format data to npz-format data: after loading and splitting the pickle data, immediately convert the training, test, and validation sets into a format suitable for Transformer training.
"""
def pkl_to_npz(data_x, data_y, data_group, win, step):

    unique_persons = set(data_group)
    unique_activities = set(data_y)

    data_x_window = []
    data_y_window = []
    data_group_window = []
    total_len = 0
    for person_id in unique_persons:
        for activity_id in unique_activities:
            person_activity_mask = (data_group == person_id) & (data_y == activity_id)
            person_activity_data = data_x[person_activity_mask]
            total_len = total_len + len(person_activity_data)
            for i in range(0, len(person_activity_data) - win + 1, step):
                window = person_activity_data[i:i + win]
                data_x_window.append(window)
                data_y_window.append(np.repeat(activity_id, win, axis=0))
                data_group_window.append(np.repeat(person_id, win, axis=0))
        
    signals = np.array(data_x_window, dtype=np.float32)
    act_labels = np.array(data_y_window, dtype=np.int64)
    person_labels = np.array(data_group_window, dtype=np.int64)
    return signals, act_labels, person_labels

"""load npz-format data"""
def data_figure_npz(CLASS_LABELS,act_labels,y_train,y_test,y_val):
    plt.figure(figsize=(10, 10), dpi=80)

    unique, counts = np.unique(act_labels, return_counts=True)
    plt.bar(CLASS_LABELS[unique], counts)
    plt.xticks(rotation=70)

    unique, counts = np.unique(y_train, return_counts=True)
    plt.bar(CLASS_LABELS[unique], counts)
    plt.xticks(rotation=70)

    unique, counts = np.unique(y_test, return_counts=True)
    plt.bar(CLASS_LABELS[unique], counts)
    plt.xticks(rotation=70)

    unique, counts = np.unique(y_val, return_counts=True)
    plt.bar(CLASS_LABELS[unique], counts)
    plt.xticks(rotation=70)
    
    plt.legend(["All", "Train", "Test", "Validation"])

    plt.show()
def load_dataset_Har(filename):
    with open(filename, 'rb') as f:
        data_x, data_y = pickle.load(f)
    print(f"from file {filename}")
    print(f"reading instances: {data_x.shape}")
    data_x = data_x.astype(np.float32)
    data_y = data_y.astype(np.uint8)
    return data_x, data_y


def get_config(config):
    print(f'load {config}')
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


# -------------------features----------------------
def median(data, kernel_size=5):
    if len(data.shape) == 1:
        n_cols = 1
    else:
        n_cols = data.shape[1]
    for i in range(n_cols):
        data[:, i] = signal.medfilt(data[:, i], kernel_size=kernel_size)
    return data


def low_pass(data, fn, fs):
    """
        5 order Butterworth low pass filter
    :param data: raw data
    :param fn: cut-off freq
    :param fs: sample freq
    :return:
    """
    low_b, low_a = signal.butter(5, 2 * fn / fs, 'lowpass')
    return signal.filtfilt(low_b, low_a, data, axis=0)


def fast_fourier_transform(data):
    complex_f = fftpack.fft(data, axis=0)
    amplitude_f = np.abs(complex_f)
    return amplitude_f

def magnitude_of_triaxial(data):
    mag = np.empty((len(data), 0))
    for i in range(data.shape[1] // 3):
        mag = np.hstack((mag, np.sqrt(np.sum(np.power(data[:, 3 * i:3 * i + 3], 2), axis=1)).reshape(-1, 1)))
    return mag


def mean_axial(data):
    return list(np.mean(data, axis=0))


def harmonic_mean_axial(data):
    n = data.shape[0]
    divide_v = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        for j, v in enumerate(data[:, i]):
            if v != 0:
                divide_v[i] += 1 / v
    result = []
    for i in range(data.shape[1]):
        if divide_v[i] != 0:
            result.append(n / divide_v[i])
        else:
            result.append(0)  # NAN replaced by 0
    return result


def waveform_length_axial(data):
    result = np.zeros(data.shape[1])
    for i in range(1, data.shape[0]):
        result += np.abs(data[i, :] - data[i-1, :])
    return list(result)


def peak2peak_axial(data):
    return list(np.max(data, axis=0) - np.min(data, axis=0))


def std_axial(data):
    return list(np.std(data, axis=0))


def mad_axial(data):
    return list(median_deviation(data, axis=0))  # calculate the median absolute deviation value of each column


def max_axial(data):
    return list(np.max(data, axis=0))


def min_axial(data):
    return list(np.min(data, axis=0))


def iqr_axial(data):
    return list(iqr(data, axis=0))  # calculate the interquartile range value of each column


def sma_axial(data):
    return list(np.sum(abs(data), axis=0))


def energy_axial(data):
    return list(np.sum(np.power(data, 2), axis=0))


def mean_energy_axial(data):
    return list(np.sum(np.power(data, 2), axis=0) / len(data))


def pearsonr_axial(data):
    assert len(data.shape) != 1 and data.shape[1] % 3 == 0
    results = []
    for i in range(data.shape[1] // 3):
        results.append(pearsonr(data[:, 3 * i], data[:, 3 * i + 1])[0])
        results.append(pearsonr(data[:, 3 * i + 1], data[:, 3 * i + 2])[0])
        results.append(pearsonr(data[:, 3 * i + 2], data[:, 3 * i])[0])
    # NAN replaced by 0
    for i in range(len(results)):
        if np.isnan(results[i]):
            results[i] = 0
    return results


def skewness_axial(data):
    return list(sp.stats.skew(data, axis=0))


def kurtosis_axial(data):
    return list(sp.stats.kurtosis(data, axis=0))


def max_freq_axial(data, sample_freq):
    if len(data.shape) == 1:
        n_col = 1
    else:
        n_col = data.shape[1]
    results = []
    freqs = sp.fftpack.fftfreq(len(data), d=1 / float(sample_freq))
    for i in range(n_col):
        results.append(freqs[data[:, i].argmax()])
    return results


def f_mean_freq_axial(data, sample_freq):
    freqs = sp.fftpack.fftfreq(len(data), d=1 / float(sample_freq))

    f_mean_freq = list(np.sum(data * np.array(freqs).reshape((-1, 1)), axis=0) / np.sum(data, axis=0))

    # NAN is replaced by 0
    for i in range(len(f_mean_freq)):
        if np.isnan(f_mean_freq[i]):
            f_mean_freq[i] = 0

    return f_mean_freq
#t_data：27 dimensions
def t_features_generate(t_data):

    t_mean = mean_axial(t_data)# average
    t_std = std_axial(t_data)# std
    t_mad = mad_axial(t_data)# Median absolute deviation
    t_max = max_axial(t_data)# max
    t_min = min_axial(t_data)# min
    t_iqr = iqr_axial(t_data)# Interquartile range (degree of dispersion)
    t_sma = sma_axial(t_data)# Amplitude area
    t_mean_energy = mean_energy_axial(t_data)# average energy
    t_pearsonr = pearsonr_axial(t_data)# Pearson correlation coefficient (measures linear correlation between data)
    t_skew = skewness_axial(t_data)# Skewness (Asymmetry)
    t_kurt = kurtosis_axial(t_data)# Kurtosis (Peakedness)
    t_harmonic_mean = harmonic_mean_axial(t_data)# Harmonic Mean: 3D, a certain feature along three axes (e.g., x, y, z)
    t_peak2peak = peak2peak_axial(t_data)# Peak-to-Peak Value (Amplitude Range)
    t_wave_len = waveform_length_axial(t_data)# Waveform Length (Degree of Variation)

    t_features_vector = t_mean + t_std + t_mad + t_max + t_min + t_iqr + t_sma + t_mean_energy \
                        + t_pearsonr + t_skew + t_kurt + t_harmonic_mean + t_peak2peak + t_wave_len

    return t_features_vector# 572 diamonds

def mag_t_features_generate(mag_t_data):

    mag_t_mean = mean_axial(mag_t_data)
    mag_t_std = std_axial(mag_t_data)
    mag_t_mad = mad_axial(mag_t_data)
    mag_t_max = max_axial(mag_t_data)
    mag_t_min = min_axial(mag_t_data)
    mag_t_iqr = iqr_axial(mag_t_data)
    mag_t_sma = sma_axial(mag_t_data)
    mag_t_mean_energy = mean_energy_axial(mag_t_data)
    mag_t_skew = skewness_axial(mag_t_data)
    mag_t_kurt = kurtosis_axial(mag_t_data)
    mag_t_harmonic_mean = harmonic_mean_axial(mag_t_data)
    mag_t_peak2peak = peak2peak_axial(mag_t_data)
    mag_t_wave_len = waveform_length_axial(mag_t_data)

    mag_t_features_vector = mag_t_mean + mag_t_std + mag_t_mad + mag_t_max + mag_t_min + mag_t_iqr + \
                            mag_t_sma + mag_t_mean_energy + mag_t_skew + mag_t_kurt + mag_t_harmonic_mean + \
                            mag_t_peak2peak + mag_t_wave_len

    return mag_t_features_vector


def f_features_generate(f_data, sample_freq):

    f_mean = mean_axial(f_data)
    f_std = std_axial(f_data)
    f_mad = mad_axial(f_data)
    f_max = max_axial(f_data)
    f_min = min_axial(f_data)
    f_iqr = iqr_axial(f_data)
    f_sma = sma_axial(f_data)
    f_skew = skewness_axial(f_data)
    f_kurt = kurtosis_axial(f_data)
    f_mean_energy = mean_energy_axial(f_data)
    f_freq_max = max_freq_axial(f_data, sample_freq)# Statistical Results of Frequency-Domain Features on Different Axes
    f_mean_freq = f_mean_freq_axial(f_data, sample_freq)
    f_harmonic_mean = harmonic_mean_axial(f_data)
    f_peak2peak = peak2peak_axial(f_data)
    f_wave_len = waveform_length_axial(f_data)

    features_vector = f_mean + f_std + f_mad + f_max + f_min + f_iqr + f_sma + f_skew + f_kurt + f_mean_energy + \
                      f_freq_max + f_mean_freq + f_harmonic_mean + f_peak2peak + f_wave_len
    return features_vector


def f_mag_t_features_generate(mag_f_data, sample_freq):

    mag_f_mean = mean_axial(mag_f_data)
    mag_f_std = std_axial(mag_f_data)
    mag_f_mad = mad_axial(mag_f_data)
    mag_f_max = max_axial(mag_f_data)
    mag_f_min = min_axial(mag_f_data)
    mag_f_iqr = iqr_axial(mag_f_data)
    mag_f_sma = sma_axial(mag_f_data)
    mag_f_skew = skewness_axial(mag_f_data)
    mag_f_kurt = kurtosis_axial(mag_f_data)
    mag_f_mean_energy = mean_energy_axial(mag_f_data)
    mag_f_freq_max = max_freq_axial(mag_f_data, sample_freq)
    mag_f_mean_freq = f_mean_freq_axial(mag_f_data, sample_freq)
    mag_f_harmonic_mean = harmonic_mean_axial(mag_f_data)
    mag_f_peak2peak = peak2peak_axial(mag_f_data)
    mag_f_wave_len = waveform_length_axial(mag_f_data)

    features_vector = mag_f_mean + mag_f_std + mag_f_mad + mag_f_max + mag_f_min + mag_f_iqr + mag_f_sma + mag_f_skew \
                      + mag_f_kurt + mag_f_mean_energy + mag_f_freq_max + mag_f_mean_freq + mag_f_harmonic_mean + \
                      mag_f_peak2peak + mag_f_wave_len
    return features_vector


def t_features_names():
    Feature = ['Mean', 'Std', 'Mad', 'Max', 'Min', 'Iqr', 'Sma', 'MeanEnergy',
               'Pearsonr', 'Skewness', 'Kurtosis', 'HarmonicMean', 'Peak2peak', 'Wavelen']
    Position = ['Hand', 'Chest', 'Ankle']
    Sensor = ['Acc', 'Gyro', 'Mag']
    Axis = ['X', 'Y', 'Z']

    # 't_FeaturePositionSensorAxis'
    features_names = []
    feature_name = 't_'
    for feature in Feature:
        temp_0 = feature_name
        feature_name = feature_name + feature
        for position in Position:
            temp_1 = feature_name
            feature_name = feature_name + position
            for sensor in Sensor:
                temp_2 = feature_name
                feature_name = feature_name + sensor
                for axis in Axis:
                    temp_3 = feature_name
                    feature_name = feature_name + axis
                    features_names.append(feature_name)
                    feature_name = temp_3
                feature_name = temp_2
            feature_name = temp_1
        feature_name = temp_0

    return features_names


def mag_t_features_names():
    Feature = ['Mean', 'Std', 'Mad', 'Max', 'Min', 'Iqr', 'Sma', 'Mean_energy',
               'Skewness', 'Kurtosis', 'HarmonicMean', 'Peak2peak', 'Wavelen']
    Position = ['Hand', 'Chest', 'Ankle']
    Sensor = ['Acc', 'Gyro', 'Mag']

    # 'mag_t_FeaturePositionSensor'
    features_names = []
    feature_name = 'mag_t_'
    for feature in Feature:
        temp_0 = feature_name
        feature_name = feature_name + feature
        for position in Position:
            temp_1 = feature_name
            feature_name = feature_name + position
            for sensor in Sensor:
                temp_2 = feature_name
                feature_name = feature_name + sensor
                features_names.append(feature_name)
                feature_name = temp_2
            feature_name = temp_1
        feature_name = temp_0
    return features_names


def f_features_names():
    Feature = ['Mean', 'Std', 'Mad', 'Max', 'Min', 'Iqr', 'Sma', 'Skewness', 'Kurtosis', 'MeanEnergy',
               'FreqOfMax', 'MeanFreq', 'HarmonicMean', 'Peak2peak', 'Wavelen']
    Position = ['Hand', 'Chest', 'Ankle']
    Sensor = ['Acc', 'Gyro', 'Mag']
    Axis = ['X', 'Y', 'Z']

    features_names = []
    feature_name = 'f_'
    for feature in Feature:
        temp_0 = feature_name
        feature_name = feature_name + feature
        for position in Position:
            temp_1 = feature_name
            feature_name = feature_name + position
            for sensor in Sensor:
                temp_2 = feature_name
                feature_name = feature_name + sensor
                for axis in Axis:
                    temp_3 = feature_name
                    feature_name = feature_name + axis
                    features_names.append(feature_name)
                    feature_name = temp_3
                feature_name = temp_2
            feature_name = temp_1
        feature_name = temp_0

    return features_names


def f_mag_features_names():
    Feature = ['Mean', 'Std', 'Mad', 'Max', 'Min', 'Iqr', 'Sma', 'Skewness', 'Kurtosis', 'MeanEnergy',
               'FreqOfMax', 'MeanFreq', 'HarmonicMean', 'Peak2peak', 'Wavelen']
    Position = ['Hand', 'Chest', 'Ankle']
    Sensor = ['Acc', 'Gyro', 'Mag']

    features_names = []
    feature_name = 'f_mag_'
    for feature in Feature:
        temp_0 = feature_name
        feature_name = feature_name + feature
        for position in Position:
            temp_1 = feature_name
            feature_name = feature_name + position
            for sensor in Sensor:
                temp_2 = feature_name
                feature_name = feature_name + sensor
                features_names.append(feature_name)
                feature_name = temp_2
            feature_name = temp_1
        feature_name = temp_0
    return features_names


# ----------------------preprocess---------------------------
def sliding_window(data_x, data_y, window_width, stride_len):
    number = (len(data_x) - window_width) // stride_len + 1
    x = np.empty((0, window_width, *data_x.shape[1:]))
    y = np.empty(0)
    for i in range(number):
        start = i * stride_len
        end = start + window_width
        data_x_t = data_x[start:end, :]
        data_y_t = (data_y[start + window_width//2],)
        x = np.vstack((x, data_x_t.reshape(1, *data_x_t.shape)))
        y = np.concatenate((y, data_y_t))
    return x, y


def normalize(data_x):
    data_x = (data_x - np.mean(data_x, axis=0)) / np.std(data_x, axis=0)
    return data_x


def labels_adjust(data_y):
    """
    – 1 lying           --->0
    – 2 sitting         --->1
    – 3 standing        --->2
    – 4 walking         --->3
    – 5 running         --->4
    – 6 cycling         --->5
    – 7 Nordic walking  --->6
    – 12 ascending stairs   --->7
    – 13 descending stairs  --->8
    – 16 vacuum cleaning    --->9
    – 17 ironing            --->10
    – 24 rope jumping       --->11
    :param data_y:
    :return:
    """
    data_y[data_y == 1] = 0
    data_y[data_y == 2] = 1
    data_y[data_y == 3] = 2
    data_y[data_y == 4] = 3
    data_y[data_y == 5] = 4
    data_y[data_y == 6] = 5
    data_y[data_y == 7] = 6
    data_y[data_y == 12] = 7
    data_y[data_y == 13] = 8
    data_y[data_y == 16] = 9
    data_y[data_y == 17] = 10
    data_y[data_y == 24] = 11
    return data_y


def discard_data(data, discard_label=0):
    """
        discard data with label=discard_label
    """
    if type(discard_label) == int:
        data = np.delete(data, np.argwhere(data[:, 0] == discard_label), axis=0)
    else:
        for label in discard_label:
            data = np.delete(data, np.argwhere(data[:, 0] == label), axis=0)
    return data


def head_tail_delete(data, n_delete):
    """ Function to delete head and tail samples(n_delete) for each label
    :param data: numpy.ndarray
    :param n_delete: int
        number of samples need to delete
    :return:
        Processed  data
    """
    last_label = data[0, 0]
    cnt = 0
    start_r = 0
    delete_r = np.empty(0, dtype=np.int)
    for r in np.arange(data.shape[0]):
        if last_label != data[r, 0]:
            if cnt < 2 * n_delete:
                delete_r = np.concatenate((delete_r, np.arange(start_r, r)))
            else:
                delete_r = np.concatenate((delete_r, np.arange(start_r, start_r + n_delete)))
                delete_r = np.concatenate((delete_r, np.arange(r - n_delete, r)))
            last_label = data[r, 0]
            start_r = r
        cnt += 1
    return np.delete(data, delete_r, axis=0)


def select_columns(data, mag_used):
    # – 1 timestamp (s)
    # – 2 activityID (see II.2. for the mapping to the activities)
    # – 3 heart rate (bpm)
    # – 4-20 IMU hand
    # – 21-37 IMU chest
    # – 38-54 IMU ankle

    # IMU hand 4-20 :
    # – 1 temperature (°C)
    # – 2-4 3D-acceleration data (ms -2 ), scale: ±16g, resolution: 13-bit
    # – 5-7 3D-acceleration data (ms -2 ), scale: ±6g, resolution: 13-bit *
    # – 8-10 3D-gyroscope data (rad/s)
    # – 11-13 3D-magnetometer data (μT)
    # – 14-17 orientation (invalid in this data collection)

    # the columns we not used
    delete_columns = np.array([0, 2]) # an array with column indices 0 and 2 (excluding 1)：timeStamp、actID、heart rate

    if not mag_used:
        hand = np.concatenate((np.array([3, ]), np.arange(7, 10), np.arange(13, 16), np.arange(16, 20)))
        chest = np.concatenate((np.array([20, ]), np.arange(24, 27), np.arange(30, 33), np.arange(33, 37)))
        ankle = np.concatenate((np.array([37, ]), np.arange(41, 44), np.arange(47, 50), np.arange(50, 54)))

    else:# These data columns need to be deleted.
        hand = np.concatenate((np.array([3, ]), np.arange(7, 10), np.arange(16, 20)))
        chest = np.concatenate((np.array([20, ]), np.arange(24, 27), np.arange(33, 37)))
        ankle = np.concatenate((np.array([37, ]), np.arange(41, 44), np.arange(50, 54)))

    delete_columns = np.concatenate([delete_columns, hand, chest, ankle])
    """
    delete the 27th columns——0-23：Temp、accXYZ、orien0123，24-26：timestamp、actID、heart rate
    New Data Arrangement：actID、(Three devices) accXYZ、gyroXYZ、magnXYZ
    """
    return np.delete(data, delete_columns, axis=1)


def divide_x_y(data):
    data_y = data[:, 0].astype(np.int)# Column Index: 0
    data_x = data[:, 1:]# All Other Columns
    return data_x, data_y


def data_preprocess(filename, mag_used):
    data = np.loadtxt(filename)
    #27 columns—-new data arrangement：actID、(Three devices) accXYZ、gyroXYZ、magnXYZ
    data = select_columns(data, mag_used)  # only use IMUs data
    data = head_tail_delete(data, 1000)  # trim[10s, 10s]
    data = discard_data(data, discard_label=0)# Delete data with label 0——other (transient activities)
    data_x, data_y = divide_x_y(data)
    data_y = labels_adjust(data_y)
    data_x = np.array([pd.Series(i).interpolate() for i in data_x.T]).T
    return data_x, data_y
