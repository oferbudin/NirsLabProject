import os
import mne
import numpy as np
import pandas as pd
from functools import partial
from typing import Tuple, Dict, List
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
import scipy.stats as sp_stats


from NirsLabProject.config.consts import *
from NirsLabProject.config.paths import Paths
from NirsLabProject.config.subject import Subject
from NirsLabProject.utils import sleeping_utils


COORDINATES = {}


def clean_channel_name(channel_name: str) -> str:
    # Remove the '-REF' or '-REF1' from the channel name
    return channel_name.replace('-REF1', '').replace('SEEG ', '').replace('-REF', '')


def get_spike_amplitude_and_length(data: np.ndarray, peak_index) -> Dict:
    # taking the spike range (SPIKE_RANGE_SECONDS) before and after the peak timestamp of the window
    spike_range_in_indexes = int(SPIKE_RANGE_SECONDS * SR)
    start_frame_index = max(0, peak_index-spike_range_in_indexes)
    end_frame_index = min(len(data), peak_index+spike_range_in_indexes)

    amplitude = data[peak_index]
    # finding closest point before the peak with half of the peak amplitude
    first_half_start = start_frame_index + np.where(data[start_frame_index:peak_index] < amplitude / 2)[0]
    if len(first_half_start) > 0:
        first_half_start = first_half_start[-1]
    else:
        # if there is no point before the peak with half of the peak amplitude, we will take the start of the window
        return {'index': peak_index, 'amplitude': amplitude, 'length': -1}

    # finding closest point after the peak with half of the peak amplitude
    second_half_end = peak_index + np.where(data[peak_index:end_frame_index] < amplitude / 2)[0]
    if len(second_half_end) > 0:
        second_half_end = second_half_end[0]
    else:
        # if there is no point after the peak with half of the peak amplitude, we will take the end of the window
        return {'index': peak_index, 'amplitude': amplitude, 'length': -1}

    return {'index': peak_index, 'amplitude': amplitude, 'length': second_half_end - first_half_start}


# returns the data of the spike in the range of SPIKE_RANGE_SECONDS seconds before and after the spike_window_timestamp
def get_data_of_spike_in_wide_window(channel_data: np.ndarray, spike_window_timestamp: float) -> Tuple[np.ndarray, int, int]:
    # taking the spike range (SPIKE_RANGE_SECONDS) before and after the timestamp of the window
    start = max(0, int((spike_window_timestamp-SPIKE_RANGE_SECONDS/2) * SR))
    end = int((spike_window_timestamp + SPIKE_RANGE_SECONDS+SPIKE_RANGE_SECONDS/2) * SR)
    return channel_data[start:end], start, end


# finds the exact spike timestamp in the timestamp window and returns the index of the spike peak
def get_spike_amplitude_index(channel_data: np.ndarray, spike_window_timestamp: float) -> int:
    wide_spike_window_data, wide_spike_window_start_index, _end = get_data_of_spike_in_wide_window(channel_data, spike_window_timestamp)

    # Find the peak of the spike in the wide_spike_window_data with length smaller than MAX_SPIKE_LENGTH_MILLISECONDS
    peaks, _ = find_peaks(wide_spike_window_data)
    spikes = map(lambda peak: get_spike_amplitude_and_length(wide_spike_window_data, peak), peaks)
    filtered_spikes = list(filter(lambda x: MIN_SPIKE_LENGTH_MILLISECONDS < x['length'] <= MAX_SPIKE_LENGTH_MILLISECONDS, spikes))
    if not filtered_spikes:
        return -1

    # Find the spike with the highest amplitude
    peak = max(filtered_spikes, key=lambda x: x['amplitude'])
    if peak['amplitude'] < MIN_AMPLITUDE_Z_SCORE:
        return -1

    # Return the spike index in the channel data
    return wide_spike_window_start_index + peak['index']


# Returns the spike indexes in the channel data for each spike in the spikes window array
def get_spikes_peak_indexes_in_spikes_windows(channel_data: np.ndarray, spikes_windows: np.ndarray):
    spikes = spikes_windows.copy()
    spikes = spikes.reshape(-1, 1)
    if spikes.shape[0] == 0:
        return None
    spikes = np.vectorize(partial(get_spike_amplitude_index, channel_data))(spikes) # Get the spike index in the channel data for each spike in the spikes array
    spikes = spikes[spikes >= 0] # Removes epochs that have no spike with length that smaller than  MAX_SPIKE_LENGTH_MILLISECONDS
    spikes = np.unique(spikes).reshape(-1, 1)
    spikes = spikes.astype(int)
    return spikes


# creates epochs from the channel data and the spikes indexes
# the epochs are created in the range of tmin seconds before and tmax after the spike peak
def create_epochs(channel_raw: mne.io.Raw, spikes: np.ndarray,
                  tmin=-SPIKE_RANGE_SECONDS, tmax=SPIKE_RANGE_SECONDS) -> mne.Epochs:
    # create epochs array shape as the expected format (n_epochs, 3) - [peak_timestamp, Not relevant,  Not relevant]
    zeros = np.zeros((spikes.shape[0], 2), dtype=int)
    spikes = np.hstack((spikes, zeros))
    indices = np.arange(spikes.shape[0]).reshape(-1, 1)
    spikes[:, -1] = indices[:, 0]
    spikes = spikes[spikes[:, 0] != spikes[0, 0]]
    return mne.Epochs(channel_raw, spikes, tmin=tmin, tmax=tmax)


def clean_channels_name_in_raw_obj(raw: mne.io.Raw):
    print('Cleaning channels names')
    mne.rename_channels(raw.info, {ch: clean_channel_name(ch) for ch in raw.ch_names})


# Removes the prefix of the channel name
def pick_seeg_and_eog_channels(raw: mne.io.Raw) -> mne.io.Raw:
    print('Picking SEEG and EOG channels')
    channels = raw.ch_names
    last_channel = len(channels) - 1
    for i in range(len(channels)-1, 0, -1):
        if channels[i].startswith('SEEG') or channels[i].startswith('EOG'):
            last_channel = i+1
            break
    seeg_channels = channels[:last_channel]
    return raw.pick_channels(seeg_channels)


# returns the channel amplitude and length of the spike as arrays
def extract_spikes_peaks_features(channel_data: np.ndarray, spikes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    spikes = np.unique(spikes).flatten()

    if spikes.shape[0] == 0:
        return np.array([]), np.array([])

    # Vectorized feature extraction using np.vectorize
    v_extract_features = np.vectorize(partial(get_spike_amplitude_and_length, channel_data))

    # Perform feature extraction on all indexes at once
    features_array = v_extract_features(spikes)
    peaks = np.array([d['amplitude'] for d in features_array])
    lengths = np.array([d['length'] for d in features_array])

    return peaks, lengths


def get_stimuli_time_windows(subject: Subject) -> List[Tuple[float, float]]:
    stim = np.array(pd.read_csv(subject.paths.subject_stimuli_path, header=None).iloc[0, :])
    stim_sessions = []
    start = stim[0] / SR
    end = None
    for (i, x) in enumerate(stim):
        # check if the stim is the last one
        if end is not None:
            start = stim[i] / SR
            end = None

        # check if the next stim is in more than 5 minutes
        if i + 1 < stim.size and stim[i + 1] - stim[i] > 5 * 60 * SR:
            end = stim[i] / SR
            # check that it isn't a single stim (like 487, 9595 sec) or shorter than 1 min (like 497)
            if stim[i] / SR - start > 60:
                stim_sessions.append((start, end))

    return stim_sessions


def remove_stimuli_from_raw(subject: Subject, block_raw: mne.io.Raw, start, end):
    raws = []
    # get the stimuli times
    stim = pd.read_csv(subject.paths.subject_stimuli_path, header=None).iloc[0, :].to_list()
    stim = [round(x) for x in stim]
    # get the relevant section from the list with all stimuli
    start_round = start * SR if start * SR in stim else round(start * SR)
    end_round = end * SR if end * SR in stim else round(end * SR)
    current = stim[len(stim) - stim[::-1].index(start_round) - 1: stim.index(end_round) + 1]
    for i, stim_time in enumerate(current):
        if i + 1 < len(current):
            tmin = current[i] / SR - start + 0.5
            tmax = current[i + 1] / SR - start - 0.5
            if tmax > tmin:
                curr_raw = block_raw.copy().crop(tmin=tmin, tmax=tmax)
                raws.append(curr_raw)

    final = mne.concatenate_raws(raws)
    return final


def get_coordinates_of_channel(channel_name_to_coordinates: dict, channel_name: str) -> Tuple:
    if channel_name in channel_name_to_coordinates:
        return tuple(c for c in channel_name_to_coordinates[channel_name])
    print(f'channel {channel_name} not found in coordinates file')
    return np.NAN, np.NAN, np.NAN


# decorator that catches exceptions and prints them
def catch_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f'function {func.__name__} with args {args}, kwargs {kwargs} failed: {e}')
            return None
    return wrapper


def calculate_coordinates():
    sum = {}
    count = {}
    for file in [file for file in os.listdir(Paths.coordinates_data_dir_path) if file.endswith('Loc.txt')]:
        loc_file_path = os.path.join(Paths.coordinates_data_dir_path, file)
        loc_data = pd.read_csv(loc_file_path, delimiter=' ', usecols=[0, 1], header=None, dtype={0: str, 1: int})
        pial_file_path = os.path.join(Paths.coordinates_data_dir_path, file.replace('PostimpLoc.txt', '.Pial'))
        pial_data = pd.read_csv(pial_file_path, delimiter=' ', skiprows=2, header=None,
                           dtype={0: float, 1: float, 2: float})

        for loc_electrode, pial_electrode in zip(loc_data.iterrows(), pial_data.iterrows()):
            electrode_name = f'{loc_electrode[1][0]}{loc_electrode[1][1]}'
            if electrode_name not in sum:
                sum[electrode_name] = [0, 0, 0]
                count[electrode_name] = 0
            count[electrode_name] += 1
            sum[electrode_name][0] += pial_electrode[1][0]
            sum[electrode_name][1] += pial_electrode[1][1]
            sum[electrode_name][2] += pial_electrode[1][2]

    for electrode in sum:
        sum[electrode][0] /= count[electrode]
        sum[electrode][1] /= count[electrode]
        sum[electrode][2] /= count[electrode]

    return sum


def remove_bad_channels(raw: mne.io.Raw):
    print('Removing bad channels, might take a while...')

    n_time_windows = int(raw.tmax - raw.tmin)
    channel_names = list(raw.ch_names)
    channel_features = np.zeros((len(channel_names), 3))

    area_norms = {}
    area_norms_stds = {}
    area_norms_skew = {}
    area_norms_kurtosis = {}

    total_channel_index = 0
    for i, channel_name in enumerate(channel_names):
        channel_area = channel_name[:-1]
        if channel_area not in area_norms:
            print(f'calculating norms for electrode {channel_area}')
            area_norms[channel_area] = []
            area_norms_stds[channel_area] = []
            area_norms_skew[channel_area] = []
            area_norms_kurtosis[channel_area] = []

        channel_index = channel_name[-1]
        channel_raw = raw.copy().pick_channels([channel_name])
        channel_data = channel_raw.get_data()[0]

        # calculate the norm for each time window
        norms = np.zeros(n_time_windows)
        for window_index in range(n_time_windows):
            norms[window_index] = np.sqrt(np.nansum(channel_data[(window_index - 1) * SR:window_index * SR] ** 2))

        area_norms[channel_area].append(norms)
        area_norms_stds[channel_area].append(np.std(norms))
        area_norms_skew[channel_area].append(skew(norms))
        area_norms_kurtosis[channel_area].append(kurtosis(norms))

        # if the next channel is from a different area or the last channel
        # calculate the median of the norms and mark the channels that are above it
        if i == len(channel_names) - 1 or channel_names[i + 1][-1] < channel_index:
            for channel_area, channels in area_norms.items():
                area_norms_stds_median = np.median(area_norms_stds[channel_area])
                area_norms_skew_median = np.median(area_norms_skew[channel_area])
                area_norms_kurtosis_median = np.median(area_norms_kurtosis[channel_area])

                for i, _ in enumerate(channels):
                    # currently the only feature that in use
                    if area_norms_stds[channel_area][i] > area_norms_stds_median * 4:
                        channel_features[total_channel_index][0] = 2
                    elif area_norms_stds[channel_area][i] > area_norms_stds_median * 2:
                        channel_features[total_channel_index][0] = 1.5
                    elif area_norms_stds[channel_area][i] > area_norms_stds_median * 1.5:
                        channel_features[total_channel_index][0] = 1

                    if area_norms_skew[channel_area][i] > area_norms_skew_median * 4:
                        channel_features[total_channel_index][1] = 2
                    elif area_norms_skew[channel_area][i] > area_norms_skew_median * 2:
                        channel_features[total_channel_index][1] = 1

                    if area_norms_kurtosis[channel_area][i] > area_norms_kurtosis_median * 4:
                        channel_features[total_channel_index][2] = 2
                    elif area_norms_kurtosis[channel_area][i] > area_norms_kurtosis_median * 2:
                        channel_features[total_channel_index][2] = 1

                    total_channel_index += 1
            area_norms = {}

    bad_channels_name = []
    for channel_name, features in zip(raw.ch_names, channel_features):
        if features[0] > 1:
            bad_channels_name.append(channel_name)

    if bad_channels_name:
        print(f'Found bad channels, removing them: {bad_channels_name}')
        channels = [channel for channel in raw.ch_names if channel not in bad_channels_name]
        return raw.pick_channels(channels)

    print('No bad channels found')
    return raw


def pars_stimuli_locations_file(subject: Subject) -> List[str]:
    df = pd.read_csv(subject.paths.stimuli_locations_file, delimiter=',')
    stimulating_electrode = ''
    for index, row in df.iterrows():
        pt = row['pt']
        if subject.p_number == int(pt):
            stimulating_electrode = row['StimulatingElectrodeArea']
    if stimulating_electrode == '':
        raise Exception(f'Could not find the stimulating electrode for subject {subject.p_number}')
    electrode_name_and_parts = stimulating_electrode.split('-')
    electrode_name = electrode_name_and_parts[0].upper()
    electrodes = [electrode_name + str(i) for i in electrode_name_and_parts[1:]]
    return electrodes


def add_flag_of_scalp_detection_to_spikes_features(flat_features: np.ndarray, scalp_spikes_spikes_windows: List[int]):
    indexes = np.zeros((flat_features.shape[0], 1))
    for spike in scalp_spikes_spikes_windows:
        window_start = spike * 1000
        match_spikes = np.where(
            np.logical_and(window_start - 100 < flat_features[:, TIMESTAMP_INDEX], flat_features[:, TIMESTAMP_INDEX] < window_start + int(SPIKE_RANGE_SECONDS*1000)))
        indexes[match_spikes[0]] = 1
    return np.concatenate((flat_features, indexes), axis=1)


def add_stimuli_flag_to_spikes_features(subject: Subject, flat_features: np.ndarray) -> np.ndarray:
    flags = np.zeros((flat_features.shape[0], 1))
    if subject.stimuli_project:
        stimuli_windows = get_stimuli_time_windows(subject)
        stimuli_session_start = stimuli_windows[0][0] * SR
        stimuli_session_end = stimuli_windows[-1][1] * SR
        flags[np.where(np.logical_and(stimuli_session_start < flat_features[:, TIMESTAMP_INDEX], flat_features[:, TIMESTAMP_INDEX] < stimuli_session_end))] = STIMULI_FLAG_DURING_STIMULI_SESSION
        flags[np.where(flat_features[:, TIMESTAMP_INDEX] < stimuli_session_start)] = STIMULI_FLAG_BEFORE_FIRST_STIMULI_SESSION
        flags[np.where(flat_features[:, TIMESTAMP_INDEX] > stimuli_session_end)] = STIMULI_FLAG_AFTER_STIMULI_SESSION
        for session in stimuli_windows:
            window_start = session[0] * SR
            window_end = session[1] * SR
            flags[np.where(np.logical_and(window_start < flat_features[:, TIMESTAMP_INDEX], flat_features[:, TIMESTAMP_INDEX] < window_end))] = STIMULI_FLAG_DURING_STIMULI_WINDOW
    return np.concatenate((flat_features, flags), axis=1)


def add_sleeping_stage_flag_to_spike_features(subject: Subject, flat_features: np.ndarray) -> np.ndarray:
    changing_points, values = sleeping_utils.get_hypnogram_changes_in_miliseconds(subject)
    flags = np.zeros((flat_features.shape[0], 1))
    for i in range(len(changing_points) - 1):
        flags[
            # finding the indexes of the spikes that are in the current sleeping stage
            np.where(
                np.logical_and(
                    changing_points[i] < flat_features[:, TIMESTAMP_INDEX],
                    flat_features[:, TIMESTAMP_INDEX] < changing_points[i + 1]
                )
            )
        ] = values[i]
    return np.concatenate((flat_features, flags), axis=1)


def stimuli_effects(subject: Subject, flat_features: np.ndarray):
    sleep_start = sleeping_utils.get_sleep_start_end_indexes(subject)
    before_stimuli = flat_features[
        np.where(
            np.logical_and(
                flat_features[:, TIMESTAMP_INDEX] > sleep_start[0],
                flat_features[:, STIMULI_FLAG_INDEX] == STIMULI_FLAG_BEFORE_FIRST_STIMULI_SESSION
            )
        )
    ]

    during_stimuli = flat_features[flat_features[:, STIMULI_FLAG_INDEX] == STIMULI_FLAG_DURING_STIMULI_SESSION]
    after_stimuli = flat_features[flat_features[:, STIMULI_FLAG_INDEX] == STIMULI_FLAG_AFTER_STIMULI_SESSION]
    return before_stimuli, during_stimuli, after_stimuli


def f_test(group1, group2):
    f = np.var(group1, ddof=1)/np.var(group2, ddof=1)
    nun = group1.size-1
    dun = group2.size-1
    p_value = 1-sp_stats.f.cdf(f, nun, dun)
    return f, p_value


def t_test(group1: np.ndarray, group2: np.ndarray):
    _, f_p_val = f_test(group1, group2)
    if f_p_val < 0.05:
        equal_var = False
    else:
        equal_var = True
    _, t_p_val = sp_stats.ttest_ind(group1, group2, equal_var=equal_var)
    f_p_val = np.format_float_scientific(f_p_val, precision=2)
    t_p_val = np.format_float_scientific(t_p_val, precision=2)
    return t_p_val, f_p_val
