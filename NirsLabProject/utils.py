import mne
import numpy as np
import pandas as pd
from functools import partial
from typing import Tuple, Dict, List
from scipy.signal import find_peaks

from NirsLabProject.config.consts import *
from NirsLabProject.config.subject import Subject


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
    filtered_spikes = list(filter(lambda x: 0 < x['length'] <= MAX_SPIKE_LENGTH_MILLISECONDS, spikes))
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
    mne.rename_channels(raw.info, {ch: clean_channel_name(ch) for ch in raw.ch_names})


# Removes the prefix of the channel name
def pick_seeg_channels(raw: mne.io.Raw) -> mne.io.Raw:
    SEEG_PREFIX = 'SEEG'
    channels = raw.ch_names
    last_channel = len(channels) - 1
    for i in range(len(channels)-1, 0, -1):
        if channels[i].startswith(SEEG_PREFIX):
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


def get_coordinates_of_channel(subject: SUBJECT, channel_name: str) -> Tuple:
    # TODO: implement
    return 1, 1, 1


# decorator that catches exceptions and prints them
def catch_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f'function {func.__name__} with args {args}, kwargs {kwargs} failed: {e}')
            return None
    return wrapper

