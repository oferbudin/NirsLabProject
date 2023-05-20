import mne
import numpy as np
from functools import partial
from typing import Tuple, List
from scipy.signal import find_peaks
import scipy.stats as sp_stats

from NirsLabProject.config.consts import *
from NirsLabProject.config.subject import Subject


def clean_channel_name(channel_name: str) -> str:
    return channel_name.replace('-REF1', '').replace('SEEG ', '').replace('-REF', '')


def extract_spike_features(data, peak_index):
    spike_range_indexes = int(SPIKE_RANGE_SECONDS * SR)
    amplitude = data[peak_index]
    start_frame_index = max(0, peak_index-spike_range_indexes)
    end_frame_index = min(len(data), peak_index+spike_range_indexes)

    first_half_start = start_frame_index + np.where(data[start_frame_index:peak_index] < amplitude / 2)[0]
    if len(first_half_start) > 0:
        first_half_start = first_half_start[-1]
    else:
        return {'index': peak_index, 'amplitude': amplitude, 'length': -1}

    second_half_end = peak_index + np.where(data[peak_index:end_frame_index] < amplitude / 2)[0]
    if len(second_half_end) > 0:
        second_half_end = second_half_end[0]
    else:
        return {'index': peak_index, 'amplitude': amplitude, 'length': -1}

    return {'index': peak_index, 'amplitude': amplitude, 'length': second_half_end - first_half_start}


# Returns the channel data between the start and end of the spike the start and end indexes
def get_data_of_spike_in_range(channel_data: np.ndarray, spike_timestamp: float) -> Tuple[np.ndarray, int, int]:
    start = max(int((spike_timestamp-SPIKE_RANGE_SECONDS/2) * SR), 0)
    end = int((spike_timestamp + SPIKE_RANGE_SECONDS+SPIKE_RANGE_SECONDS/2) * SR)
    return channel_data[start:end], start, end


# Spike index in the channel data
def get_spike_index(channel_data: np.ndarray, spike_timestamp: float) -> int:
    spike_data, spike_start_index, _ = get_data_of_spike_in_range(channel_data, spike_timestamp)

    peaks, _ = find_peaks(spike_data)
    spikes = map(lambda peak: extract_spike_features(spike_data, peak), peaks)
    filtered_spikes = list(filter(lambda x: 0 < x['length'] <= MAX_SPIKE_LENGTH_MILLISECONDS, spikes))
    if not filtered_spikes:
        return -1

    peak = max(filtered_spikes, key=lambda x: x['amplitude'])
    if peak['amplitude'] < MIN_AMPLITUDE_Z_SCORE:
        return -1

    return spike_start_index + peak['index']


def get_real_spikes_indexs(channel_data: np.ndarray, spikes: np.ndarray):
    spikes = spikes.copy()
    spikes = spikes.reshape(-1, 1)
    spikes = np.vectorize(partial(get_spike_index, channel_data))(spikes)
    spikes = spikes[spikes >= 0]  # Removes epochs that have no spike with length that smaller than  MAX_SPIKE_LENGTH_MILLISECONDS
    spikes = np.unique(spikes).reshape(-1, 1)
    spikes = spikes.astype(int)
    return spikes


def create_epochs(channel_raw: mne.io.Raw, channel_data: np.ndarray, spikes: np.ndarray,
                  tmin=-SPIKE_RANGE_SECONDS, tmax=SPIKE_RANGE_SECONDS) -> mne.Epochs:
    spikes = get_real_spikes_indexs(channel_data, spikes)
    zeros = np.zeros((spikes.shape[0], 2), dtype=int)
    spikes = np.hstack((spikes, zeros))
    indices = np.arange(spikes.shape[0]).reshape(-1, 1)
    spikes[:, -1] = indices[:, 0]
    spikes = spikes[spikes[:, 0] != spikes[0, 0]]
    return mne.Epochs(channel_raw, spikes, tmin=tmin, tmax=tmax)


def clean_channels_name(raw: mne.io.Raw):
    mne.rename_channels(raw.info, {ch: clean_channel_name(ch) for ch in raw.ch_names})


def pick_seeg_channels(raw: mne.io.Raw) -> int:
    SEEG_PREFIX = 'SEEG'
    channels = raw.ch_names
    last_channel = len(channels) - 1
    for i in range(len(channels)-1, 0, -1):
        if channels[i].startswith(SEEG_PREFIX):
            last_channel = i+1
            break
    seeg_channels = channels[:last_channel]
    return raw.pick_channels(seeg_channels)


def extract_spikes_features(channel_data: np.ndarray, spikes: np.ndarray) -> Tuple:
    spikes = get_real_spikes_indexs(channel_data, spikes)
    spikes = np.unique(spikes).flatten()

    # Vectorized feature extraction using np.vectorize
    v_extract_features = np.vectorize(partial(extract_spike_features, channel_data))

    # Perform feature extraction on all indexes at once
    features_array = v_extract_features(spikes)
    indexes = np.array([d['index'] for d in features_array])
    peaks = np.array([d['amplitude'] for d in features_array])
    lengths = np.array([d['length'] for d in features_array])

    return indexes, peaks, lengths
