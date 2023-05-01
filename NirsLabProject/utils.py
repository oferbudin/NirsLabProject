import mne
import numpy as np
from functools import partial
from typing import Tuple, List
from scipy.signal import find_peaks

from NirsLabProject.config.consts import *


def clean_channel_name(channel_name: str) -> str:
    return channel_name.replace('-REF1', '').replace('SEEG ', '').replace('-REF', '')


# Returns the channel data between the start and end of the spike the start and end indexes
def get_data_of_spike_in_range(channel_data: np.ndarray, spike_timestamp: float) -> Tuple[np.ndarray, int, int]:
    start = max(int((spike_timestamp - SPIKE_RANGE_SECONDS / 2) * SR), 0)
    end = int((spike_timestamp + SPIKE_RANGE_SECONDS / 2) * SR)
    return channel_data[start:end], start, end


# Spike index in the channel data
def get_spike_index(channel_data: np.ndarray, spike_timestamp: float) -> int:
    spike_data, spike_start_index, _ = get_data_of_spike_in_range(channel_data, spike_timestamp)

    maximum_gradient_index = np.argmax(np.gradient(spike_data))
    peaks, _ = find_peaks(spike_data, width=(20, 70))
    spike_indexes = peaks[peaks > maximum_gradient_index]

    if len(spike_indexes) == 0:
        spike_index_offset = np.argmax(spike_data)
    else:
        spike_index_offset = spike_indexes[0]

    return spike_start_index + spike_index_offset


def get_centered_spike_range(channel_data: np.ndarray, spike_timestamp: float) -> tuple:
    spike_data, start, end = get_data_of_spike_in_range(channel_data, spike_timestamp)

    if len(spike_data) < SPIKE_RANGE_SECONDS:
        print(f'Dropped spike {spike_timestamp}')
        return -1, -1

    spike_index = get_spike_index(spike_data, start)

    start = spike_index - int(SPIKE_RANGE_SECONDS/2 * SR)
    end = spike_index + int(SPIKE_RANGE_SECONDS/2 * SR)
    return start, end


def create_epochs(channel_raw: mne.io.Raw, channel_data: np.ndarray, spikes: np.ndarray, tmin=-SPIKE_RANGE_SECONDS, tmax=SPIKE_RANGE_SECONDS):
    spikes = spikes.copy()
    spikes = spikes.reshape(-1, 1)
    spikes = np.vectorize(partial(get_spike_index, channel_data))(spikes)
    spikes = spikes.astype(int)
    zeros = np.zeros((spikes.shape[0], 2), dtype=int)
    spikes = np.hstack((spikes, zeros))
    indices = np.arange(spikes.shape[0]).reshape(-1, 1)
    spikes[:, -1] = indices[:, 0]
    spikes = spikes[spikes[:, 0] != spikes[0, 0]]
    return mne.Epochs(channel_raw, spikes, tmin=tmin, tmax=tmax)


def extract_spike_features(data, start, end):
    peak_index = data[start:end].argmax() + start
    amplitude = data[start:end].max()

    first_half_start = start + np.where(data[start:peak_index] < amplitude / 2)[0][-1]
    second_half_end = peak_index + np.where(data[peak_index:end] < amplitude / 2)[0][0]
    spike_length = second_half_end - first_half_start
    return amplitude, spike_length


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