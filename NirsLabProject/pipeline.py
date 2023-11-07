import os

import mne
import time
from typing import List, Dict
import scipy.stats as sp_stats
from joblib import Parallel, delayed


import numpy as np

from NirsLabProject import spikes_detection
from NirsLabProject import plotting
from NirsLabProject import utils
from NirsLabProject.config.subject import Subject
from NirsLabProject.group_spikes import group_spikes
from NirsLabProject.config.consts import *


def resample_and_filter_data(subject: Subject):
    if not os.path.exists(subject.paths.subject_resampled_fif_path):
        raw = mne.io.read_raw_edf(subject.paths.subject_raw_edf_path)
        raw = utils.pick_seeg_channels(raw)
        utils.clean_channels_name_in_raw_obj(raw)
        print(f'Resampling data, it might take some time... (around {len(raw.ch_names) * 5 // 60} minutes)')
        raw.resample(SR)
        raw.save(subject.paths.subject_resampled_fif_path)
    return mne.io.read_raw_fif(subject.paths.subject_resampled_fif_path)


# Filter the data and plot the spikes
# Returns an array of features, each feature is an array of the features of the spikes in the channel
def channel_processing(subject: Subject, raw: mne.io.Raw, spikes_windows: Dict[str, np.ndarray], channel_name: str, channel_index: int):
    channel_raw = raw.copy().pick_channels([channel_name])
    channel_raw.load_data()
    filtered_channel_raw = channel_raw.copy().filter(l_freq=LOW_THRESHOLD_FREQUENCY, h_freq=HIGH_THRESHOLD_FREQUENCY)
    filtered_channel_data = filtered_channel_raw.get_data()[0]
    filtered_channel_data = sp_stats.zscore(filtered_channel_data)
    if '018' in subject.name and 'RA' in channel_name:
        channel_spikes_windows = spikes_windows[channel_name] if 'LFP' not in subject.name else spikes_windows[f'{channel_name[:-1]}3']
    else:
        channel_spikes_windows = spikes_windows[channel_name] if 'LFP' not in subject.name else spikes_windows[f'{channel_name[:-1]}1']
    if 'LFP' in subject.name:
        if '018' in subject.name and 'RA' in channel_name:
            channel_spikes_indexes = np.load(
                os.path.join(subject.paths.subject_spikes_dir_path, f'peaks-{channel_name[:-1]}3.npz.npy').replace('_LFP',''))
        else:
            channel_spikes_indexes = np.load(os.path.join(subject.paths.subject_spikes_dir_path, f'peaks-{channel_name[:-1]}1.npz.npy').replace('_LFP', ''))
    else:
        channel_spikes_indexes = utils.get_spikes_peak_indexes_in_spikes_windows(filtered_channel_data, channel_spikes_windows)
        np.save(os.path.join(subject.paths.subject_spikes_dir_path, f'peaks-{channel_name}.npz'), channel_spikes_indexes)
    amplitudes, lengths = utils.extract_spikes_peaks_features(filtered_channel_data, channel_spikes_indexes)

    features = [None] * NUM_OF_FEATURES
    features[TIMESTAMP_INDEX] = channel_spikes_indexes
    features[CHANNEL_INDEX] = np.full((channel_spikes_indexes.shape[0], 1), channel_index)
    features[AMPLITUDE_INDEX] = amplitudes.reshape((-1, 1))
    features[DURATION_INDEX] = lengths.reshape((-1, 1))
    x, y, z = utils.get_coordinates_of_channel(subject, channel_name)
    features[CORD_X_INDEX] = np.full((channel_spikes_indexes.shape[0], 1), x)
    features[CORD_Y_INDEX] = np.full((channel_spikes_indexes.shape[0], 1), y)
    features[CORD_Z_INDEX] = np.full((channel_spikes_indexes.shape[0], 1), z)
    channel_spikes_features = np.concatenate(
        features,
        axis=1
    )

    # if channel_name.endswith('1') or 'LFP' in subject.name:
    plotting.create_TFR_plot(subject, channel_raw, channel_spikes_indexes, channel_name)
    plotting.create_PSD_plot(subject, channel_raw, channel_spikes_indexes, channel_name)
    plotting.create_ERP_plot(subject, filtered_channel_raw, channel_spikes_indexes, channel_name)
    plotting.create_channel_features_histograms(subject, amplitudes, lengths, channel_name)

    return channel_spikes_features


def main(subject_name: str):
    subject = Subject(subject_name, True)

    raw = resample_and_filter_data(subject)
    raw = mne.io.read_raw(subject.paths.subject_raw_fif_path)
    spikes_windows = spikes_detection.detect_spikes_of_subject(subject, raw)

    # calls channel_processing with the given arguments in parallel on all cpu cores for each channel
    channel_names = spikes_windows.keys() if 'LFP' not in subject.name else raw.ch_names
    channels_spikes = Parallel(n_jobs=os.cpu_count(), backend='multiprocessing')(
        delayed(channel_processing)(subject, raw, dict(spikes_windows), channel_name, i) for i, channel_name in enumerate(channel_names)
    )

    # creating a dictionary of the results
    channels_spikes_features = {channel_name: channel_spikes for channel_name, channel_spikes in zip(channel_names, channels_spikes)}

    # converting the timestamps to seconds
    channel_spikes = {channel_name: channel_spikes[:, 0] / SR for channel_name, channel_spikes in channels_spikes_features.items()}

    # raster plot with hypnogram and histogram
    plotting.create_raster_plot(
        subject=subject,
        spikes=channel_spikes,
        tmin=raw.tmin,
        tmax=raw.tmax,
        add_hypnogram=True,
        add_histogram=True,
    )

    # raster plot with histogram
    plotting.create_raster_plot(
        subject=subject,
        spikes=channel_spikes,
        tmin=raw.tmin,
        tmax=raw.tmax,
        add_hypnogram=False,
        add_histogram=True,
    )

    # extracts groups features
    groups, flat_features = group_spikes(channels_spikes_features)
    print(groups)


if __name__ == '__main__':
    start_time = time.time()
    # main('p396')
    # main('p487')
    main('p018_MTL')
    # main('p018_MTL_LFP')
    print(f'Time taken: {(time.time() - start_time) / 60} minutes')