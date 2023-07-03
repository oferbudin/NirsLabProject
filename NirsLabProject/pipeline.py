import os
import time
from typing import Dict

import mne
import numpy as np
import scipy.stats as sp_stats
from joblib import Parallel, delayed

from NirsLabProject.utils import general_utils as utils
from NirsLabProject.config.consts import *
from NirsLabProject.config.subject import Subject
from NirsLabProject.utils.group_spikes import group_spikes
from NirsLabProject.utils import scalp_spikes_detection, intracranial_spikes_detection, plotting


def resample_and_filter_data(subject: Subject):
    if FORCE_LOAD_EDF or not os.path.exists(subject.paths.subject_resampled_fif_path):
        print('Reading raw data...')
        raw = mne.io.read_raw_edf(subject.paths.subject_raw_edf_path)
        raw = utils.pick_seeg_and_eog_channels(raw)
        # raw = utils.remove_bad_channels(raw)
        utils.clean_channels_name_in_raw_obj(raw)
        if raw.info['sfreq'] != SR:
            print(f'Resampling data, it might take some time... (around {len(raw.ch_names) * 5 // 60} minutes)')
            raw.resample(SR, n_jobs=2)
        print('Saving resampled data...')
        raw.save(subject.paths.subject_resampled_fif_path, overwrite=True)

    print('Reading resampled data...')
    raw = mne.io.read_raw_fif(subject.paths.subject_resampled_fif_path)
    seeg_raw = raw.copy().pick_channels([ch for ch in raw.ch_names if 'EOG' not in ch])
    eog_raw = raw.copy().pick_channels([ch for ch in raw.ch_names if 'EOG' in ch])
    return seeg_raw, eog_raw


# Filter the data and plot the spikes
# Returns an array of features, each feature is an array of the features of the spikes in the channel
def channel_processing(subject: Subject, raw: mne.io.Raw, spikes_windows: Dict[str, np.ndarray], channel_name: str, channel_index: int, channel_name_to_coordinates: dict):
    print(f'Processing channel {channel_name}...')

    channel_raw = raw.copy().pick_channels([channel_name])
    channel_raw.load_data()
    filtered_channel_raw = channel_raw.copy().filter(l_freq=LOW_THRESHOLD_FREQUENCY, h_freq=HIGH_THRESHOLD_FREQUENCY)
    filtered_channel_data = filtered_channel_raw.get_data()[0]
    filtered_channel_data = sp_stats.zscore(filtered_channel_data)
    channel_spikes_windows = spikes_windows[channel_name]
    channel_spikes_indexes = utils.get_spikes_peak_indexes_in_spikes_windows(filtered_channel_data, channel_spikes_windows)
    if channel_spikes_indexes is None:
        return None
    amplitudes, lengths = utils.extract_spikes_peaks_features(filtered_channel_data, channel_spikes_indexes)

    features = [None] * NUM_OF_FEATURES
    features[TIMESTAMP_INDEX] = channel_spikes_indexes
    features[CHANNEL_INDEX] = np.full((channel_spikes_indexes.shape[0], 1), channel_index)
    features[AMPLITUDE_INDEX] = amplitudes.reshape((-1, 1))
    features[DURATION_INDEX] = lengths.reshape((-1, 1))
    x, y, z = utils.get_coordinates_of_channel(channel_name_to_coordinates, channel_name)
    features[CORD_X_INDEX] = np.full((channel_spikes_indexes.shape[0], 1), x)
    features[CORD_Y_INDEX] = np.full((channel_spikes_indexes.shape[0], 1), y)
    features[CORD_Z_INDEX] = np.full((channel_spikes_indexes.shape[0], 1), z)

    channel_spikes_features = np.concatenate(
        features,
        axis=1
    )

    if channel_name.endswith('1'):
        plotting.create_TFR_plot(subject, channel_raw, channel_spikes_indexes, channel_name)
        plotting.create_PSD_plot(subject, channel_raw, channel_spikes_indexes, channel_name)
        plotting.create_ERP_plot(subject, filtered_channel_raw, channel_spikes_indexes, channel_name)
        plotting.create_channel_features_histograms(subject, amplitudes, lengths, channel_name)

    return channel_spikes_features


def extract_spikes_features(subject: Subject, seeg_raw: mne.io.Raw, intracranial_spikes_spikes_windows: Dict[str, np.ndarray]):
    print('Extracting spikes features...')
    if FORCE_CALCULATE_SPIKES_FEATURES or not os.path.exists(subject.paths.subject_channels_spikes_features_path):
        # calls channel_processing with the given arguments in parallel on all cpu cores for each channel
        channel_names = intracranial_spikes_spikes_windows.keys()
        channel_name_to_coordinates = utils.calculate_coordinates()
        channels_spikes = Parallel(n_jobs=os.cpu_count(), backend='multiprocessing')(
            delayed(channel_processing)(subject, seeg_raw, dict(intracranial_spikes_spikes_windows), channel_name, i,
                                        channel_name_to_coordinates) for i, channel_name in enumerate(channel_names) if
            channel_name in seeg_raw.ch_names
        )

        # creating a dictionary of the results
        channels_spikes_features = {channel_name: channel_spikes for channel_name, channel_spikes in
                                    zip(channel_names, channels_spikes) if channel_spikes is not None}

        # saving the results
        np.save(subject.paths.subject_channels_spikes_features_path, channels_spikes_features)

    return np.load(subject.paths.subject_channels_spikes_features_path, allow_pickle=True).item()


def get_flat_features(subject: Subject, seeg_raw: mne.io.Raw, intracranial_spikes_spikes_windows: Dict[str, np.ndarray], scalp_spikes_spikes_windows: np.ndarray):
    # extracts spikes features
    channels_spikes_features = extract_spikes_features(subject, seeg_raw, intracranial_spikes_spikes_windows)

    # extracts groups features
    groups, flat_features, index_to_channel = group_spikes(channels_spikes_features)

    # adds scalp spikes flag to the features
    flat_features = utils.add_flag_of_scalp_detection_to_spikes_features(flat_features, scalp_spikes_spikes_windows)

    # adds stimuli flag to the features
    flat_features = utils.add_stimuli_flag_to_spikes_features(subject, flat_features)

    # adds sleeping stage flag to the features
    flat_features = utils.add_sleeping_stage_flag_to_spike_features(subject, flat_features)

    np.save(subject.paths.subject_flat_features_path, flat_features)

    return flat_features, channels_spikes_features, index_to_channel, groups


def create_raster_plots(subject: Subject, seeg_raw: mne.io.Raw, channels_spikes_features: Dict[str, np.ndarray]):
    # converting the timestamps to seconds
    channel_spikes = {channel_name: channel_spikes[:, TIMESTAMP_INDEX] / SR for channel_name, channel_spikes in
                      channels_spikes_features.items()}
    # raster plot with hypnogram and histogram
    plotting.create_raster_plot(
        subject=subject,
        spikes=channel_spikes,
        tmin=seeg_raw.tmin,
        tmax=seeg_raw.tmax,
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
    main('p487')
    # main('p489')
    print(f'Time taken: {(time.time() - start_time) / 60} minutes')