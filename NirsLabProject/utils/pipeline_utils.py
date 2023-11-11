import os
from typing import Dict

import mne
import numpy as np
import scipy.stats as sp_stats
from joblib import Parallel, delayed
from NirsLabProject.config.paths import Paths

from NirsLabProject.utils import general_utils as utils
from NirsLabProject.config.consts import *

from NirsLabProject.utils import general_utils as utils
from NirsLabProject.config.consts import *
from NirsLabProject.config.subject import Subject
from NirsLabProject.utils.group_spikes import group_spikes
from NirsLabProject.utils import plotting


def resample_and_filter_data(subject: Subject):
    if FORCE_LOAD_EDF or not os.path.exists(subject.paths.subject_resampled_fif_path):
        print('Reading raw data...')
        raw = mne.io.read_raw_edf(subject.paths.subject_raw_edf_path)
        raw = utils.pick_seeg_and_eog_channels(raw)
        utils.clean_channels_name_in_raw_obj(raw)
        print(f'Raw data shape: {raw.tmax - raw.tmin} seconds, {raw.ch_names} channels, {raw.info["sfreq"]} Hz')
        raw = utils.remove_bad_channels(raw)
        if raw.info['sfreq'] != SR:
            print(f'Resampling data, it might take some time... (around {len(raw.ch_names) * 5 // 60} minutes)')
            raw.resample(SR, n_jobs=2)
            print('Saving resampled data...')
        raw.save(subject.paths.subject_resampled_fif_path, overwrite=True)
    else:
        print('Data was already resampled, reading it...')
        raw = mne.io.read_raw_fif(subject.paths.subject_resampled_fif_path)
        print(
            f'Raw data shape: {raw.tmax - raw.tmin} seconds, {raw.info["sfreq"]} Hz, channels  {raw.ch_names}')

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
    features[CHANNEL_INDEX] = np.full((channel_spikes_indexes.shape[0], 1), int(f'{subject.p_number}{channel_index}'))
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
        plotting.create_PSD_plot(subject, channel_raw, channel_spikes_indexes, channel_name)
        plotting.create_TFR_plot(subject, channel_raw, channel_spikes_indexes, channel_name)
        plotting.create_ERP_plot(subject, filtered_channel_raw, channel_spikes_indexes, channel_name)
        plotting.create_channel_features_histograms(subject, amplitudes, lengths, channel_name)

    return channel_spikes_features


def extract_spikes_features(subject: Subject, seeg_raw: mne.io.Raw, intracranial_spikes_spikes_windows: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    print('Extracting spikes features...')
    if FORCE_CALCULATE_SPIKES_FEATURES or not os.path.exists(subject.paths.subject_channels_spikes_features_path):
        # calls channel_processing with the given arguments in parallel on all cpu cores for each channel
        channel_names = intracranial_spikes_spikes_windows.keys()
        channel_name_to_coordinates = utils.calculate_coordinates(subject)
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


def get_index_to_channel(subject: Subject, channels_spikes_features: Dict[str, np.ndarray]) -> Dict[int, str]:
    index_to_channel = {}
    for channel_name in channels_spikes_features.keys():
        if channels_spikes_features[channel_name].shape[0] == 0:
            continue
        index = channels_spikes_features[channel_name][0][CHANNEL_INDEX]
        index_to_channel[index] = channel_name
    np.save(subject.paths.subject_channel_name_to_index_path, index_to_channel)
    return index_to_channel


def get_flat_features(subject: Subject, seeg_raw: mne.io.Raw, intracranial_spikes_spikes_windows: Dict[str, np.ndarray], scalp_spikes_spikes_windows: np.ndarray):
    # extracts spikes features
    channels_spikes_features = extract_spikes_features(subject, seeg_raw, intracranial_spikes_spikes_windows)

    index_to_channel = get_index_to_channel(subject, channels_spikes_features)

    # extracts groups features
    groups, flat_features = group_spikes(subject, channels_spikes_features, index_to_channel)

    # adds scalp spikes flag to the features
    flat_features = utils.add_flag_of_scalp_detection_to_spikes_features(flat_features, scalp_spikes_spikes_windows)

    # adds stimuli flag to the features
    flat_features = utils.add_stimuli_flag_to_spikes_features(subject, flat_features)

    # adds sleeping stage flag to the features
    flat_features = utils.add_sleeping_stage_flag_to_spike_features(subject, flat_features)

    # adds subject id to the features
    subject_id = np.ones((flat_features.shape[0], 1)) * subject.p_number
    flat_features = np.concatenate((flat_features, subject_id), axis=1)

    np.save(subject.paths.subject_flat_features_path, flat_features)

    return flat_features, channels_spikes_features, index_to_channel, groups


def create_raster_plots(subject: Subject, seeg_raw: mne.io.Raw, channels_spikes_features: Dict[str, np.ndarray], scalp_spikes_windows: np.ndarray):
    # converting the timestamps to seconds
    channel_spikes = {channel_name: channel_spikes[:, TIMESTAMP_INDEX] / SR for channel_name, channel_spikes in
                      channels_spikes_features.items()}
    eog_channels_spikes = {'EOG1': [spike for spike in scalp_spikes_windows]}
    channel_spikes.update(eog_channels_spikes)

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
        tmin=seeg_raw.tmin,
        tmax=seeg_raw.tmax,
        add_hypnogram=False,
        add_histogram=True,
    )


def save_electrodes_coordinates(subject: Subject, seeg_raw: mne.io.Raw):
    if subject.stimuli_project:
        stimulation_locations = utils.pars_stimuli_locations_file(subject)
    else:
        stimulation_locations = []
    plotting.save_electrodes_position(seeg_raw, subject, stimulation_locations)

def get_features_of_subjects(subjects):
    flat_features = []
    index_to_channel = {}
    for subj in subjects:
        flat_features.append(np.load(subj.paths.subject_flat_features_path))
        index_to_channel.update(np.load(subj.paths.subject_channel_name_to_index_path, allow_pickle=True).item())

    flat_features = np.concatenate(flat_features)
    return flat_features[flat_features[:, AMPLITUDE_INDEX] < MAX_AMPLITUDE_Z_SCORE], index_to_channel


def detection_project_intersubjects_plots(sourasky: bool = False, show: bool = False):
    # all subjects
    subjects = [Subject(d, True) for d in os.listdir(Paths.products_data_dir_path) if d.startswith('p')]

    # only sourasky subjects
    if sourasky:
        subjects = filter(lambda subj: subj.sourasky_project, subjects)

    # only detection projects
    subjects = filter(lambda subj: not subj.stimuli_project, subjects)
    # only subjects with features
    subjects = filter(lambda subj: os.path.exists(subj.paths.subject_flat_features_path), subjects)

    flat_features, index_to_channel = get_features_of_subjects(subjects)
    subject = Subject(DETECTION_PROJECT_INTERSUBJECTS_SUBJECT_NAME, True)
    plotting.create_raincloud_plot_for_all_spikes_features(subject, flat_features, show=show)
    plotting.plot_scalp_detection_probability_for_every_electrode_in_3d(subject, flat_features, index_to_channel)


def stimuli_effects(show: bool = False):
    subjects = [Subject(d, True) for d in os.listdir(Paths.products_data_dir_path) if d.startswith('p')]
    subjects = filter(lambda subj: subj.stimuli_project, subjects)
    subjects = filter(lambda subj: os.path.exists(subj.paths.subject_flat_features_path), subjects)

    subject_stats = {
        AMPLITUDE_INDEX: {},
        DURATION_INDEX: {},
        GROUP_EVENT_DURATION_INDEX: {},
        GROUP_EVENT_SIZE_INDEX: {},
        GROUP_EVENT_SPATIAL_SPREAD_INDEX: {},
        GROUP_EVENT_DEEPEST_INDEX: {},
        GROUP_EVENT_SHALLOWEST_INDEX: {},
    }

    feature_id_to_title = {
        AMPLITUDE_INDEX: 'Spike Amplitude Average',
        DURATION_INDEX: 'Spike Width Average',
        GROUP_EVENT_DURATION_INDEX: 'Spike Group Event Duration Average',
        GROUP_EVENT_SIZE_INDEX: 'Spike Group Event Size Average',
        GROUP_EVENT_SPATIAL_SPREAD_INDEX: 'Spike Group Event Spatial Spread Average',
        GROUP_EVENT_DEEPEST_INDEX: 'Spike Group Event Deepest Electrode Avrage',
        GROUP_EVENT_SHALLOWEST_INDEX: 'Spike Group Event Shallowest Electrode Avrage',
    }

    for subj in subjects:
        subj_features = np.load(subj.paths.subject_flat_features_path)

        group_ids = subj_features[:, GROUP_INDEX]
        unique_indices = np.unique(group_ids, return_index=True)[1]
        unique_group_subj_features = subj_features[unique_indices]

        if not os.path.exists(subj.paths.subject_sleep_scoring_path):
            continue

        # g for groups - the difference is that we take a representative spike for each group
        g_before, g_stim_block, g_pause_block, g_during_window, g_after = utils.stimuli_effects(subj, unique_group_subj_features)
        before, stim_block, pause_block, during_window, after = utils.stimuli_effects(subj, subj_features)

        for feature_index in subject_stats.keys():
            if GROUP_INDEX <= feature_index <= GROUP_EVENT_SPATIAL_SPREAD_INDEX:
                baseline_mean = np.mean(g_before[:, feature_index])
                stim_block_mean = np.mean(g_stim_block[:, feature_index])
                pause_block_mean = np.mean(g_pause_block[:, feature_index])
                during_window_mean = np.mean(g_during_window[:, feature_index])
                after_mean = np.mean(g_after[:, feature_index])
            else:
                baseline_mean = np.mean(before[:, feature_index])
                stim_block_mean = np.mean(stim_block[:, feature_index])
                pause_block_mean = np.mean(pause_block[:, feature_index])
                during_window_mean = np.mean(during_window[:, feature_index])
                after_mean = np.mean(after[:, feature_index])

            if not (np.isnan(baseline_mean) or np.isnan(stim_block_mean) or np.isnan(during_window_mean)):
                subject_stats[feature_index][subj.name] = [
                    0,
                    100 * (baseline_mean - stim_block_mean) / baseline_mean,
                    100 * (baseline_mean - pause_block_mean) / baseline_mean,
                    100 * (baseline_mean - during_window_mean) / baseline_mean,
                    100 * (baseline_mean - after_mean) / baseline_mean,
                ]

    subject = Subject(STIMULI_PROJECT_INTERSUBJECTS_SUBJECT_NAME, True)
    for feature_index, stats in subject_stats.items():
        plotting.plot_stimuli_effects(subject, stats, feature_id_to_title[feature_index], show=show)
