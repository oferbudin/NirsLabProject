import os
import re
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


def group_spikes_by_electrodes(raw_edf: mne.io.Raw) -> Dict[str, list]:
    electrodes = {}
    for electrode in raw_edf.ch_names:
        electrode_name, _ = utils.extract_channel_name_and_contact_number(electrode)
        if electrode_name not in electrodes:
            electrodes[electrode_name] = []
        electrodes[electrode_name].append(electrode)
    return electrodes


def resample_and_filter_data(subject: Subject):
    print('Reading raw data...')
    if FORCE_LOAD_EDF or not os.listdir(subject.paths.subject_resampled_data_dir_path):
        print('Reampled data not exist exists, loading...')
        raw = mne.io.read_raw_edf(subject.paths.subject_raw_edf_path)
        raw = utils.pick_seeg_and_eog_channels(raw)
        utils.clean_channels_name_in_raw_obj(subject, raw)
        print(f'Raw data shape: {raw.tmax - raw.tmin} seconds, {raw.ch_names} channels, {raw.info["sfreq"]} Hz')
        for electrode_name, group in group_spikes_by_electrodes(raw).items():
            electrode_path = subject.paths.subject_resampled_fif_path(subject.name, electrode_name)
            if FORCE_LOAD_EDF or not os.path.exists(electrode_path):
                print(f'Pre-processing channels of electrode {electrode_name}...')
                raw_electrode = raw.copy().pick_channels(group)
                # raw_electrode = utils.remove_bad_channels(raw_electrode) TODO: check if needed - not working
                if raw_electrode.info['sfreq'] != SR:
                    print(f'Resampling data, it might take some time... (around {len(raw_electrode.ch_names) * 5 // 60} minutes)')
                    raw_electrode = raw_electrode.resample(SR, verbose=True, n_jobs=2)
                print('applying notch filter...')
                raw_electrode = raw_electrode.notch_filter(50 if subject.sourasky_project else 60, n_jobs=2, verbose=True)
                print('applying band pass filter...')
                raw_electrode = raw_electrode.filter(0.1, 499, verbose=True, n_jobs=2)
                print('Saving resampled data...')
                raw_electrode.save(electrode_path, split_size='2GB', verbose=True, overwrite=True)
                del raw_electrode

    loaded_raw = {}
    electrodes_fif = os.listdir(subject.paths.subject_resampled_data_dir_path)
    for electrode_name in electrodes_fif:
        electrode_name = re.search(r'_resampled_(.*)\.fif', electrode_name)
        if electrode_name is None:
            continue
        electrode_name = electrode_name.group(1)
        electrode_path = subject.paths.subject_resampled_fif_path(subject.name, electrode_name)
        print(f'Data for electrode {electrode_name} was already resampled, reading it...')
        channel_raw = mne.io.read_raw_fif(electrode_path)
        utils.clean_channels_name_in_raw_obj(subject, channel_raw)
        loaded_raw[electrode_name] = channel_raw
        print(f'Raw data shape: {channel_raw.tmax - channel_raw.tmin} seconds, {channel_raw.ch_names} channels, {channel_raw.info["sfreq"]} Hz')
    return loaded_raw


# Filter the data and plot the spikes
# Returns an array of features, each feature is an array of the features of the spikes in the channel
def channel_processing(subject: Subject, raw: mne.io.Raw, spikes_windows: Dict[str, np.ndarray], channel_name: str, channel_index: int, channel_name_to_coordinates: dict):
    print(f'Processing channel {channel_name}...')

    channel_raw = raw.copy().pick_channels([channel_name])
    channel_raw.load_data()
    filtered_channel_raw = channel_raw.copy()
    filtered_channel_data = channel_raw.get_data()[0]
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


def extract_spikes_features(subject: Subject, seeg_raw: dict[str, mne.io.Raw], intracranial_spikes_spikes_windows: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    print('Extracting spikes features...')
    if FORCE_CALCULATE_SPIKES_FEATURES or not os.path.exists(subject.paths.subject_channels_spikes_features_path):
        channels_spikes_features = {}
        detected_channel_names = intracranial_spikes_spikes_windows.keys()
        channel_name_to_coordinates = utils.calculate_coordinates(subject)
        index = 0
        for electrode_name, electrode_raw in seeg_raw.items():
            for channel_name in electrode_raw.ch_names:
                if channel_name not in detected_channel_names:
                    continue
                channel_features = channel_processing(
                    subject, electrode_raw, dict(intracranial_spikes_spikes_windows), channel_name, index, channel_name_to_coordinates)

                if channel_features is None:
                    continue

                index += 1
                channels_spikes_features[channel_name] = channel_features

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


def get_flat_features(subject: Subject, seeg_raw: dict[str, mne.io.Raw], intracranial_spikes_spikes_windows: Dict[str, np.ndarray], scalp_spikes_spikes_windows: np.ndarray):
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


def create_raster_plots(subject: Subject, seeg_raw: dict[str, mne.io.Raw], channels_spikes_features: Dict[str, np.ndarray], scalp_spikes_windows: np.ndarray):
    # converting the timestamps to seconds
    channel_spikes = {channel_name: channel_spikes[:, TIMESTAMP_INDEX] / SR for channel_name, channel_spikes in
                      channels_spikes_features.items()}
    eog_channels_spikes = {'EOG1': [spike for spike in scalp_spikes_windows]}
    channel_spikes.update(eog_channels_spikes)

    tmin = tmax = 0
    for channel_name, channel_raw in seeg_raw.items():
        if channel_name:
            tmin = channel_raw.tmin
            tmax = channel_raw.tmax
        break

    # raster plot with hypnogram and histogram
    plotting.create_raster_plot(
        subject=subject,
        spikes=channel_spikes,
        tmin=tmin,
        tmax=tmax,
        add_hypnogram=True,
        add_histogram=True,
    )

    # raster plot with histogram
    plotting.create_raster_plot(
        subject=subject,
        spikes=channel_spikes,
        tmin=tmin,
        tmax=tmax,
        add_hypnogram=False,
        add_histogram=True,
    )


def save_electrodes_coordinates(subject: Subject, raw: dict[str, mne.io.Raw]):
    if subject.stimuli_project:
        stimulation_locations = utils.pars_stimuli_locations_file(subject)
    else:
        stimulation_locations = []
    plotting.save_electrodes_position(raw, subject, stimulation_locations)


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


########################################## stimuli project #############################################################

def baseline_diff(a, b):
    res = ((b - a) / max(b, a)) * 100
    if np.isnan(res):
        return 0
    return res


def get_subjects(filters=None, sort_key=None):
    subjects = [Subject(d, True) for d in os.listdir(Paths.products_data_dir_path) if d.startswith('p')]
    if filters:
        for filt in filters:
            subjects = filter(filt, subjects)
    if sort_key:
        subjects = sorted(list(subjects), key=sort_key)
    return subjects


def get_stimuli_subject_blocks(subj: Subject, only_nrem: bool = True):
    subj_features = np.load(subj.paths.subject_flat_features_path)

    group_ids = subj_features[:, GROUP_INDEX]
    unique_indices = np.unique(group_ids, return_index=True)[1]
    unique_group_subj_features = subj_features[unique_indices]

    # g for groups - the difference is that we take a representative spike for each group
    g_before, g_stim_block, g_pause_block, g_during_window, g_after = utils.stimuli_effects(
        subj, unique_group_subj_features, only_nrem=only_nrem
    )
    before, stim_block, pause_block, during_window, after = utils.stimuli_effects(
        subj, subj_features, only_nrem=only_nrem
    )

    return {
        'before window': before,
        'stim block': stim_block,
        'pause block': pause_block,
        'during window': during_window,
        'after window': after,
        'group before window': g_before,
        'group stim block': g_stim_block,
        'group pause block': g_pause_block,
        'group during window': g_during_window,
        'group after window': g_after,
    }


def get_control_subject_blocks(subj: Subject, stimuli_subjects: Subject, only_nrem: bool = True):
    g_before, g_stim_block, g_pause_block, g_during_window, g_after = utils.control_stimuli_effects(
        subj, stimuli_subjects, only_nrem=only_nrem
    )
    before, stim_block, pause_block, during_window, after = utils.control_stimuli_effects(
        subj, stimuli_subjects, only_nrem=only_nrem
    )
    return {
        'before window': before,
        'stim block': stim_block,
        'pause block': pause_block,
        'during window': during_window,
        'after window': after,
        'group before window': g_before,
        'group stim block': g_stim_block,
        'group pause block': g_pause_block,
        'group during window': g_during_window,
        'group after window': g_after,
    }


def plot_stimuli_effect_with_control(subjects, subjects_stats, subjects_blocks, block_names, feature_id_to_title, show: bool = False, compare_to_base_line: bool = False):
    for subj, stimuli_subjects in subjects.items():
        data_of_blocks = subjects_blocks[subj]

        for feature_index in subjects_stats.keys():
            # calculate the means for each block
            is_group_feature = GROUP_INDEX <= feature_index <= GROUP_EVENT_SPATIAL_SPREAD_INDEX
            prefix = 'group ' if is_group_feature else ''
            block_means = {
                block_name: np.mean(data_of_blocks[prefix+block_name][:, feature_index])
                for block_name in block_names
            }

            if not (any(np.isnan(mean) for mean in block_means.values())):
                subject_types = ['stimuli', 'control']
                subj_type = subject_types[0] if subj == stimuli_subjects else subject_types[1]

                # no need to compare to baseline for group features
                if compare_to_base_line:
                    _block_names = block_names[1:]
                else:
                    _block_names = block_names[:]

                # calculate the block final values
                for block_name in _block_names:
                    # add the block to the stats
                    if not subjects_stats[feature_index].get(block_name):
                        subjects_stats[feature_index][block_name] = {
                            subj_type: [] for subj_type in subject_types
                        }

                    # if compare_to_base_line is True, calculate the difference between the block and the baseline
                    if compare_to_base_line:
                        subjects_stats[feature_index][block_name][subj_type].append(
                            baseline_diff(block_means['before window'], block_means[block_name])
                        )
                    else:
                        subjects_stats[feature_index][block_name][subj_type].append(
                            block_means[block_name]
                        )

    subject = Subject(STIMULI_PROJECT_INTERSUBJECTS_SUBJECT_NAME, True)
    path = os.path.join(subject.paths.subject_stimuli_effects_plots_dir_path, 'control')
    if not os.path.exists(path):
        os.makedirs(path)
    for feature_index, stats in subjects_stats.items():
        plotting.create_box_plot_for_stimuli(
            figure_path=path,
            data_channels=subjects_stats[feature_index],
            feature_name=('Baseline - ' if compare_to_base_line else 'Raw - ') + feature_id_to_title[feature_index],
            show=show,
        )


def stimuli_effects(show: bool = False, control: bool = False, compare_to_base_line: bool = False):
    subjects_stats = {
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

    stimuli_subjects = get_subjects(
        filters=[
            lambda subj: subj.stimuli_project,
            lambda subj: os.path.exists(subj.paths.subject_flat_features_path),
        ],
        sort_key=lambda subj: subj.p_number,
    )
    subjects = {
        s: s for s in stimuli_subjects
    }

    if control:
        control_subjects = get_subjects(
            filters=[
                lambda subj: not subj.stimuli_project,
                lambda subj: not subj.sourasky_project,
                lambda subj: subj.p_number not in [415],
                lambda subj: os.path.exists(subj.paths.subject_flat_features_path),
            ],
            sort_key=lambda subj: subj.p_number,
        )

        if len(control_subjects) > len(stimuli_subjects):
            control_subjects = control_subjects[:len(stimuli_subjects)]

        subjects.update({
            s: stimuli_subjects[i] for i, s in enumerate(control_subjects)
        })

    subjects_blocks = {}
    for subj, stimuli_subjects in subjects.items():
        if subj == stimuli_subjects:
            if os.path.exists(subj.paths.subject_sleep_scoring_path):
                data_of_blocks = get_stimuli_subject_blocks(subj)
            else:
                print(f'stimuli {subj.p_number}')
                continue
        else:
            print(f'control {subj.p_number} with {stimuli_subjects.p_number}')
            if os.path.exists(subj.paths.subject_hypnogram_path):
                data_of_blocks = get_control_subject_blocks(subj, stimuli_subjects)
            else:
                continue
        subjects_blocks[subj] = data_of_blocks

    block_names = ['before window', 'stim block', 'pause block', 'after window']
    if control:
        plot_stimuli_effect_with_control(
            subjects, subjects_stats.copy(), subjects_blocks, block_names, feature_id_to_title, show, compare_to_base_line
        )






