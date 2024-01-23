import os
import re
import traceback

import mne
import numpy as np
import pandas as pd
from functools import partial
from typing import Tuple, Dict, List
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
import scipy.stats as sp_stats
from collections import Counter


from NirsLabProject.config.consts import *
from NirsLabProject.config.paths import Paths
from NirsLabProject.config.subject import Subject
from NirsLabProject.utils import sleeping_utils


COORDINATES = {}


def clean_channel_name(subject: Subject, channel_name: str) -> str:
    if subject.sourasky_project:
        channel_name = foramt_sourasky_patients_channel_names(channel_name)
    # Remove the '-REF' or '-REF1' from the channel name
    return channel_name.replace('-REF1', '').replace('SEEG ', '').replace('-REF', '').replace('-REF1', '')


def get_spike_amplitude_and_length(data: np.ndarray, peak_index) -> Dict:
    # taking the spike range (SPIKE_RANGE_SECONDS) before and after the peak timestamp of the window
    spike_range_in_indexes = int(SPIKE_RANGE_SECONDS * SR)
    start_frame_index = max(0, peak_index-spike_range_in_indexes)
    end_frame_index = min(len(data), peak_index+spike_range_in_indexes)

    amplitude = data[peak_index]
    # finding the closest point before the peak with half of the peak amplitude
    first_half_start = start_frame_index + np.where(data[start_frame_index:peak_index] < amplitude / 2)[0]
    if len(first_half_start) > 0:
        first_half_start = first_half_start[-1]
    else:
        # if there is no point before the peak with half of the peak amplitude, we will take the start of the window
        return {'index': peak_index, 'amplitude': amplitude, 'length': -1}

    # finding the closest point after the peak with half of the peak amplitude
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
    wide_spike_window_data, wide_spike_window_start_index, _end = get_data_of_spike_in_wide_window(
        channel_data, spike_window_timestamp)

    # Find the peak of the spike in the wide_spike_window_data with length smaller than MAX_SPIKE_LENGTH_MILLISECONDS
    peaks, _ = find_peaks(wide_spike_window_data)
    spikes = map(lambda p: get_spike_amplitude_and_length(wide_spike_window_data, p), peaks)
    filtered_spikes = list(
        filter(lambda x: MIN_SPIKE_LENGTH_MILLISECONDS < x['length'] <= MAX_SPIKE_LENGTH_MILLISECONDS, spikes)
    )
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
    # Get the spike index in the channel data for each spike in the spikes array
    spikes = np.vectorize(partial(get_spike_amplitude_index, channel_data))(spikes)
    # Removes epochs that have no spike with length that smaller than  MAX_SPIKE_LENGTH_MILLISECONDS
    spikes = spikes[spikes >= 0]
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


def clean_channels_name_in_raw_obj(subject: Subject, raw: mne.io.Raw):
    print('Cleaning channels names')
    mne.rename_channels(raw.info, {ch: clean_channel_name(subject, ch) for ch in raw.ch_names})


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
    seeg_channels = [ch for ch in seeg_channels if len(ch) > 2 and not ch.startswith('EEG')]
    return raw.pick_channels([ch for ch in seeg_channels if len(ch) > 2])


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
            print(f'function {func.__name__} with args {args}, kwargs {kwargs} failed: {traceback.format_exc()}')
            # raise e
    return wrapper


def calculate_coordinates_sourasky(subject: Subject):
    data = pd.read_csv(
        filepath_or_buffer=Paths.sourasky_coordinates_path,
        delimiter=','
    )

    subject_name = f'D0{subject.p_number}' if subject.p_number < 100 else f'D{subject.p_number}'
    subjects_data = data[data['Subject'] == subject_name]
    if subjects_data.empty:
        return {}

    coords = {}
    for i, row in subjects_data.iterrows():
        coords[row['Channel Name'].replace("'", "")] = (row['MNI_X'], row['MNI_Y'], row['MNI_Z'])

    return coords


def read_coordinates_files(electrodes_name_file_path: str, electrodes_location_file_path: str):
    name_data = pd.read_csv(
        filepath_or_buffer=electrodes_name_file_path,
        delimiter=' ',
        usecols=[0],
        skiprows=2,
        dtype={0: str}
    )

    location_data = pd.read_csv(
        filepath_or_buffer=electrodes_location_file_path,
        delimiter=' ',
        skiprows=2,
        header=None,
        dtype={0: float, 1: float, 2: float}
    )

    name_to_coordinates = {}
    for names_row, location_row in zip(name_data.iterrows(), location_data.iterrows()):
        electrode_name = f'{names_row[1][0]}'
        name_to_coordinates[electrode_name] = (
            location_row[1][0],
            location_row[1][1],
            location_row[1][2]
        )
    return name_to_coordinates


def calculate_coordinates(subject: Subject):
    print(f'calculating coordinates for subject {subject.p_number}')
    if subject.sourasky_project:
        return calculate_coordinates_sourasky(subject)
    if subject.p_number in [5101, 5107]:
        subject = Subject('p510', subject.bipolar_model)
    if os.path.exists(subject.paths.subject_electrode_name_file) and os.path.isfile(subject.paths.subject_electrode_locations):
        return read_coordinates_files(subject.paths.subject_electrode_name_file, subject.paths.subject_electrode_locations)

    print('no coordinates file found, calculating coordinates from average coordinates files')
    _sum = {}
    count = {}
    for file in [file for file in os.listdir(Paths.coordinates_data_dir_path) if file.endswith('.electrodeNames')]:
        loc_file_path = os.path.join(Paths.coordinates_data_dir_path, file)
        pial_file_path = os.path.join(Paths.coordinates_data_dir_path, file.replace('.electrodeNames', '.Pial'))
        electrode_name_to_location = read_coordinates_files(loc_file_path, pial_file_path)
        for electrode_name, location in electrode_name_to_location.items():
            if electrode_name not in _sum:
                _sum[electrode_name] = [0, 0, 0]
                count[electrode_name] = 0
            count[electrode_name] += 1
            _sum[electrode_name][0] += location[0]
            _sum[electrode_name][1] += location[1]
            _sum[electrode_name][2] += location[2]

    for electrode in _sum:
        _sum[electrode][0] /= count[electrode]
        _sum[electrode][1] /= count[electrode]
        _sum[electrode][2] /= count[electrode]

    return _sum


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
        if 'EOG' in channel_name:
            continue

        channel_area = extract_channel_name_and_contact_number(channel_name)[0]
        if channel_area not in area_norms:
            print(f'calculating norms for electrode {channel_area}')
            area_norms[channel_area] = []
            area_norms_stds[channel_area] = []
            area_norms_skew[channel_area] = []
            area_norms_kurtosis[channel_area] = []

        _, contact_number = extract_channel_name_and_contact_number(channel_name)
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
        if i == len(channel_names) - 1 or channel_names[i + 1][-1] < contact_number:
            for channel_area, channels in area_norms.items():
                area_norms_stds_median = np.median(area_norms_stds[channel_area])
                area_norms_skew_median = np.median(area_norms_skew[channel_area])
                area_norms_kurtosis_median = np.median(area_norms_kurtosis[channel_area])

                for j, _ in enumerate(channels):
                    # currently the only feature that in use
                    if area_norms_stds[channel_area][j] > area_norms_stds_median * 4:
                        channel_features[total_channel_index][0] = 2
                    elif area_norms_stds[channel_area][j] > area_norms_stds_median * 2:
                        channel_features[total_channel_index][0] = 1.5
                    elif area_norms_stds[channel_area][j] > area_norms_stds_median * 1.5:
                        channel_features[total_channel_index][0] = 1

                    if area_norms_skew[channel_area][j] > area_norms_skew_median * 4:
                        channel_features[total_channel_index][1] = 2
                    elif area_norms_skew[channel_area][j] > area_norms_skew_median * 2:
                        channel_features[total_channel_index][1] = 1

                    if area_norms_kurtosis[channel_area][j] > area_norms_kurtosis_median * 4:
                        channel_features[total_channel_index][2] = 2
                    elif area_norms_kurtosis[channel_area][j] > area_norms_kurtosis_median * 2:
                        channel_features[total_channel_index][2] = 1

                    total_channel_index += 1
            area_norms = {}

    bad_channels_name = []
    for channel_name, features in zip(raw.ch_names, channel_features):
        if features.sum() > 2 or features[0] >= 1:
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
        print(f'Could not find the stimulating electrode for subject {subject.p_number}')
        return []
    electrode_name_and_parts = stimulating_electrode.split('-')
    electrode_name = electrode_name_and_parts[0].upper()
    electrodes = [electrode_name + str(i) for i in electrode_name_and_parts[1:]]
    return electrodes


def add_flag_of_scalp_detection_to_spikes_features(flat_features: np.ndarray, scalp_spikes_spikes_windows: np.ndarray):
    indexes = np.zeros((flat_features.shape[0], 1))
    if scalp_spikes_spikes_windows is None:
        return np.concatenate((flat_features, indexes), axis=1)
    for spike in scalp_spikes_spikes_windows:
        window_start = spike * 1000
        match_spikes = np.where(
            np.logical_and(
                window_start - 100 < flat_features[:, TIMESTAMP_INDEX],
                flat_features[:, TIMESTAMP_INDEX] < window_start + int(SPIKE_RANGE_SECONDS*1000)
            )
        )
        indexes[match_spikes[0]] = 1
    return np.concatenate((flat_features, indexes), axis=1)


def _add_stimuli_flag_to_spikes_features(stimuli_windows, flat_features):
    flags = np.zeros((flat_features.shape[0], 1))
    stimuli_session_start = stimuli_windows[0][0] * SR
    stimuli_session_end = stimuli_windows[-1][1] * SR

    flags[
        np.where(
            np.logical_and(
                stimuli_session_start < flat_features[:, TIMESTAMP_INDEX],
                flat_features[:, TIMESTAMP_INDEX] < stimuli_session_end
            )
        )
    ] = STIMULI_FLAG_STIMULI_SESSION

    flags[
        np.where(
            flat_features[:, TIMESTAMP_INDEX] < stimuli_session_start
        )
    ] = STIMULI_FLAG_BEFORE_FIRST_STIMULI_SESSION

    flags[
        np.where(
            flat_features[:, TIMESTAMP_INDEX] > stimuli_session_end
        )
    ] = STIMULI_FLAG_AFTER_STIMULI_SESSION

    for session in stimuli_windows:
        window_start = session[0] * SR
        window_end = session[1] * SR
        flags[
            np.where(
                np.logical_and(
                    window_start < flat_features[:, TIMESTAMP_INDEX],
                    flat_features[:, TIMESTAMP_INDEX] < window_end
                )
            )
        ] = STIMULI_FLAG_DURING_STIMULI_BLOCK
    return flags


def add_stimuli_flag_to_spikes_features(subject: Subject, flat_features: np.ndarray) -> np.ndarray:
    if subject.stimuli_project:
        stimuli_windows = get_stimuli_time_windows(subject)
        flags = _add_stimuli_flag_to_spikes_features(stimuli_windows, flat_features)
    else:
        flags = np.zeros((flat_features.shape[0], 1))

    return np.concatenate((flat_features, flags), axis=1)


# No validation for stimuli subject
def add_stimuli_flag_from_another_subject_to_spikes_features(stimuli_subject: Subject,  flat_features: np.ndarray) -> np.ndarray:
    stimuli_windows = get_stimuli_time_windows(stimuli_subject)
    flags = _add_stimuli_flag_to_spikes_features(stimuli_windows, flat_features)
    flat_features[:, STIMULI_FLAG_INDEX] = flags[:, 0]
    return flat_features


def add_sleeping_stage_flag_to_spike_features(subject: Subject, flat_features: np.ndarray) -> np.ndarray:
    flags = np.zeros((flat_features.shape[0], 1))
    try:
        changing_points, values = sleeping_utils.get_hypnogram_changes_in_miliseconds(subject)
    except Exception as e:
        print(f'Could not get sleeping stages for subject {subject.p_number}, error: {e}')
        flags[:] = np.NAN
        return np.concatenate((flat_features, flags), axis=1)
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


def control_stimuli_effects(control_subject: Subject, stimuli_subject: Subject, group: bool = False, only_nrem: bool = True, multi_chnannel_event=False):
    # load control_subject features as the record started 30 minutes from the beginning
    # of the first sleeping cycle
    control_offset_from_beginning = (30 * 60) * SR
    control_offset_from_beginning += sleeping_utils.get_sleep_start_end_indexes(control_subject)[0]
    control_subject_features = np.load(control_subject.paths.subject_flat_features_path)
    control_subject_features[:, TIMESTAMP_INDEX] -= control_offset_from_beginning
    control_subject_features = control_subject_features[control_subject_features[:, TIMESTAMP_INDEX] >= 0]
    control_subject_features = add_stimuli_flag_from_another_subject_to_spikes_features(stimuli_subject, control_subject_features)

    if group:
        group_ids = control_subject_features[:, GROUP_INDEX]
        unique_indices = np.unique(group_ids, return_index=True)[1]
        control_subject_features = control_subject_features[unique_indices]
        if multi_chnannel_event:
            control_subject_features = control_subject_features[
                control_subject_features[:, GROUP_EVENT_SIZE_INDEX] > 2
                ]

    return stimuli_effects(control_subject, control_subject_features, only_nrem)


def stimuli_effects(subject: Subject, flat_features: np.ndarray, only_nrem: bool = True):
    sleep_start = sleeping_utils.get_sleep_start_end_indexes(subject)

    baseline_features = flat_features[flat_features[:, STIMULI_FLAG_INDEX] == STIMULI_FLAG_BEFORE_FIRST_STIMULI_SESSION]
    stimuli_blocks_features = flat_features[flat_features[:, STIMULI_FLAG_INDEX] == STIMULI_FLAG_DURING_STIMULI_BLOCK]
    stimuli_session_features = flat_features[flat_features[:, STIMULI_FLAG_INDEX] == STIMULI_FLAG_STIMULI_SESSION]

    start_of_stimuli_window = min(
        stimuli_blocks_features[0, TIMESTAMP_INDEX],
        stimuli_session_features[0, TIMESTAMP_INDEX],
    )

    end_of_stimuli_window = max(
        stimuli_blocks_features[-1, TIMESTAMP_INDEX],
        stimuli_session_features[-1, TIMESTAMP_INDEX],
    )

    # Before Stimuli
    before_stimuli = baseline_features[
        baseline_features[:, TIMESTAMP_INDEX] > sleep_start[0],
    ]

    if only_nrem:
        before_stimuli = before_stimuli[
            before_stimuli[:, HYPNOGRAM_FLAG_INDEX] == HYPNOGRAM_FLAG_NREM,
        ]

    # Stimuli Block
    if only_nrem:
        stim_blocks = stimuli_blocks_features[
                stimuli_blocks_features[:, HYPNOGRAM_FLAG_INDEX] == HYPNOGRAM_FLAG_NREM,
        ]
    else:
        stim_blocks = stimuli_blocks_features

    # During Session
    if only_nrem:
        during_session = flat_features[
             flat_features[:, HYPNOGRAM_FLAG_INDEX] == HYPNOGRAM_FLAG_NREM,
        ]
    else:
        during_session = flat_features

    during_session = during_session[
        during_session[:, TIMESTAMP_INDEX] <= end_of_stimuli_window,
    ]
    during_session = during_session[
        during_session[:, TIMESTAMP_INDEX] >= start_of_stimuli_window,
    ]

    # Pause Block
    if only_nrem:
        pause_block = during_session[
            during_session[:, HYPNOGRAM_FLAG_INDEX] == HYPNOGRAM_FLAG_NREM,
        ]
    else:
        pause_block = during_session

    pause_block = pause_block[
        pause_block[:, STIMULI_FLAG_INDEX] != STIMULI_FLAG_DURING_STIMULI_BLOCK,
    ]
    pause_block = pause_block[
        pause_block[:, STIMULI_FLAG_INDEX] != STIMULI_FLAG_DURING_STIMULI_BLOCK,
    ]

    # After Stimuli
    if only_nrem:
        after_stimuli = flat_features[
            flat_features[:, HYPNOGRAM_FLAG_INDEX] == HYPNOGRAM_FLAG_NREM,
        ]
    else:
        after_stimuli = flat_features

    after_stimuli = after_stimuli[
        after_stimuli[:, STIMULI_FLAG_INDEX] == STIMULI_FLAG_AFTER_STIMULI_SESSION,
    ]
    baseline_duration = before_stimuli[-1, TIMESTAMP_INDEX] - before_stimuli[0, TIMESTAMP_INDEX]

    after_stimuli = after_stimuli[
        after_stimuli[:, TIMESTAMP_INDEX] < (after_stimuli[:, TIMESTAMP_INDEX][0] + baseline_duration),
    ]

    return before_stimuli, stim_blocks, pause_block, during_session, after_stimuli


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


def remove_subject_number_from_indexes(subject_number: int, indexes: np.ndarray) -> np.ndarray:
    subject_number = int(float(subject_number))
    for i in range(len(indexes)):
        indexes[i] = float(str(indexes[i]).replace(f'{subject_number}', ''))
    return indexes


def calculate_sub_group_probabilities_3d(subject_number: int, group_of_indexes: np.ndarray,
                                         sub_group_of_indexes: np.ndarray) -> dict:
    group = remove_subject_number_from_indexes(subject_number, group_of_indexes)
    sub_group = remove_subject_number_from_indexes(subject_number, sub_group_of_indexes)

    # Compute frequency counts for sub_g (sub group) and group (big group) that contains sub_g
    sub_group_bins = np.arange(np.min(sub_group), np.max(sub_group) + 2)
    sub_group_counts = np.histogram(sub_group, bins=sub_group_bins)[0]

    # Compute frequency counts for group and sub_group
    group_bins = np.arange(np.min(group), np.max(group) + 2)
    group_counts = np.histogram(group, bins=group_bins)[0]

    # Create histograms for sub_group and group
    sub_group_hist = {n: count for n, count in zip(sub_group_bins, sub_group_counts)}
    group_hist = {n: count for n, count in zip(group_bins, group_counts)}

    # Compute probabilities for sub_group and group that contains sub_g
    return {add_subject_number_to_index(subject_number, n): (sub_group_hist.get(n, 0) / count) for n, count in
            group_hist.items()}


def add_subject_number_to_index(subject_number: int, index: int) -> int:
    subject_number = int(subject_number)
    index = int(index)
    return int(f'{subject_number}{index}')


def foramt_sourasky_patients_channel_names(chanel_name):

    name_formatting_dict = {
        "LAH": "LH",
        "LSTGa": "LSTG",
        "LEC": "LECa",
        "LPHG": "LpPH"
        ""
    }

    channel_name = chanel_name.replace('_lfp', '')
    name, number = extract_channel_name_and_contact_number(channel_name)
    name = re.sub(r'\d', '', name)
    if name in name_formatting_dict:
        name = name_formatting_dict[name]
    name = name.replace('-', '')
    return name + number


def extract_channel_name_and_contact_number(channel_name: str):
    search = re.search(r'(\S*\D)(\d*)$', channel_name)
    return search.group(1), str(int(search.group(2)))
