import os
import time
import traceback

import numpy as np

from NirsLabProject.config import consts
from NirsLabProject.config.consts import *
from NirsLabProject.config.paths import Paths
from NirsLabProject.config.subject import Subject
from NirsLabProject.utils import pipeline_utils
from NirsLabProject.utils import scalp_spikes_detection, intracranial_spikes_detection, plotting
from NirsLabProject.utils.google_drive_download import GoogleDriveDownloader


def main(subject_name: str, bipolar_model: bool = True, model_name: str = '', min_z_score: float = MIN_AMPLITUDE_Z_SCORE):
    subject = Subject(subject_name, bipolar_model, model_name, min_z_score)

    # resamples and filters the data
    raw = pipeline_utils.resample_and_filter_data(subject)
    if not raw:
        print(f'Error in resampling and filtering the data of subject {subject_name}')
        return
    if 'EOG' in raw.keys():
        eog_raw = raw.pop('EOG')
    else:
        eog_raw = None

    # detects scalp spikes if the subject is not from the detections project
    if subject.stimuli_project:
        scalp_spikes_spikes_windows = np.array([])
    else:
        scalp_spikes_spikes_windows = np.array([])
        # scalp_spikes_spikes_windows = scalp_spikes_detection.detect_spikes_of_subject(subject, eog_raw)

    # detects intracranial spikes
    intracranial_spikes_spikes_windows = intracranial_spikes_detection.detect_spikes_of_subject(subject, raw)

    flat_features, channels_spikes_features, index_to_channel_name, groups = pipeline_utils.get_flat_features(subject, raw, intracranial_spikes_spikes_windows, scalp_spikes_spikes_windows)
    channel_name_to_index = {name: index for index, name in index_to_channel_name.items()}

    # creates raster plots of the intracranial spikes
    pipeline_utils.create_raster_plots(subject, raw, channels_spikes_features)

    # plot the electrodes coordinates in 3D space
    pipeline_utils.save_electrodes_coordinates(subject, raw)

    # plots the spikes features histograms in 3D space
    plotting.plot_avg_spike_amplitude_by_electrode(subject, channels_spikes_features)
    plotting.plot_number_of_spikes_by_electrode(subject, channels_spikes_features)

    if subject.stimuli_project:
        for electrode_name in raw.keys():
            electrode_raw = raw[electrode_name]
            channel_name = electrode_raw.ch_names[0]
            print(f'Creating erp, tfr and psd plots for {electrode_name} - {channel_name}')
            plotting.create_erp_of_stimuli_and_pause_blocks(subject, flat_features, electrode_raw, channel_name, channel_name_to_index)
            plotting.create_tfr_of_stimuli_and_pause_blocks(subject, flat_features, electrode_raw, channel_name, channel_name_to_index)
            plotting.create_psd_of_stimuli_and_no_stimuli_blocks(subject, flat_features, electrode_raw, channel_name, channel_name_to_index)
        # plots the effects of the stimuli on the spikes features before, during and after the stimuli
        # plotting.stimuli_effects_raincloud_plots(subject, flat_features, index_to_channel)
    else:
        # plots the correlation between the scalp spikes and the intracranial spikes
        plotting.create_raincloud_plot_for_all_spikes_features(subject, flat_features)
        plotting.plot_scalp_detection_probability_for_every_electrode_in_3d(subject, flat_features, index_to_channel_name)
        for electrode_name in raw.keys():
            electrode_raw = raw[electrode_name]
            channel_name = electrode_raw.ch_names[0]
            print(f'Creating erp and tfr plots for {electrode_name} - {channel_name}')
            plotting.create_erp_of_detected_and_not_detected(subject, flat_features, electrode_raw, channel_name, channel_name_to_index)
            plotting.create_tfr_of_detected_and_not_detected(subject, flat_features, electrode_raw, channel_name, channel_name_to_index)
        plotting.create_eog_tfr(subject, flat_features, eog_raw, 'LH1', channel_name_to_index)
        plotting.create_eog_erp(subject, flat_features, eog_raw, 'LH1', channel_name_to_index)


# subjects_names can be a list of subjects that have files in Google Drive
# or None to download all subjects
# e.g. run_all_detection_project(['p1', 'p2'])
def run_all_stimuli_project(subjects_names: list = None, model_name: str = ''):
    gdd = GoogleDriveDownloader()
    for p in gdd.download_subject_data_one_by_one(consts.STIMULI_PROJECT_GOOGLE_DRIVE_LINK, subjects_names):
        try:
            for min_z_score in [1, 2]:
                print(f'Processing {p} with max z score {min_z_score}')
                main(p.name, model_name=model_name, bipolar_model=False, min_z_score=min_z_score)
        except Exception as e:
            print(f'Failed to process {p.name} due to {traceback.format_exc()}')


# subjects_names can be a list of subjects that have files in Google Drive
# or None to download all subjects
# e.g. run_all_detection_project(['p1', 'p2'])
def run_all_detection_project(subjects_names: list = None, model_name: str = ''):
    gdd = GoogleDriveDownloader()
    for p in gdd.download_subject_data_one_by_one(consts.DETECTION_PROJECT_GOOGLE_FRIVE_LINK, subjects_names):
        try:
            for min_z_score in [1, 2]:
                print(f'Processing {p} with max z score {min_z_score}')
                main(p.name, model_name=model_name, bipolar_model=False, min_z_score=min_z_score)
        except Exception as e:
            print(f'Failed to process {p.name} due to {traceback.format_exc()}')

if __name__ == '__main__':
    start_time = time.time()
    main('p496', bipolar_model=False, model_name='lgbm_full_f15_s25_b_V5.pkl')
    # subjects_names = ['p5101', 'p5107', 'p545', 'p544', 'p538', 'p520', 'p515', 'p505', 'p499', 'p497', 'p496', 'p490', 'p489', 'p489', 'p489']
    subjects_names = []
    run_all_stimuli_project(model_name='lgbm_full_f15_s25_b_V5.pkl', subjects_names=subjects_names)

    # subjects_names = ['p5101', 'p5107', 'p545', 'p544', 'p538', 'p520', 'p515', 'p505', 'p499', 'p497', 'p496', 'p490', 'p489', 'p489', 'p489']
    # subjects_names = ['p5101', 'p5107', 'p545', 'p544', 'p538', 'p520', 'p515', 'p505', 'p499', 'p498','p497', 'p496', 'p490', 'p489', 'p488', 'p487', 'p486']
    # run_all_stimuli_project(model_name='lgbm_full_f15_s25_b_V5.pkl', subjects_names=subjects_names)

    # pipeline_utils.detection_project_intersubjects_plots(True)
    pipeline_utils.stimuli_effects(control=True, compare_to_base_line=True)
    pipeline_utils.stimuli_effects(control=True, compare_to_base_line=False)

    print(f'Time taken: {(time.time() - start_time) / 60} minutes')
