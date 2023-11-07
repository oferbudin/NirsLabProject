import os
import time

import numpy as np

from NirsLabProject.config import consts
from NirsLabProject.config.consts import *
from NirsLabProject.config.paths import Paths
from NirsLabProject.config.subject import Subject
from NirsLabProject.utils import pipeline_utils
from NirsLabProject.utils import scalp_spikes_detection, intracranial_spikes_detection, plotting
from NirsLabProject.utils.google_drive_download import download_subject_data_one_by_one


def main(subject_name: str):
    subject = Subject(subject_name, True)

    # resamples and filters the data
    seeg_raw, eog_raw = pipeline_utils.resample_and_filter_data(subject)

    # detects scalp spikes if the subject is not from the detections project
    if subject.stimuli_project:
        scalp_spikes_spikes_windows = np.array([])
    else:
        scalp_spikes_spikes_windows = scalp_spikes_detection.detect_spikes_of_subject(subject, eog_raw)

    # detects intracranial spikes
    intracranial_spikes_spikes_windows = intracranial_spikes_detection.detect_spikes_of_subject(subject, seeg_raw)

    flat_features, channels_spikes_features, index_to_channel, groups = pipeline_utils.get_flat_features(subject, seeg_raw, intracranial_spikes_spikes_windows, scalp_spikes_spikes_windows)

    # creates raster plots of the intracranial spikes
    pipeline_utils.create_raster_plots(subject, seeg_raw, channels_spikes_features, scalp_spikes_spikes_windows)

    # plot the electrodes coordinates in 3D space
    pipeline_utils.save_electrodes_coordinates(subject, seeg_raw)

    # plots the spikes features histograms in 3D space
    plotting.plot_avg_spike_amplitude_by_electrode(subject, channels_spikes_features)
    plotting.plot_number_of_spikes_by_electrode(subject, channels_spikes_features)

    if subject.stimuli_project:
        pass
        # plots the effects of the stimuli on the spikes features before, during and after the stimuli
        # plotting.stimuli_effects_raincloud_plots(subject, flat_features, index_to_channel)
    else:
        # plots the correlation between the scalp spikes and the intracranial spikes
        plotting.create_raincloud_plot_for_all_spikes_features(subject, flat_features)
        plotting.plot_scalp_detection_probability_for_every_electrode_in_3d(subject, flat_features, index_to_channel)


# subjects_names can be a list of subjects that have files in Google Drive
# or None to download all subjects
# e.g. run_all_detection_project(['p1', 'p2'])
def run_all_stimuli_project(subjects_names: list = None):
    for p in download_subject_data_one_by_one(consts.STIM_PROJECT_GOOGLE_FRIVE_LINK, subjects_names):
        print(f'Processing {p}')
        try:
            main(p.name)
        except Exception as e:
            print(f'Failed to process {p.name} due to {e}')


# subjects_names can be a list of subjects that have files in Google Drive
# or None to download all subjects
# e.g. run_all_detection_project(['p1', 'p2'])
def run_all_detection_project(subjects_names: list = None):
    for p in download_subject_data_one_by_one(consts.DETECTION_PROJECT_GOOGLE_FRIVE_LINK, subjects_names):
        print(f'Processing {p}')
        try:
            main(p.name)
        except Exception as e:
            print(f'Failed to process {p.name} due to {e}')


if __name__ == '__main__':
    start_time = time.time()

    # for p in [p.split('.')[0] for p in os.listdir(Paths.raw_data_dir_path) if p.startswith('p')]:
    #         print(f'Processing {p}')
    #         main(p)

    run_all_stimuli_project()

    pipeline_utils.detection_project_intersubjects_plots(True)
    pipeline_utils.stimuli_effects()

    print(f'Time taken: {(time.time() - start_time) / 60} minutes')
