import os

import numpy as np
import pandas as pd

from NirsLabProject.config import consts
from NirsLabProject.config.paths import Paths


class Subject:
    def __init__(self, subject: str, bipolar_model: bool):
        self.name = subject
        self.p_number = int(self.name[1:])
        self.bipolar_model = bipolar_model
        self.paths = Paths(subject, self.bipolar_model)
        if self.p_number >= consts.STIMULI_PROJECT_FIRST_P_NUMBER:
            self.stimuli_project = True
        else:
            self.stimuli_project = False

        if self.p_number < consts.SOURASKY_PROJECT_LAST_P_NUMBER:
            self.sourasky_project = True
        else:
            self.sourasky_project = False

        os.makedirs(self.paths.subject_products_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_plots_dir_path, exist_ok=True)

        if self.name == consts.DETECTION_PROJECT_INTERSUBJECTS_SUBJECT_NAME:
            os.makedirs(self.paths.subject_raincloud_plots_dir_path, exist_ok=True)
            os.makedirs(self.paths.subject_features_3d_plots_dir_path, exist_ok=True)
        elif self.name == consts.STIMULI_PROJECT_INTERSUBJECTS_SUBJECT_NAME:
            os.makedirs(self.paths.subject_stimuli_effects_plots_dir_path, exist_ok=True)
        else:
            os.makedirs(self.paths.subject_resampled_data_dir_path, exist_ok=True)
            os.makedirs(self.paths.subject_spikes_dir_path, exist_ok=True)
            os.makedirs(self.paths.subject_raster_plots_dir_path, exist_ok=True)
            os.makedirs(self.paths.subject_psd_plots_dir_path, exist_ok=True)
            os.makedirs(self.paths.subject_tfr_plots_dir_path, exist_ok=True)
            os.makedirs(self.paths.subject_erp_plots_dir_path, exist_ok=True)
            os.makedirs(self.paths.subject_histogram_plots_dir_path, exist_ok=True)
            os.makedirs(self.paths.subject_electrodes_dir_path, exist_ok=True)
            os.makedirs(self.paths.subject_channels_spikes_features_dir_path, exist_ok=True)
            os.makedirs(self.paths.subject_features_3d_plots_dir_path, exist_ok=True)
            os.makedirs(self.paths.subject_raincloud_plots_dir_path, exist_ok=True)

    def save_flat_features_to_csv(self):
        titles = [name for index, name in consts.FEATURES_NAMES.items()]
        index_to_channel_name = np.load(self.paths.subject_channel_name_to_index_path, allow_pickle=True).item()
        subject_flat_features = np.load(self.paths.subject_flat_features_path)
        df = pd.DataFrame(subject_flat_features, columns=titles)
        index_to_channel_convert = lambda x: index_to_channel_name[x]
        df['channel'] = df['channel'].apply(index_to_channel_convert)
        df['group_focal'] = df['group_focal'].apply(index_to_channel_convert)
        df['group'] = df['group'].apply(lambda x: float(str(x)[3:]))
        df['stimuli_flag'] = df['stimuli_flag'].apply(lambda x: consts.STIMULI_FLAGS_NAMES[x])
        df['hypnogram_flag'] = df['hypnogram_flag'].apply(lambda x: consts.HYPNOGRAM_FLAGS_NAMES[x])
        df['is_in_scalp'] = df['is_in_scalp'].apply(lambda x: x == 1)
        df.to_csv(self.paths.subject_flat_features_dataframe_path, index=False)
