import os
from NirsLabProject.config.paths import Paths
from NirsLabProject.config import consts


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

        os.makedirs(self.paths.subject_products_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_resampled_data_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_spikes_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_plots_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_raster_plots_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_psd_plots_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_tfr_plots_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_erp_plots_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_histogram_plots_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_electrodes_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_channels_spikes_features_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_features_3d_plots_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_raincloud_plots_dir_path, exist_ok=True)
