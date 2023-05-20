import os
from NirsLabProject.config.paths import Paths


class Subject:
    def __init__(self, subject: str, bipolar_model: bool):
        self.name = subject
        self.bipolar_model = bipolar_model
        self.paths = Paths(subject, self.bipolar_model)

        os.makedirs(self.paths.subject_products_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_resampled_data_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_spikes_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_plots_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_raster_plots_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_psd_plots_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_tfr_plots_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_erp_plots_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_histogram_plots_dir_path, exist_ok=True)
