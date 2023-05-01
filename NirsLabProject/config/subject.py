import os
from NirsLabProject.config.paths import Paths


class Subject:
    def __init__(self, subject: str):
        self.name = subject
        self.paths = Paths(subject)

        os.makedirs(self.paths.subject_plots_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_raster_plots_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_psd_plots_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_tfr_plots_dir_path, exist_ok=True)
        os.makedirs(self.paths.subject_erp_plots_dir_path, exist_ok=True)
