import os


class Paths:
    project_absolute_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(project_absolute_path, '../data')
    models_dir_path = os.path.join(data_dir_path, 'models')
    raw_data_dir_path = os.path.join(data_dir_path, 'raw_data')
    resampled_data_dir_path = os.path.join(data_dir_path, 'resampled')
    hypnogram_data_dir_path = os.path.join(data_dir_path, 'hypnograms')
    spikes_dir_path = os.path.join(project_absolute_path, '../data/spikes')
    plots_dir_path = os.path.join(data_dir_path, 'plots')

    def __init__(self, subject: str):
        self.subject_plots_dir_path = os.path.join(self.plots_dir_path, subject)
        self.subject_raster_plots_dir_path = os.path.join(self.subject_plots_dir_path, "raster")
        self.subject_erp_plots_dir_path = os.path.join(self.subject_plots_dir_path, "ERPs")
        self.subject_psd_plots_dir_path = os.path.join(self.subject_plots_dir_path, "PSDs")
        self.subject_tfr_plots_dir_path = os.path.join(self.subject_plots_dir_path, "TFRs")
        self.subject_raw_edf_path = os.path.join(self.raw_data_dir_path, f'{subject}_raw.edf')
        self.subject_resampled_fif_path = os.path.join(self.resampled_data_dir_path, f'{subject}_resampled.fif')
        self.subject_hypnogram_path = os.path.join(self.hypnogram_data_dir_path, f'{subject}_hypno.txt')
        self.subject_spikes_path = os.path.join(self.spikes_dir_path, f'{subject}_spikes.npz')
        self.subject_raster_plot_path = os.path.join(self.subject_raster_plots_dir_path, f'{subject}_raster_plot.png')
        self.subject_hypno_raster_plot_path = os.path.join(self.subject_raster_plots_dir_path, f'{subject}_hypno_raster_plot.png')
