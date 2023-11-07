import os


class Paths:
    project_absolute_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(project_absolute_path, '../data')
    models_dir_path = os.path.join(data_dir_path, 'models')
    raw_data_dir_path = os.path.join(data_dir_path, 'raw_data')
    stimuli_dir_path = os.path.join(data_dir_path, 'stimuli')
    hypnogram_data_dir_path = os.path.join(data_dir_path, 'hypnograms')
    products_data_dir_path = os.path.join(data_dir_path, 'products')

    def __init__(self, subject: str, bipolar: bool):
        self.subject_products_dir_path = os.path.join(self.products_data_dir_path, subject)
        self.subject_products_dir_path_by_model = os.path.join(self.subject_products_dir_path, 'bipolar_model' if bipolar else 'unichannel_model')
        self.subject_plots_dir_path = os.path.join(self.subject_products_dir_path_by_model, 'plots')
        self.subject_spikes_dir_path = os.path.join(self.subject_products_dir_path_by_model, 'spikes')
        self.subject_resampled_data_dir_path = os.path.join(self.subject_products_dir_path, 'resampled')
        self.subject_raster_plots_dir_path = os.path.join(self.subject_plots_dir_path, "raster")
        self.subject_erp_plots_dir_path = os.path.join(self.subject_plots_dir_path, "ERPs")
        self.subject_psd_plots_dir_path = os.path.join(self.subject_plots_dir_path, "PSDs")
        self.subject_tfr_plots_dir_path = os.path.join(self.subject_plots_dir_path, "TFRs")
        self.subject_histogram_plots_dir_path = os.path.join(self.subject_plots_dir_path, "Histograms")
        self.subject_raw_edf_path = os.path.join(self.raw_data_dir_path, f'{subject}.edf')
        self.subject_raw_fif_path = os.path.join(self.raw_data_dir_path, f'{subject}.fif')
        self.subject_stimuli_path = os.path.join(self.stimuli_dir_path, f'{subject}_stim_timing.csv')
        self.subject_resampled_fif_path = os.path.join(self.subject_resampled_data_dir_path, f'{subject}_resampled.fif')
        self.subject_hypnogram_path = os.path.join(self.hypnogram_data_dir_path, f'{subject}_hypno.txt')
        self.subject_sleep_scoring_path = os.path.join(self.hypnogram_data_dir_path, f'{subject}_sleep_scoring.m')
        self.subject_spikes_path = os.path.join(self.subject_spikes_dir_path, f'{subject}_spikes.npz')
        self.subject_raster_plot_path = os.path.join(self.subject_raster_plots_dir_path, f'{subject}_raster_plot.png')
        self.subject_hypno_raster_plot_path = os.path.join(self.subject_raster_plots_dir_path, f'{subject}_hypno_raster_plot.png')
        self.subject_cut_hypno_raster_plot_path = os.path.join(self.subject_raster_plots_dir_path, f'{subject}_cut_hypno_raster_plot.png')
