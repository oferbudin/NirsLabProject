import os


# ----------------------------------------------------------------------------------------------------------------------

SR = 1000
DIVISION_FACTOR = 4
SUBJECT = 'p396'


# -------------------------------------------- Paths -------------------------------------------------------------------

project_absolute_path = os.path.dirname(os.path.abspath(__file__))
raw_data_dir_path = os.path.join(project_absolute_path, 'raw_data')
spikes_dir_path = os.path.join(project_absolute_path, 'spikes')
raster_plots_path = os.path.join(project_absolute_path, 'raster_plots')
subject_raw_edf_path = os.path.join(raw_data_dir_path, f'{SUBJECT}_raw.edf')
subject_spikes_path = os.path.join(spikes_dir_path, f'{SUBJECT}_spikes.npz')
subject_raster_plot_path = os.path.join(raster_plots_path, f'{SUBJECT}_raster_plot.png')

# ----------------------------------------------------------------------------------------------------------------------
