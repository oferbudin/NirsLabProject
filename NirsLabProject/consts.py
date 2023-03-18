import os


# ----------------------------------------------------------------------------------------------------------------------

SR = 1000  # TODO: ask if it is the correct sampling rate, looks like that there are 2000 samples in a second?
DIVISION_FACTOR = 4
SUBJECT = 'p402'


# -------------------------------------------- Paths -------------------------------------------------------------------

project_absolute_path = os.path.dirname(os.path.abspath(__file__))
raw_data_dir_path = os.path.join(project_absolute_path, 'raw_data')
spikes_dir_path = os.path.join(project_absolute_path, 'spikes')
subject_raw_edf_path = os.path.join(raw_data_dir_path, f'{SUBJECT}_raw.edf')
subject_spikes_path = os.path.join(spikes_dir_path, f'{SUBJECT}_spikes.npz')

# ----------------------------------------------------------------------------------------------------------------------
