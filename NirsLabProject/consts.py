import os


# ----------------------------------------------------------------------------------------------------------------------

SR = 1000
subj = 'p402'

# -------------------------------------------- Paths -------------------------------------------------------------------

project_absolute_path = os.path.dirname(os.path.abspath(__file__))
raw_data_dir_path = os.path.join(project_absolute_path, 'raw_data')
spikes_dir_path = os.path.join(project_absolute_path, 'spikes')
subject_raw_edf_path = os.path.join(raw_data_dir_path, f'{subj}_raw.edf')
subject_spikes_path = os.path.join(spikes_dir_path, f'{subj}_spikes.npz')

# ----------------------------------------------------------------------------------------------------------------------
