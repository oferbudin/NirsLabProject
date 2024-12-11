import os
import warnings
import mne


from NirsLabProject.config.paths import Paths
from NirsLabProject.config import consts

mne.set_log_level('WARNING')
warnings.filterwarnings('ignore')

os.makedirs(Paths.raw_data_dir_path, exist_ok=True)
os.makedirs(Paths.stimuli_dir_path, exist_ok=True)
os.makedirs(Paths.bad_channels_dir_path, exist_ok=True)
os.makedirs(Paths.models_dir_path, exist_ok=True)
os.makedirs(Paths.products_data_dir_path, exist_ok=True)
os.makedirs(Paths.hypnogram_data_dir_path, exist_ok=True)
