import os
import warnings

from NirsLabProject.config.paths import Paths
from NirsLabProject.config import consts


warnings.filterwarnings('ignore')

os.makedirs(Paths.raw_data_dir_path, exist_ok=True)
os.makedirs(Paths.resampled_data_dir_path, exist_ok=True)
os.makedirs(Paths.hypnogram_data_dir_path, exist_ok=True)
os.makedirs(Paths.spikes_dir_path, exist_ok=True)
os.makedirs(Paths.plots_dir_path, exist_ok=True)