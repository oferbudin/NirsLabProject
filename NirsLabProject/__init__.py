import os
import warnings

from NirsLabProject.config.paths import Paths
from NirsLabProject.config import consts


warnings.filterwarnings('ignore')

os.makedirs(Paths.raw_data_dir_path, exist_ok=True)
os.makedirs(Paths.stimuli_dir_path, exist_ok=True)
os.makedirs(Paths.models_dir_path, exist_ok=True)
os.makedirs(Paths.products_data_dir_path, exist_ok=True)
os.makedirs(Paths.hypnogram_data_dir_path, exist_ok=True)
