import mne
import numpy as np
import scipy.stats as sp_stats
from joblib import Parallel, delayed

from NirsLabProject.config.consts import *
from NirsLabProject.config.subject import Subject
from NirsLabProject import utils


def __process_channel(subject: Subject, raw: mne.io.Raw, channel_name: str, channel_index: int) -> np.ndarray:
    channel_raw = raw.copy().pick_channels([channel_name])
    channel_raw.load_data()
    channel_raw = channel_raw.filter(l_freq=LOW_THRESHOLD_FREQUENCY, h_freq=HIGH_THRESHOLD_FREQUENCY, n_jobs=2)
    channel_data = channel_raw.get_data()
    channel_data = np.apply_along_axis(sp_stats.zscore, 1, channel_data)[0]
    original_spikes = np.load(subject.paths.subject_spikes_path)[channel_name]
    spikes = utils.get_real_spikes_indexs(channel_data, original_spikes)
    spikes = np.concatenate([spikes, np.full((spikes.shape[0], 1), channel_index)], axis=1)
    return spikes

def

index_to_channel = {i:name for i, name in enumerate(raw.ch_names)}

import time
s = time.time()
all_spikes = Parallel(n_jobs=-1)(delayed(process_channel)(name, index) for index, name in enumerate(raw.ch_names))