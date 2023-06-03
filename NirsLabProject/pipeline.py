import os

import mne
import time
from typing import List, Dict
import scipy.stats as sp_stats
from joblib import Parallel, delayed


import numpy as np

from NirsLabProject import spikes_detection
from NirsLabProject import plotting
from NirsLabProject import utils
from NirsLabProject.config.subject import Subject
from NirsLabProject.group_spikes import group_spikes
from NirsLabProject.config.consts import *


def resample_and_filter_data(subject: Subject):
    if not os.path.exists(subject.paths.subject_resampled_fif_path):
        raw = mne.io.read_raw_edf(subject.paths.subject_raw_edf_path)
        utils.pick_seeg_channels(raw)
        utils.clean_channels_name(raw)
        print(f'Resampling data, it might take some time... (around {len(raw.ch_names) * 4 // 60} minutes)')
        raw.resample(SR, n_jobs=2)
        raw.save(subject.paths.subject_resampled_fif_path)
    return mne.io.read_raw_fif(subject.paths.subject_resampled_fif_path)


def channel_processing(subject: Subject, raw: mne.io.Raw, original_spikes: Dict[str, np.ndarray], channel_name: str):
    channel_raw = raw.copy().pick_channels([channel_name])
    channel_raw.load_data()
    filtered_channel_raw = channel_raw.copy().filter(l_freq=LOW_THRESHOLD_FREQUENCY, h_freq=HIGH_THRESHOLD_FREQUENCY,
                                                     n_jobs=2)
    filtered_channel_data = filtered_channel_raw.get_data()[0]
    filtered_channel_data = sp_stats.zscore(filtered_channel_data)
    channel_spikes = original_spikes[channel_name]
    channel_spikes = utils.get_real_spikes_indexs(filtered_channel_data, channel_spikes)

    if channel_name.endswith('1'):
        plotting.create_TFR_plot(subject, channel_raw, channel_spikes, channel_name)
        plotting.create_PSD_plot(subject, channel_raw, channel_spikes, channel_name)
        plotting.create_ERP_plot(subject, filtered_channel_raw, channel_spikes, channel_name)
        plotting.create_channel_features_histograms(subject, filtered_channel_data, channel_spikes, channel_name)

    return channel_spikes


def main(subject_name: str):
    subject = Subject(subject_name, False)

    raw = resample_and_filter_data(subject)

    spikes = spikes_detection.detect_spikes_of_subject(subject, raw)

    plotting.create_raster_plot(
        subject=subject,
        spikes=spikes,
        add_hypnogram=True,
        add_histogram=True
    )

    plotting.create_raster_plot(
        subject=subject,
        spikes=spikes,
        add_hypnogram=True,
        add_histogram=True,
        cut_hypnogram=False
    )

    plotting.create_raster_plot(
        subject=subject,
        spikes=spikes,
        add_hypnogram=False,
        add_histogram=True,
    )

    # running channel_processing funcion on every channel in parallel
    channels_spikes = Parallel(n_jobs=os.cpu_count(), backend='multiprocessing')(
        delayed(channel_processing)(subject, raw, dict(spikes), channel_name) for channel_name in raw.ch_names
    )
    channels_spikes = {channel_name: channel_spikes for channel_name, channel_spikes in zip(raw.ch_names, channels_spikes)}

    group_spikes(channels_spikes)


if __name__ == '__main__':
    start_time = time.time()
    main('p396')
    print(f'Time taken: {(time.time() - start_time) / 60} minutes')