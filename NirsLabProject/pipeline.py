import os

import mne
from typing import List
import scipy.stats as sp_stats


import numpy as np

from NirsLabProject import spikes_detection
from NirsLabProject import plotting
from NirsLabProject import utils
from NirsLabProject.config.subject import Subject
from NirsLabProject.config.consts import *


def resample_and_filter_data(subject: Subject, raw: mne.io.Raw):
    if not os.path.exists(subject.paths.subject_resampled_fif_path):
        print(f'Resampling data, it might take some time... (around {len(raw.ch_names) * 4 // 60} minutes)')
        raw.resample(SR, n_jobs=2)
        print(f'Filtering data, it might take some time... (around {len(raw.ch_names) * 4 / 60} minutes)')
        raw.filter(l_freq=LOW_THRESHOLD_FREQUENCY, h_freq=HIGH_THRESHOLD_FREQUENCY, n_jobs=2)
        raw.save(subject.paths.subject_resampled_fif_path)
    return mne.io.read_raw_fif(subject.paths.subject_resampled_fif_path)


def main():
    subject = Subject('p396')

    raw = mne.io.read_raw_edf(subject.paths.subject_raw_edf_path)
    utils.pick_seeg_channels(raw)
    utils.clean_channels_name(raw)
    raw = resample_and_filter_data(subject, raw)

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
        add_hypnogram=False,
        add_histogram=True
    )

    for i, channel_name in enumerate(raw.ch_names):
        if channel_name.endswith('1'):
            channel_raw = raw.copy().pick_channels([channel_name])
            channel_data = channel_raw.get_data()
            channel_data = np.apply_along_axis(sp_stats.zscore, 1, channel_data)[0]
            channel_spikes = spikes[channel_name]
            plotting.create_ERP_plot(subject, channel_raw, channel_data, channel_spikes, channel_name)
            plotting.create_TFR_plot(subject, channel_raw, channel_data, channel_spikes, channel_name,)
            plotting.create_PSD_plot(subject, channel_raw, channel_name)


if __name__ == '__main__':
    main()