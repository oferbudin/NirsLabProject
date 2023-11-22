import os
import mne
import time
import joblib
import numpy as np
import pandas as pd
import antropy as ant
import scipy.signal as sp_sig
import scipy.stats as sp_stats
from scipy.integrate import simps
from typing import List, Dict, Tuple
from joblib import Parallel, delayed
from sklearn.preprocessing import robust_scale

from NirsLabProject.utils import general_utils as utils
from NirsLabProject.utils import sleeping_utils
from NirsLabProject.config.consts import *
from NirsLabProject.config.paths import Paths
from NirsLabProject.config.subject import Subject


mne.set_config('MNE_BROWSER_BACKEND', 'qt')


class Model:
    # Bandpass filter
    freq_broad = (0.1, 500)

    # FFT & bandpower parameters
    bands = [
        (0.1, 4, 'delta'), (4, 8, 'theta'),
        (8, 12, 'alpha'), (12, 16, 'sigma'), (16, 30, 'beta'),
        (30, 100, 'gamma'), (100, 300, 'fast')
    ]

    def calculate_standart_descriptive_statistics (self, epochs: np.ndarray):
        # Calculate standard descriptive statistics
        hmob, hcomp = ant.hjorth_params(epochs, axis=1)

        feat = {
            'epoch_id': np.arange(len(epochs)),
            'std': np.std(epochs, ddof=1, axis=1),
            'iqr': sp_stats.iqr(epochs, axis=1),
            'skew': sp_stats.skew(epochs, axis=1),
            'kurt': sp_stats.kurtosis(epochs, axis=1),
            'nzc': ant.num_zerocross(epochs, axis=1),
            'hmob': hmob,
            'hcomp': hcomp
        }

        return feat

    def calculate_spectral_power_features(self, epochs: np.ndarray, feat: dict) -> Tuple[np.ndarray, np.ndarray]:
        # Calculate spectral power features (for EEG + EOG)
        freqs, psd = sp_sig.welch(epochs, SR)
        bp = self.bandpower_from_psd_ndarray(psd, freqs, bands=self.bands)
        for j, (_, _, b) in enumerate(self.bands):
            feat[b] = bp[j]
        return freqs, psd

    def add_total_power(self, feat: dict, freqs: np.ndarray, psd: np.ndarray):
        # Add total power
        idx_broad = np.logical_and(
            freqs >= self.freq_broad[0], freqs <= self.freq_broad[1])
        dx = freqs[1] - freqs[0]
        feat['abspow'] = np.trapz(psd[:, idx_broad], dx=dx)

    def calculate_entropy_and_fractal_dimension_features(self, epochs: np.ndarray, feat: dict):
        # Calculate entropy and fractal dimension features
        feat['perm'] = np.apply_along_axis(
            ant.perm_entropy, axis=1, arr=epochs, normalize=True)
        feat['higuchi'] = np.apply_along_axis(
            ant.higuchi_fd, axis=1, arr=epochs)
        feat['petrosian'] = ant.petrosian_fd(epochs, axis=1)

    def smoothing_and_normalization(self, feat: pd.DataFrame) -> pd.DataFrame:
        return feat

    def add_power_ratio(self, feat: np.ndarray):
        pass

    def bandpower_from_psd_ndarray(self, psd, freqs, bands, relative=True):
        # Type checks
        assert isinstance(bands, list), 'bands must be a list of tuple(s)'
        assert isinstance(relative, bool), 'relative must be a boolean'

        # Safety checks
        freqs = np.asarray(freqs)
        psd = np.asarray(psd)
        assert freqs.ndim == 1, 'freqs must be a 1-D array of shape (n_freqs,)'
        assert psd.shape[-1] == freqs.shape[-1], 'n_freqs must be last axis of psd'

        # Extract frequencies of interest
        all_freqs = np.hstack([[b[0], b[1]] for b in bands])
        fmin, fmax = min(all_freqs), max(all_freqs)
        idx_good_freq = np.logical_and(freqs >= fmin, freqs <= fmax)
        freqs = freqs[idx_good_freq]
        res = freqs[1] - freqs[0]

        # Trim PSD to frequencies of interest
        psd = psd[..., idx_good_freq]

        # Check if there are negative values in PSD
        if (psd < 0).any():
            msg = (
                "There are negative values in PSD. This will result in incorrect "
                "bandpower values. We highly recommend working with an "
                "all-positive PSD. For more details, please refer to: "
                "https://github.com/raphaelvallat/yasa/issues/29")
            print(msg)

        # Calculate total power
        total_power = simps(psd, dx=res, axis=-1)
        total_power = total_power[np.newaxis, ...]

        # Initialize empty array
        bp = np.zeros((len(bands), *psd.shape[:-1]), dtype=np.float)

        # Enumerate over the frequency bands
        labels = []
        for i, band in enumerate(bands):
            b0, b1, la = band
            labels.append(la)
            idx_band = np.logical_and(freqs >= b0, freqs <= b1)
            bp[i] = simps(psd[..., idx_band], dx=res, axis=-1)

        if relative:
            bp /= total_power
        return bp

    def calc_features(self, epochs: np.ndarray, subject_name: str):
        feat = self.calculate_standart_descriptive_statistics(epochs)
        freqs, psd = self.calculate_spectral_power_features(epochs, feat)
        self.add_power_ratio(feat)

        self.add_total_power(feat, freqs, psd)
        self.calculate_entropy_and_fractal_dimension_features(epochs, feat)

        # Convert to dataframe
        feat = pd.DataFrame(feat)
        feat = self.smoothing_and_normalization(feat)
        return feat

    def channel_feat(self, features: pd.DataFrame, raw: mne.io.Raw, channel: str) :
        pass

    def format_raw(self, raw: mne.io.Raw) -> np.ndarray:
        # must override in child class
        raise NotImplementedError

    def predict(self, raw: mne.io.Raw, subject: Subject) -> np.ndarray:
        # format the raw data
        x = self.format_raw(raw)

        # calculate the features
        features = self.calc_features(x, subject.name)
        self.channel_feat(features, raw, raw.ch_names[0])

        # check for nans
        features = np.nan_to_num(features[self.model_lgbm.feature_name_])

        # predict using the models
        y_lgbm = self.model_lgbm.predict(features)
        y_rf = self.model_rf.predict(features)

        # combine the predictions
        y = np.array(y_lgbm) + np.array(y_rf)
        y[y == 2] = 1

        spikes_onsets = np.where(y == 1)[0] / DIVISION_FACTOR
        return spikes_onsets


# Bipolar model class for bipolar montage data - subtracting two consecutive channels
class BipolarModel(Model):
    model_lgbm = joblib.load(os.path.join(Paths.models_dir_path, 'LGBM_V1.pkl'))
    model_rf = joblib.load(os.path.join(Paths.models_dir_path, 'RF_V1.pkl'))

    def __init__(self):
        super(BipolarModel, self).__init__()

    def smoothing_and_normalization(self, feat: pd.DataFrame) -> pd.DataFrame:
        roll1 = feat.rolling(window=1, center=True, min_periods=1, win_type='triang').mean()
        roll1[roll1.columns] = robust_scale(roll1, quantile_range=(5, 95))
        roll1 = roll1.iloc[:, 1:].add_suffix('_cmin_norm')

        roll3 = feat.rolling(window=3, center=True, min_periods=1, win_type='triang').mean()
        roll3[roll3.columns] = robust_scale(roll3, quantile_range=(5, 95))
        roll3 = roll3.iloc[:, 1:].add_suffix('_pmin_norm')

        # Add to current set of features
        return feat.join(roll1).join(roll3)

    def add_power_ratio(self, feat: dict):
        # Add power ratios for EEG
        delta = feat['delta']
        feat['dt'] = delta / feat['theta']
        feat['ds'] = delta / feat['sigma']
        feat['db'] = delta / feat['beta']
        feat['dg'] = delta / feat['gamma']
        feat['df'] = delta / feat['fast']
        feat['at'] = feat['alpha'] / feat['theta']
        feat['gt'] = feat['gamma'] / feat['theta']
        feat['ft'] = feat['fast'] / feat['theta']
        feat['ag'] = feat['gamma'] / feat['alpha']
        feat['af'] = feat['fast'] / feat['alpha']

    def format_raw(self, raw: mne.io.Raw) -> np.ndarray:
        epochs = []
        window_size = int(SR / DIVISION_FACTOR)
        raw.load_data()

        # Create bipolar channel
        raw_bi = mne.set_bipolar_reference(raw, raw.ch_names[0], raw.ch_names[1], ch_name='bi')
        raw_data = raw_bi.get_data()[0]

        # Normalization
        raw_data = sp_stats.zscore(raw_data)
        for i in range(0, len(raw_data), window_size):
            curr_block = raw_data[i: i + window_size]
            if i + window_size < len(raw_data):
                epochs.append(curr_block)
        return np.array(epochs)

    def get_channels(self, channels: List[str]) -> List[List[str]]:
        bi_channels = []

        # get the channels for bipolar reference
        for i, chan in enumerate(channels):
            if i + 1 < len(channels):
                next_chan = channels[i + 1]
                # check that its the same contact
                ch1, _ = utils.extract_channel_name_and_contact_number(next_chan)
                ch2, _ = utils.extract_channel_name_and_contact_number(chan)
                if ch1 == ch2:
                    bi_channels.append([chan, next_chan])

        return bi_channels


# One channel model class that uses the raw data of one channel for prediction
class UniChannelModel(Model):
    model_lgbm = joblib.load(os.path.join(Paths.models_dir_path, 'LGBM_V2.pkl'))
    model_rf = joblib.load(os.path.join(Paths.models_dir_path, 'RF_V2.pkl'))

    def add_power_ratio(self, feat: np.ndarray):
        # Add power ratios for EEG
        feat['at'] = feat['alpha'] / feat['theta']
        feat['gt'] = feat['gamma'] / feat['theta']
        feat['ft'] = feat['fast'] / feat['theta']
        feat['ag'] = feat['gamma'] / feat['alpha']
        feat['af'] = feat['fast'] / feat['alpha']
        feat['sf'] = feat['sigma'] / feat['fast']
        feat['bf'] = feat['beta'] / feat['fast']
        feat['gf'] = feat['gamma'] / feat['fast']

    def channel_feat(self, features: pd.DataFrame, raw: mne.io.Raw, channel: str):
        raw_data = raw.pick_channels([channel]).resample(SR).get_data()[0]
        ch_feat = {
            'median': np.median(raw_data),
            'ptp': np.ptp(raw_data),
        }
        for feat in ch_feat.keys():
            features[feat] = ch_feat[feat]

    def format_raw(self, raw: mne.io.Raw) -> np.ndarray:
        epochs = []
        window_size = int(SR / DIVISION_FACTOR)
        raw.load_data()
        raw_data = raw.get_data()[0]

        # Normalization
        raw_data = sp_stats.zscore(raw_data)
        for i in range(0, len(raw_data), window_size):
            curr_block = raw_data[i: i + window_size]
            if i + window_size < len(raw_data):
                epochs.append(curr_block)

        return np.array(epochs)

    def get_channels(self, channels: List[str]) -> List[List[str]]:
        return [[channel] for channel in channels]


def save_detection_to_npz_file(detections: Dict[str, np.ndarray], subject: Subject):
    print(f"Saving detections of subject {subject.name}")
    np.savez(subject.paths.subject_spikes_path, **detections)


def handle_stimuli(model, raw: mne.io.Raw, subject: Subject) -> np.ndarray:
    spikes = np.asarray([])

    stim_sections_sec = utils.get_stimuli_time_windows(subject)
    stim_start_sec = stim_sections_sec[0][0]

    # get all raw data until the first stim
    baseline_raw = raw.copy().crop(tmin=0, tmax=stim_start_sec)
    spikes = np.concatenate((spikes, model.predict(baseline_raw, subject)), axis=0)

    # fill sections of stim and the stops between
    for i, (start, end) in enumerate(stim_sections_sec):
        # fill the current stim
        raw_without_stim = utils.remove_stimuli_from_raw(subject, raw.copy().crop(tmin=start, tmax=end), start, end)
        new_spikes = start + model.predict(raw_without_stim, subject)
        spikes = np.concatenate((spikes, new_spikes), axis=0)
        if i + 1 < len(stim_sections_sec):
            # the stop is the time between the end of the curr section and the start of the next, buffer of 0.5 sec of the stim
            next_data = raw.copy().crop(tmin=end + 0.5, tmax=stim_sections_sec[i + 1][0] - 0.5)
            new_spikes = end + 0.5 + model.predict(next_data, subject)
            spikes = np.concatenate((spikes, new_spikes), axis=0)
        else:
            data_to_the_end = raw.copy().crop(tmin=end + 0.5, tmax=raw.tmax)
            new_spikes = end + 0.5 + model.predict(data_to_the_end, subject)
            spikes = np.concatenate((spikes, new_spikes), axis=0)
    return spikes


def detect_spikes_of_subject_for_specific_channels(subject: Subject, raw: mne.io.Raw, channels: list, model) -> Dict[str, np.ndarray]:
    s = time.time()
    print(f'Detecting spikes for channels {channels}')
    channels_raw = raw.copy().pick_channels(channels)
    print(f'Loading data for channels {channels}')
    channels_raw.load_data()
    if os.path.exists(subject.paths.subject_stimuli_path):
        channel_spikes = handle_stimuli(model, channels_raw, subject)
    else:
        channel_spikes = model.predict(channels_raw, subject)
    print(f'Finished detecting spikes for channels {channels} | Took {time.time() - s}')
    return {channels[0]: channel_spikes}


def detect_spikes_of_subject(subject: Subject, electrodes_raw: dict[str, mne.io.Raw], sleep_cycle_data: bool = False) -> dict:
    print(f'Detecting Intracranial spikes for subject {subject.name}')
    if not FORCE_DETECT_SPIKES and os.path.exists(subject.paths.subject_spikes_path):
        print(f'Spikes already detected for subject {subject.name}')
        return np.load(subject.paths.subject_spikes_path, allow_pickle=True)

    # if sleep_cycle_data:
    #     start_timestamp, end_timestamp = sleeping_utils.get_timestamps_in_seconds_of_first_rem_sleep(subject)
    #     raw.crop(tmin=start_timestamp, tmax=end_timestamp)

    if subject.bipolar_model:
        model = BipolarModel()
    else:
        model = UniChannelModel()

    spikes = {}
    for _, raw in electrodes_raw.items():
        all_channels = model.get_channels(raw.ch_names)

        # run on each channel and detect the spikes between stims
        channels_spikes = Parallel(n_jobs=min(2, os.cpu_count()//2), backend='multiprocessing')(
            delayed(detect_spikes_of_subject_for_specific_channels)(subject, raw, channels, model) for channels in all_channels
        )

        for d in channels_spikes:
            spikes.update(d)
    save_detection_to_npz_file(spikes, subject)
    return spikes
