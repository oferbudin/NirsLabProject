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
from NirsLabProject.config.consts import *
from NirsLabProject.config.paths import Paths
from NirsLabProject.config.subject import Subject

import mne
from mne_features.feature_extraction import extract_features
from mne_features.univariate import get_univariate_funcs, compute_pow_freq_bands


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

    def format_raw(self, raw: mne.io.Raw, subj: Subject) -> np.ndarray:
        # must override in child class
        raise NotImplementedError

    def predict(self, raw: mne.io.Raw, subject: Subject) -> np.ndarray:
        # format the raw data
        start_time = time.time()
        features = self.format_raw(raw, subject)
        print(f'Feature extraciotn time: {time.time() - start_time}')
        features = np.nan_to_num(features[self.feature_names])

        # predict using the model
        # predictions = self.model.predict(features)
        predictions = self.model.predict_proba(features)
        y = (predictions[:, 1] >= 0.8).astype(int)

        spikes_onsets = np.where(y == 1)[0] / DIVISION_FACTOR
        return spikes_onsets


# Bipolar model class for bipolar montage data - subtracting two consecutive channels
class BipolarModel(Model):
    # model_lgbm = joblib.load(os.path.join(Paths.models_dir_path, 'LGBM_V1.pkl'))
    # model_rf = joblib.load(os.path.join(Paths.models_dir_path, 'RF_V1.pkl'))

    def __init__(self, model_name: str = ''):
        super(BipolarModel, self).__init__()
        self.model_name = model_name
        self.model = joblib.load(os.path.join(Paths.models_dir_path, model_name))

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

    def format_raw(self, raw: mne.io.Raw, subj: Subject) -> np.ndarray:
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
    def __init__(self, model_name: str = ''):
        super(UniChannelModel, self).__init__()
        self.model_name = model_name
        model, feature_names = joblib.load(os.path.join(Paths.models_dir_path, model_name)).values()
        self.model = model
        self.feature_names = feature_names

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

    def format_raw(self, raw: mne.io.Raw, subj: Subject) -> np.ndarray:
        chan = raw.ch_names[0]
        chan_part = f'{chan}_{raw.first_samp}_{raw.last_samp}'
        if not os.path.exists(subj.paths.subject_channels_spikes_features_intracranial_model(chan_part)):
            window_size = int(SR / DIVISION_FACTOR)  # 250 ms
            x = pd.DataFrame()

            epochs = []
            chan_raw = raw.copy().pick([chan]).get_data().flatten()
            # normalize chan
            chan_norm = (chan_raw - chan_raw.mean()) / chan_raw.std()
            # run on all 250ms epochs
            for i in range(0, len(chan_norm) - window_size, window_size):
                epochs.append(chan_norm[i: i + window_size])

            curr_feat = extract_epochs_top_features(epochs, subj.p_number, raw.info['sfreq'])
            chan_feat = {
                'chan_name': chan,
                'chan_ptp': np.ptp(chan_norm),
                'chan_skew': sp_stats.skew(chan_norm),
                'chan_kurt': sp_stats.kurtosis(chan_norm),
            }

            for feat in chan_feat.keys():
                curr_feat[feat] = chan_feat[feat]

            # save the epochs as column for debugging
            curr_feat['epoch'] = epochs
            x = pd.concat([x, curr_feat], axis=0)

            x.to_pickle(subj.paths.subject_channels_spikes_features_intracranial_model(chan_part))
        else:
            x = pd.read_pickle(subj.paths.subject_channels_spikes_features_intracranial_model(chan_part))

        return x

    def get_channels(self, channels: List[str]) -> List[List[str]]:
        return [[channel] for channel in channels]


def save_detection_to_npz_file(detections: Dict[str, np.ndarray], subject: Subject):
    print(f"Saving detections of subject {subject.name}")
    np.savez(subject.paths.subject_spikes_path, **detections)


def remove_stimuli_segments(raw, subject: Subject):
    total_time = raw.tmax - raw.tmin
    raw_segs = []
    last_end = 0.0  # Initialize the end time of the last segment

    stimuli_time_windows = utils.get_stimuli_time_windows(subject)
    for seg_start, seg_end in stimuli_time_windows:
        raw_segs.append(
            raw.copy().crop(
                tmin=last_end,
                tmax=seg_start,
                include_tmax=False,
            )
        )
        last_end = seg_end

    # Append the remaining part after the last rejected segment
    raw_segs.append(
        raw.copy().crop(
            tmin=last_end,
            tmax=None,  # Crop till the end of the recording
            include_tmax=False,
        )
    )

    stimuli_time_windows_time = sum([seg_end - seg_start for seg_start, seg_end in stimuli_time_windows])
    print(f'Raw data time: {total_time} | Stimuli time windows: {stimuli_time_windows_time} was removed from the raw data.')
    new_raw = mne.concatenate_raws(raw_segs)
    print(f'New raw data time: {new_raw.tmax - new_raw.tmin}')
    return new_raw


def handle_stimuli(model, raw: mne.io.Raw, subject: Subject) -> np.ndarray:
    channels_raw = remove_stimuli_segments(raw.copy(), subject)
    return model.predict(channels_raw, subject)


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

    if subject.bipolar_model:
        model = BipolarModel(subject.model_name)
    else:
        model = UniChannelModel(subject.model_name)

    spikes = {}
    for _, raw in electrodes_raw.items():
        all_channels = model.get_channels(raw.ch_names)

        # run on each channel and detect the spikes between stims
        channels_spikes = Parallel(n_jobs=1, backend='multiprocessing')(
            delayed(detect_spikes_of_subject_for_specific_channels)(subject, raw, channels, model) for channels in all_channels
        )

        for d in channels_spikes:
            spikes.update(d)
    save_detection_to_npz_file(spikes, subject)
    return spikes


def extract_epochs_features_mne(epochs, subj, sr):
    feat = {
        'subj': np.full(len(epochs), subj),
        'epoch_id': np.arange(len(epochs)),
    }

    selected_funcs = get_univariate_funcs(sr)
    selected_funcs.pop('spect_edge_freq', None)
    bands_dict = {'theta': (4, 8), 'alpha': (8, 12), 'sigma': (12, 16), 'beta': (16, 30), 'gamma': (30, 100), 'fast': (100, 300)}
    params = {'pow_freq_bands__freq_bands': bands_dict, 'pow_freq_bands__ratios': 'all', 'pow_freq_bands__psd_method': 'multitaper',
              'energy_freq_bands__freq_bands': bands_dict}
    X_new = extract_features(np.array(epochs)[:, np.newaxis, :], sr, selected_funcs, funcs_params=params, return_as_df=True)
    X_new['abspow'] = compute_pow_freq_bands(sr, np.array(epochs), {'total': (0.1, 500)}, False, psd_method='multitaper')
    # rename columns
    names = []
    for name in X_new.columns:
        if type(name) is tuple:
            if name[1] == 'ch0':
                names.append(name[0])
            else:
                names.append(name[0] + '_' + name[1].replace('ch0_', ''))
        else:
            names.append(name)

    X_new.columns = names

    # add ratios between bands
    X_new['energy_freq_bands_ab'] = X_new['energy_freq_bands_alpha'] / X_new['energy_freq_bands_beta']
    X_new['energy_freq_bands_ag'] = X_new['energy_freq_bands_alpha'] / X_new['energy_freq_bands_gamma']
    X_new['energy_freq_bands_as'] = X_new['energy_freq_bands_alpha'] / X_new['energy_freq_bands_sigma']
    X_new['energy_freq_bands_af'] = X_new['energy_freq_bands_alpha'] / X_new['energy_freq_bands_fast']
    X_new['energy_freq_bands_at'] = X_new['energy_freq_bands_alpha'] / X_new['energy_freq_bands_theta']
    X_new['energy_freq_bands_bt'] = X_new['energy_freq_bands_beta'] / X_new['energy_freq_bands_theta']
    X_new['energy_freq_bands_bs'] = X_new['energy_freq_bands_beta'] / X_new['energy_freq_bands_sigma']
    X_new['energy_freq_bands_bg'] = X_new['energy_freq_bands_beta'] / X_new['energy_freq_bands_gamma']
    X_new['energy_freq_bands_bf'] = X_new['energy_freq_bands_beta'] / X_new['energy_freq_bands_fast']
    X_new['energy_freq_bands_st'] = X_new['energy_freq_bands_sigma'] / X_new['energy_freq_bands_theta']
    X_new['energy_freq_bands_sg'] = X_new['energy_freq_bands_sigma'] / X_new['energy_freq_bands_gamma']
    X_new['energy_freq_bands_sf'] = X_new['energy_freq_bands_sigma'] / X_new['energy_freq_bands_fast']
    X_new['energy_freq_bands_gt'] = X_new['energy_freq_bands_gamma'] / X_new['energy_freq_bands_theta']
    X_new['energy_freq_bands_gf'] = X_new['energy_freq_bands_gamma'] / X_new['energy_freq_bands_fast']
    X_new['energy_freq_bands_ft'] = X_new['energy_freq_bands_fast'] / X_new['energy_freq_bands_theta']

    # Convert to dataframe
    feat = pd.DataFrame(feat)
    feat = pd.concat([feat, X_new], axis=1)

    return feat


def extract_epochs_top_features(epochs, subj, sr):
    mobility, complexity = ant.hjorth_params(epochs, axis=1)
    feat = {
        'subj': np.full(len(epochs), subj),
        'epoch_id': np.arange(len(epochs)),
        'kurtosis': sp_stats.kurtosis(epochs, axis=1),
        'hjorth_mobility': mobility,
        'hjorth_complexity': complexity,
        'ptp_amp': np.ptp(epochs, axis=1),
        'samp_entropy': np.apply_along_axis(ant.sample_entropy, axis=1, arr=epochs)
    }

    selected_funcs = ['teager_kaiser_energy']
    X_new = extract_features(np.array(epochs)[:, np.newaxis, :], sr, selected_funcs,
                             return_as_df=True)
    # rename columns
    names = []
    for name in X_new.columns:
        if type(name) is tuple:
            if name[1] == 'ch0':
                names.append(name[0])
            else:
                names.append(name[0] + '_' + name[1].replace('ch0_', ''))
        else:
            names.append(name)

    X_new.columns = names

    # Convert to dataframe
    feat = pd.DataFrame(feat)
    feat = pd.concat([feat, X_new], axis=1)

    return feat