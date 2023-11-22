import os
import mne
import joblib
import numpy as np
import pandas as pd
import antropy as ant
import scipy.signal as sp_sig
import scipy.stats as sp_stats
from scipy.integrate import simps

from NirsLabProject.config.consts import *
from NirsLabProject.config.paths import Paths
from NirsLabProject.config.subject import Subject


def format_raw_night(channel, raw: mne.io.Raw, norm='raw'):
    epochs = []
    window_size = int(SR / DIVISION_FACTOR)
    raw_channel = raw.copy().pick_channels([channel])
    if '-' in channel and 'REF' not in channel:
        chans = channel.split('-')
        mne.set_bipolar_reference(raw_channel, chans[0], chans[1], ch_name=channel)
        raw_data = raw_channel.get_data()[0]
    else:
        raw_data = raw_channel.get_data()[0]

    if norm == 'raw':
        raw_data = (raw_data - raw_data.mean()) / raw_data.std()
    for i in range(0, len(raw_data), window_size):
        curr_block = raw_data[i: i + window_size]
        if i + window_size < len(raw_data):
            epochs.append(curr_block)

    # Normalization
    epochs = np.array(epochs)
    if norm == 'epochs':
        epochs = (epochs - epochs.mean()) / epochs.std()
    return epochs


def channel_feat(channel: str, raw: mne.io.Raw):
    raw_channel = raw.copy().pick_channels([channel])
    raw_data = raw_channel.get_data()[0]
    feat = {
        'median': np.median(raw_data),
        'ptp': np.ptp(raw_data),
    }
    return feat


# yasa function for power features in each band
def bandpower_from_psd_ndarray(psd, freqs, bands, relative=True):
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


def calc_features_before_split(epochs, subject: Subject):
    # Bandpass filter
    freq_broad = (0.1, 500)
    # FFT & bandpower parameters
    bands = [
        (0.1, 4, 'delta'), (4, 8, 'theta'),
        (8, 12, 'alpha'), (12, 16, 'sigma'), (16, 30, 'beta'),
        (30, 100, 'gamma'), (100, 300, 'fast')
    ]

    # Calculate standard descriptive statistics
    hmob, hcomp = ant.hjorth_params(epochs, axis=1)

    feat = {
        # 'subj': np.full(len(epochs), subject.name),
        'epoch_id': np.arange(len(epochs)),
        'std': np.std(epochs, ddof=1, axis=1),
        'iqr': sp_stats.iqr(epochs, axis=1),
        'skew': sp_stats.skew(epochs, axis=1),
        'kurt': sp_stats.kurtosis(epochs, axis=1),
        'nzc': ant.num_zerocross(epochs, axis=1),
        'hmob': hmob,
        'hcomp': hcomp
    }

    # Calculate spectral power features (for EEG + EOG)
    freqs, psd = sp_sig.welch(epochs, SR)
    bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands)
    for j, (_, _, b) in enumerate(bands):
        feat[b] = bp[j]

    # Add power ratios for EEG
    feat['at'] = feat['alpha'] / feat['theta']
    feat['ag'] = feat['gamma'] / feat['alpha']
    feat['sf'] = feat['sigma'] / feat['fast']
    feat['bf'] = feat['beta'] / feat['fast']
    feat['gf'] = feat['gamma'] / feat['fast']
    # need those?
    feat['gt'] = feat['gamma'] / feat['theta']
    feat['ft'] = feat['fast'] / feat['theta']
    feat['af'] = feat['fast'] / feat['alpha']

    # Add total power
    idx_broad = np.logical_and(
        freqs >= freq_broad[0], freqs <= freq_broad[1])
    dx = freqs[1] - freqs[0]
    feat['abspow'] = np.trapz(psd[:, idx_broad], dx=dx)

    # Calculate entropy and fractal dimension features
    feat['perm'] = np.apply_along_axis(
        ant.perm_entropy, axis=1, arr=epochs, normalize=True)
    feat['higuchi'] = np.apply_along_axis(
        ant.higuchi_fd, axis=1, arr=epochs)
    feat['petrosian'] = ant.petrosian_fd(epochs, axis=1)

    # Convert to dataframe
    feat = pd.DataFrame(feat)

    return feat


def get_all_feat_eog_with_chan_feat(eog_num: str, subject: Subject, raw: mne.io.Raw):
    feat_all = pd.DataFrame()
    channel_name = 'EOG' + eog_num
    x = format_raw_night(channel_name, raw)
    chan_feat = channel_feat(channel_name, raw)
    features = calc_features_before_split(x, subject)
    for feat in chan_feat.keys():
        features[feat] = chan_feat[feat]
    feat_all = pd.concat([feat_all, features], axis=0)

    return feat_all


def detect_spikes_of_subject(subject: Subject, raw: mne.io.Raw) -> np.ndarray:
    if 'EOG1' not in raw.ch_names or 'EOG2' not in raw.ch_names:
        return np.array([])
        raise Exception('EOG1 or EOG2 not in raw channels')
    xgb_model = joblib.load(os.path.join(Paths.models_dir_path, SCALP_MODEL_NAME))
    raw.load_data()
    raw = raw.copy()
    raw = raw.notch_filter(50 if subject.sourasky_project else 60)
    # raw = raw.filter(0.1, 40)
    feat_eog1 = get_all_feat_eog_with_chan_feat('1', subject, raw)
    feat_eog2 = get_all_feat_eog_with_chan_feat('2', subject, raw)
    all_feat = pd.concat([feat_eog1, feat_eog2.iloc[:, 2:].add_suffix('_2')], axis=1)
    res = xgb_model.predict_proba(all_feat)
    return np.where(res[:, 1] > SCALP_MODEL_PROBABILITIES)[0]*0.25
