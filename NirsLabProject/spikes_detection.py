import mne
import pandas as pd
import numpy as np
import scipy.stats as sp_stats
import scipy.signal as sp_sig
from scipy.integrate import simps
import antropy as ant
from sklearn.preprocessing import robust_scale
import joblib

from NirsLabProject.consts import *

model_lgbm = joblib.load('models/LGBM_V1.pkl')
model_rf = joblib.load('models/RF_V1.pkl')


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


def calc_features(epochs, subject):
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
        'subj': np.full(len(epochs), subject),
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
    # feat.index.name = 'epoch'

    ############################
    # SMOOTHING & NORMALIZATION
    ############################
    roll1 = feat.rolling(window=1, center=True, min_periods=1, win_type='triang').mean()
    roll1[roll1.columns] = robust_scale(roll1, quantile_range=(5, 95))
    roll1 = roll1.iloc[:, 1:].add_suffix('_cmin_norm')

    roll3 = feat.rolling(window=3, center=True, min_periods=1, win_type='triang').mean()
    roll3[roll3.columns] = robust_scale(roll3, quantile_range=(5, 95))
    roll3 = roll3.iloc[:, 1:].add_suffix('_pmin_norm')

    # Add to current set of features
    feat = feat.join(roll1).join(roll3)

    return feat


def format_raw(raw):
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


def detect_spikes(raw, subject, plot=True):
    x = format_raw(raw)
    features = calc_features(x, subject)
    features = np.nan_to_num(features[model_lgbm.feature_name_])
    y_lgbm = model_lgbm.predict(features)
    y_rf = model_rf.predict(features)
    y = np.array(y_lgbm) + np.array(y_rf)
    y[y == 2] = 1
    spikes_onsets = np.where(y == 1)[0] / DIVISION_FACTOR
    if plot:
        raw.set_annotations(mne.Annotations(spikes_onsets, [0.25] * len(spikes_onsets), ['spike'] * len(spikes_onsets)))
        mne.set_bipolar_reference(raw, raw.ch_names[0], raw.ch_names[1], ch_name='bi', drop_refs=False).plot(
            duration=30, scalings='auto')
    return spikes_onsets


def save_detection_to_npz_file(detections, subject):
    print(f"Saving detections for subject {subject}")
    np.savez(subject_spikes_path, **detections)


def clean_channel_name(channel):
    return channel.replace('-REF1', '').replace('SEEG ', '')


def create_bipolar_channels(channels):
    bi_channels = []

    # get the channels for bipolar reference
    for i, chan in enumerate(channels):
        if i + 1 < len(channels):
            next_chan = channels[i + 1]
            # check that its the same contact
            if clean_channel_name(next_chan)[:-1] == clean_channel_name(chan)[:-1]:
                bi_channels.append([chan, next_chan])

    return bi_channels


def get_index_of_last_seeg_channel(channels):
    SEEG_PREFIX = 'SEEG'
    for i in range(len(channels)-1, 0, -1):
        if channels[i].startswith(SEEG_PREFIX):
            return i+1
    return len(channels)

if __name__ == '__main__':
    raw_data = mne.io.read_raw_edf(subject_raw_edf_path)
    all_channels = raw_data.ch_names
    last_channel_index = get_index_of_last_seeg_channel(all_channels)
    seeg_channels = all_channels[:last_channel_index]
    bipolar_channels = create_bipolar_channels(seeg_channels)

    spikes = {}
    import time
    # run on each channel and detect the spikes between stims
    for channels in bipolar_channels:
        s = time.time()
        print(f'Detecting spikes for channels {channels}')
        raw_copy = raw_data.copy()
        raw = raw_copy.pick_channels(channels)
        raw.crop(tmax=60*60*2)
        raw.resample(SR)
        channel_spikes = detect_spikes(raw, SUBJECT, False)
        spikes[clean_channel_name(channels[0])] = channel_spikes
        print(f'Finished detecting spikes for channels {channels}')
        print(f'Took {time.time() - s}')

    save_detection_to_npz_file(spikes, SUBJECT)
