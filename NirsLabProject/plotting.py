import os
import mne
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mne.time_frequency import tfr_morlet
import scipy.io as sio

from NirsLabProject import utils
from NirsLabProject.config.subject import Subject
from NirsLabProject.config.consts import *


COLOR_CODES = np.array(['#000000', '#332288', '#117733', '#CC6677', '#AA4499'])


# assigns color to each channel so no two consecutive channels will have the same color
def get_channel_names_and_color(channels: List[str]) -> Dict[str, str]:
    colors = {}
    channels = dict.fromkeys(c[:-1] for c in channels)
    for i, channel in enumerate(channels):
        colors[channel] = COLOR_CODES[i % len(COLOR_CODES)]
    return colors


# adds stimuli windows to the provided plot (ax)
def add_stimuli(subject: Subject, ax, number_of_channels: int, tmax: float, color: str):
    if not os.path.exists(subject.paths.subject_stimuli_path):
        return False

    stimuli_times = utils.get_stimuli_time_windows(subject)
    stimuli_times = np.asarray([int(timestamp) for tup in stimuli_times for timestamp in tup])

    edges = np.arange(tmax)
    edges = np.append(0, edges[stimuli_times])

    values = np.ones_like(stimuli_times) * number_of_channels
    values[::2] = 0  # 0 for even indexes, 1 for even indexes

    ax2 = ax.twiny()
    ax2.tick_params(
        top=False,
        labeltop=False,
        left=False,
        labelleft=False,
        right=False,
        labelright=False,
        bottom=False,
        labelbottom=False)
    ax.stairs(values, edges, fill=True, color=color)
    return True


def add_hypnogram_to_fig(subject: Subject, ax, number_of_channels: int, sleeping_stage: int, tmax: float, color: str = '#DCDCDC'):
    if os.path.exists(subject.paths.subject_hypnogram_path):
        hypno = np.loadtxt(subject.paths.subject_hypnogram_path)
        edges = np.arange(tmax/HYPNOGRAM_SAMPLES_INTERVAL_IN_SECONDS) * HYPNOGRAM_SAMPLES_INTERVAL_IN_SECONDS

        if sleeping_stage == WAKE:
            # make sure that Wake is displayed in color <color>
            hypno = pd.Series(hypno).map({0: 5, 1: 0, 2: 0, 3: 0, 4: 0}).values
        elif sleeping_stage == REM:
            # make sure that REM is displayed in color <color>
            hypno = pd.Series(hypno).map({0: 0, 1: 0, 2: 0, 3: 0, 4: 5}).values
        else:
            # make sure that REM is displayed after Wake
            hypno = pd.Series(hypno).map({0: 5, 1: 3, 2: 2, 3: 1, 4: 4}).values
    elif os.path.exists(subject.paths.subject_sleep_scoring_path):
        f = sio.loadmat(subject.paths.subject_sleep_scoring_path)
        data = f['sleep_score']
        hypno = np.array(data)[0]
        hypno = pd.Series(hypno).map({0: 5, -1: 5, 1: 0}).values
        edges = np.arange(tmax)
    else:
        raise Exception(f'No hypnogram or sleep score file found for subject {subject.subject_id}')

    # reduce data and bin edges to only moments of change in the hypnogram
    # (to avoid drawing thousands of tiny individual lines when sf is high)
    change_points = np.nonzero(np.ediff1d(hypno, to_end=1))
    values = hypno[change_points]
    values = number_of_channels / 5 * values.clip(0)

    edges = np.append(0, edges[change_points[0] // 1000])
    edges[-1] += HYPNOGRAM_SAMPLES_INTERVAL_IN_SECONDS

    ax2 = ax.twiny()
    ax2.tick_params(top=False, labeltop=False, left=False, labelleft=False, right=False, labelright=False, bottom=False,
                    labelbottom=False)
    ax.stairs(values, edges, fill=True, color=color)


def add_histogram_to_fig(ax, channels_data: List[np.ndarray]):
    # Histogram of every channel - Y ax
    y = []
    ax_histogram_y = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)  # sets the location of the subplot
    ax_histogram_y.tick_params(axis="y", labelleft=False)
    bins = np.arange(0, len(channels_data), 1)
    for i, c in enumerate(channels_data):
        y.extend(len(c) * [i])
    ax_histogram_y.hist(y, bins=bins, orientation='horizontal')

    # Histogram of events in time - X ax
    binwidth = 10  # how many seconds are in one bin
    ax_histogram_x = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)  # sets the location of the subplot
    ax_histogram_x.tick_params(axis="x", labelbottom=False)
    x = np.concatenate(channels_data)
    lim = (int(np.max(np.abs(x)) / binwidth) + 1) * binwidth
    bins = np.arange(0, lim + binwidth, binwidth)
    ax_histogram_x.hist(x, bins=bins)


def get_model_name(subject: Subject) -> str:
    return f"{'bipolar' if subject.bipolar_model else 'one channel'} model"


@utils.catch_exception
# based on https://pythontic.com/visualization/charts/spikerasterplot
def create_raster_plot(subject: Subject, tmin: float, tmax: float, spikes: Dict[str, np.ndarray], add_histogram: bool = True,
                       add_hypnogram: bool = True, show: bool = False):
    channels_name = list(spikes.keys())
    channels_data = list(spikes.values())

    # set plot size and locations
    fig = plt.figure(layout='constrained')
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    plt.title(f"{subject.name} raster - {get_model_name(subject)}")

    # assign a different color for every channel's group (same electrod2)
    channels_color = get_channel_names_and_color(channels_name)
    colors = [channels_color[channel[:-1]] for channel in channels_name]
    colors.reverse()

    # reverse the data so the deepest channel of every electrode will be above the rest
    channels_name.reverse()
    channels_data.reverse()

    # add hypnogram to the plot if needed
    add_histogram and add_histogram_to_fig(ax, channels_data)

    # set raster plot start and end time to be the same as the edf file
    channels_data[0][0] = tmin
    channels_data[0][-1] = tmax

    # add the main plot - the raster
    ax.eventplot(
        positions=channels_data,
        color=colors,
        linelengths=[0.3]*len(channels_data),
        linewidth=[0.8]*len(channels_data)
    )

    # sets margins to the plot
    ax.margins(x=0.005, y=0.005)

    # set y axis labels so only the first channel of every electrode will be labeled
    yticks = []
    for channel_name in channels_name:
        if channel_name[-1] == '1':
            yticks.append(channel_name[:-1])
        else:
            yticks.append('')

    # set y axis labels
    ax.set_yticklabels(yticks)
    ax.set_yticks(np.arange(len(channels_name)))
    plt.yticks(fontsize=8)

    # set x axis labels values to minutes
    xticks = ax.get_xticklabels()
    xticks_values = [int(tick.get_text()) for tick in xticks[1:]]

    # if the plot is more than 2 hours the labels will be hours and not minutes
    if max(xticks_values) > 120*60:
        # set x axis labels values to hours
        ax.set(
            xticklabels=[t // 3600 for t in range(0, max(xticks_values), 3600)],
            xticks=[t for t in range(0, max(xticks_values), 3600)]
        )
        plt.xlabel('Hours')
    else:
        # set x axis labels values to minutes
        [tick.set_text(int(tick.get_text()) // 60) for tick in xticks[1:]]
        ax.set_xticklabels(xticks)
        plt.xlabel('Minutes')

    # set plot proportions
    fig.set_figwidth(14)
    fig.set_figheight(7)

    legend = []
    if add_hypnogram:
        legend.append(mpatches.Patch(color=NREM_COLOR, label='NREM'))

        # for the detection project we have full hypnogram in subject.paths.subject_hypnogram_path
        # so REM and Wake windows are separated
        if os.path.exists(subject.paths.subject_hypnogram_path):
            # add wake time windows to the plot
            add_hypnogram_to_fig(subject, ax, len(channels_data), WAKE, tmax, WAKE_COLOR)
            legend.append(mpatches.Patch(color=WAKE_COLOR, label='Wake'))

            # add REM time windows to the plot
            add_hypnogram_to_fig(subject, ax, len(channels_data), REM, tmax, REM_COLOR)
            legend.append(mpatches.Patch(color=REM_COLOR, label='REM'))

        else:
            # for the stimulation project we have only NREM and not NREM windows
            add_hypnogram_to_fig(subject, ax, len(channels_data), REM, tmax, REM_COLOR)
            legend.append(mpatches.Patch(color=REM_COLOR, label='REM/Wake'))
        raster_path = subject.paths.subject_hypno_raster_plot_path

    else:
        raster_path = subject.paths.subject_raster_plot_path

    # add stimuli to the plot if needed
    if add_stimuli(subject, ax, len(channels_data), tmax, STIMULI_COLOR):
        legend.append(mpatches.Patch(color=STIMULI_COLOR, label='Stimuli'))

    # add legend to the plot if needed
    if legend:
        ax.legend(handles=legend, bbox_to_anchor=(1.15, 1.25))

    # save the plot to the subject's folder
    plt.savefig(raster_path, dpi=1000)

    # opens the plot if needed
    if show:
        plt.show()


@utils.catch_exception
def create_ERP_plot(subject: Subject, channel_raw: mne.io.Raw,
                    spikes: np.ndarray, channel_name: str, show: bool = False):
    """
    channel_data: is a (N,) np array with the MNE channel (it can be filtred, etc..)
    """
    epochs = utils.create_epochs(channel_raw, spikes, -1, 1)
    fig = epochs.plot_image(
        show=show,
        picks=[channel_name],
        vmin=-150,
        vmax=150,
        title=f'{subject.name} {channel_name} ERP\nn={len(spikes)} - {get_model_name(subject)}'
    )[0]
    fig.savefig(os.path.join(subject.paths.subject_erp_plots_dir_path, f'{subject.name}-{channel_name}.png'),  dpi=1000)


@utils.catch_exception
# channel_raw must not be filtred
def create_TFR_plot(subject: Subject, channel_raw: mne.io.Raw,
                    spikes_timestamps: np.ndarray, channel_name: str, show: bool = False):
    epochs = utils.create_epochs(channel_raw, spikes_timestamps, -1, 1)
    freqs = np.logspace(*np.log10([5, 250]), num=100)
    power, _ = tfr_morlet(
        inst=epochs,
        freqs=freqs,
        use_fft=True,
        return_itc=True,
        decim=3,
        n_jobs=1,
        n_cycles=freqs/2
    )
    power.plot(
        picks=[channel_name],
        show=show,
        mode='logratio',
        baseline=(-1, 1),
        title=f'{subject.name} {channel_name} TFR\nn={len(spikes_timestamps)} - {get_model_name(subject)}'
    )
    plt.savefig(os.path.join(subject.paths.subject_tfr_plots_dir_path, f'{subject.name}-{channel_name}.png'),  dpi=1000)


@utils.catch_exception
# channel_raw must not be filtred
def create_PSD_plot(subject: Subject, channel_raw: mne.io.Raw,
                    spikes_timestamps: np.ndarray, channel_name: str, show: bool = False):
    fig, ax = plt.subplots(2)
    fig.set_size_inches(5, 10)

    # plot_psd is obsolete, but the new function plot_psd_topomap is not supporting fig saving
    channel_raw.plot_psd(
        fmin=0,
        fmax=250,
        picks=[channel_name],
        ax=ax[0],
        show=show,
    )
    epochs = utils.create_epochs(channel_raw, spikes_timestamps, -1, 1)
    epochs.plot_psd(
        fmin=0,
        fmax=250,
        ax=ax[1],
        show=show,
    )
    ax[0].set_title(f'{subject.name} {channel_name} PSD - raw - {get_model_name(subject)}')
    ax[0].set_xlabel('Frequency (Hz)')
    ax[1].set_title(f'{subject.name} {channel_name} PSD - {len(spikes_timestamps)} events - {get_model_name(subject)}')
    ax[1].set_xlabel('Frequency (Hz)')
    fig.tight_layout()
    plt.savefig(os.path.join(subject.paths.subject_psd_plots_dir_path, f'{subject.name}-{channel_name}.png'),  dpi=1000)


@utils.catch_exception
def create_channel_features_histograms(subject: Subject, amplitudes: np.ndarray,
                                       lengths: np.ndarray, channel_name: str):

    fig, ax = plt.subplots(2)
    ax[0].hist(amplitudes, bins='auto')
    ax[0].set_title(f'{subject.name} {channel_name} - Amplitudes Histogram\nn={len(amplitudes)} - {get_model_name(subject)}')
    ax[0].set_xlabel('Zscore')
    ax[1].hist(lengths, bins='auto')
    ax[1].set_title(f'{subject.name} {channel_name} - Lengths Histogram\nn={len(amplitudes)} - {get_model_name(subject)}')
    ax[1].set_xlabel('Msec')
    fig.tight_layout()
    plt.savefig(os.path.join(subject.paths.subject_histogram_plots_dir_path, f'{subject.name}-{channel_name}.png'),  dpi=1000)
