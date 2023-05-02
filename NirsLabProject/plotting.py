import os
import mne
import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet

from NirsLabProject import utils
from NirsLabProject import sleeping_utils
from NirsLabProject.config.subject import Subject
from NirsLabProject.config.consts import *


COLOR_CODES = np.array(['black', 'blue', 'red'])


def get_channel_names_and_color(channels: List[str]) -> Dict[str, str]:
    colors = {}
    channels = dict.fromkeys(c[:-1] for c in channels)
    for i, channel in enumerate(channels):
        colors[channel] = COLOR_CODES[i % len(COLOR_CODES)]
    return colors


def add_hypnogram_to_fig(subject: Subject, ax, number_of_channels: int):
    hypnogram = np.loadtxt(subject.paths.subject_hypnogram_path)
    start, end = sleeping_utils.get_hypnogram_indexes_of_first_rem_sleep(subject)
    hypno = hypnogram[(start - TIME_IN_MINUTES_BEFORE_SLEEP_START*HYPNOGRAM_SR):(end + TIME_IN_MINUTES_AFTER_REM_END*HYPNOGRAM_SR)]

    bins = np.arange(hypno.size) * HYPNOGRAM_SAMPLES_INTERVAL_IN_SECONDS

    # make sure that REM is displayed after Wake
    hypno = pd.Series(hypno).map({0: 5, 1: 3, 2: 2, 3: 1, 4: 4}).values

    # reduce data and bin edges to only moments of change in the hypnogram
    # (to avoid drawing thousands of tiny individual lines when sf is high)
    change_points = np.nonzero(np.ediff1d(hypno, to_end=1))
    hypno = hypno[change_points]
    bins = np.append(0, bins[change_points])
    bins[-1] += HYPNOGRAM_SAMPLES_INTERVAL_IN_SECONDS

    ax2 = ax.twiny()
    ax2.tick_params(top=False, labeltop=False, left=False, labelleft=False, right=False, labelright=False, bottom=False,
                    labelbottom=False)
    ax.stairs(number_of_channels / 5 * hypno.clip(0), bins, fill=True, color='#DCDCDC')


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


# based on https://pythontic.com/visualization/charts/spikerasterplot
def create_raster_plot(subject: Subject, spikes: Dict[str, np.ndarray],
                       add_histogram: bool = True, add_hypnogram: bool = True, show: bool = False):
    channels_name = list(spikes.keys())
    channels_data = list(spikes.values())

    # set plot size and locations
    fig = plt.figure(layout='constrained')
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    plt.title(subject.name)

    # assign a different color for every channel's group (same electrod2)
    channels_color = get_channel_names_and_color(channels_name)
    colors = [channels_color[channel[:-1]] for channel in channels_name]
    colors.reverse()

    # reverse the data so the deepest channel of every electrode will be above the rest
    channels_name.reverse()
    channels_data.reverse()

    if add_hypnogram:
        # cuts the spikes data, so the plot will contain only spikes of the first sleep cycle
        start, end = sleeping_utils.get_timestamps_in_seconds_of_first_rem_sleep(subject)
        channels_data = list(map(lambda x: x[np.where(np.logical_and(x >= start, x <= end))] - start, channels_data))

    add_histogram and add_histogram_to_fig(ax, channels_data)

    # add the main plot - the raster
    ax.eventplot(
        positions=channels_data,
        color=colors,
        linelengths=[0.3]*len(channels_data),
        linewidth=[2]*len(channels_data)
    )

    # set y axis labels
    yticks = []
    for c in channels_name:
        if c in SLEEP_STATES:
            yticks.append(c)
        elif c[-1] == '1':
            yticks.append(c[:-1])
        else:
            yticks.append('')

    # set y axis labels
    ax.set_yticklabels(yticks)
    ax.set_yticks(np.arange(len(channels_name)))
    plt.yticks(fontsize=8)

    # set x axis labels values to minutes
    xticks = ax.get_xticklabels()
    [tick.set_text(int(tick.get_text()) // 60) for tick in xticks[1:]]
    ax.set_xticklabels(xticks)
    plt.xlabel('Minutes')

    # set plot proportions
    fig.set_figwidth(14)
    fig.set_figheight(7)

    if add_hypnogram:
        add_hypnogram_to_fig(subject, ax, len(channels_data))
        plt.savefig(subject.paths.subject_hypno_raster_plot_path, dpi=1000)
    else:
        plt.savefig(subject.paths.subject_raster_plot_path, dpi=1000)

    if show:
        plt.show()


def create_ERP_plot(subject: Subject, channel_raw: mne.io.Raw, channel_data: np.ndarray,
                    spikes: np.ndarray, channel_name: str, show: bool = False):
    """
    channel_data: is a (N,) np array with the MNE channel (it can be filtred, etc..)
    """
    epochs = utils.create_epochs(channel_raw, channel_data, spikes)
    fig = epochs.plot_image(
        show=show,
        picks=[channel_name],
        vmin=-150,
        vmax=150,
        title=f'{subject.name}-{channel_name}-ERP'
    )[0]
    fig.savefig(os.path.join(subject.paths.subject_erp_plots_dir_path, f'{subject.name}-{channel_name}.png'),  dpi=1000)


def create_TFR_plot(subject: Subject, channel_raw: mne.io.Raw, channel_data: np.ndarray,
                    spikes: np.ndarray, channel_name: str, show: bool = False):
    epochs = utils.create_epochs(channel_raw, channel_data, spikes, -1, 1)
    freqs = np.logspace(*np.log10([LOW_THRESHOLD_FREQUENCY, HIGH_THRESHOLD_FREQUENCY]), num=100)
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
        title=f'{subject.name}-{channel_name}-TFR'
    )
    plt.savefig(os.path.join(subject.paths.subject_tfr_plots_dir_path, f'{subject.name}-{channel_name}.png'),  dpi=1000)


def create_PSD_plot(subject: Subject, channel_raw: mne.io.Raw, channel_name: str, show: bool = False):
    spectrum = channel_raw.compute_psd(
        fmin=0,
        fmax=250,
        picks=[channel_name],
    )
    spectrum.plot(
        show=show,
    )
    plt.title(f'{subject.name}-{channel_name}-PSD')
    plt.savefig(os.path.join(subject.paths.subject_psd_plots_dir_path, f'{subject.name}-{channel_name}.png'),  dpi=1000)
