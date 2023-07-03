import os
import mne
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from mne.time_frequency import tfr_morlet
import scipy.io as sio
from nilearn import plotting


from NirsLabProject.utils import general_utils as utils
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
        edges = np.arange(tmax) * HYPNOGRAM_SAMPLES_INTERVAL_IN_SECONDS

        divide_by = 1
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
        divide_by = 1000  # because sleep scoring is in milliseconds
    else:
        raise Exception(f'No hypnogram or sleep score file found for subject {subject.subject_id}')

    # reduce data and bin edges to only moments of change in the hypnogram
    # (to avoid drawing thousands of tiny individual lines when sf is high)
    change_points = np.nonzero(np.ediff1d(hypno, to_end=1))
    values = hypno[change_points]
    values = number_of_channels / 5 * values.clip(0)
    edges = np.append(0, edges[change_points[0] // divide_by])
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
    binwidth = 60  # how many seconds are in one bin
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
    for i, channel_name in enumerate(channels_name):
        if i == len(channels_name)-1 or channel_name[:-1] != channels_name[i+1][:-1]:
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


# def show_electrodes(raw: mne.io.Raw, subject: Subject):
#     cords = utils.calculate_coordinates()
#     ch_names = set(name.replace('RACr', 'RAC').replace('LACr', 'LAC') for name in raw.ch_names).intersection(cords.keys())
#     ch_names = [name for name in ch_names]
#     raw = raw.copy().pick_channels(ch_names)
#     montage = mne.channels.make_dig_montage(
#         {
#             name:np.array(cords, dtype=np.float32)/1000 for name, cords in cords.items() if name in set(raw.ch_names)
#         },
#         coord_frame='mri',
#         nasion=np.array([0, 0, 0], dtype=np.float32),
#     )
#
#     epochs = mne.Epochs(raw, np.array([[0, 0, 0]]))
#     epochs.set_montage(montage)
#
#
#     trans = mne.channels.compute_native_head_t(montage)
#     sample_path = mne.datasets.sample.data_path()
#     subjects_dir = sample_path / "subjects"
#
#     view_kwargs = dict(azimuth=105, elevation=100, focalpoint=(0, 0, -15))
#     brain = mne.viz.Brain(
#         "fsaverage",
#         subjects_dir=subjects_dir,
#         cortex="low_contrast",
#         alpha=0.25,
#         background="white",
#     )
#
#     brain.add_volume_labels(aseg="aparc+aseg", labels=('Left-Amygdala', 'Right-Amygdala', 'ctx-rh-lateralorbitofrontal', 'ctx-rh-medialorbitofrontal'), alpha=0.5)
#
#     brain.add_sensors(epochs.info, trans=trans)
#     brain.add_head(alpha=0.25, color="tan")
#
#     path = subject.paths.subject_electrodes_dir_path
#
#     brain.show_view(view='medial', distance=400)
#     brain.save_image(os.path.join(path, 'medial.png'))
#
#     brain.reset_view()
#     brain.show_view(view='ventral', distance=400)
#     brain.save_image(os.path.join(path, 'ventral.png'))
#
#     brain.reset_view()
#     brain.show_view(view='frontal', distance=400)
#     brain.save_image(os.path.join(path, 'frontal.png'))
#
#     brain.close()

def save_electrodes_position(raw: mne.io.Raw, subject: Subject, stimulation_locations: List[str]):
    print('Saving electrodes position')
    ch_to_cord = utils.calculate_coordinates()
    ch_names = set(name for name in raw.ch_names).intersection(ch_to_cord.keys())
    ch_names = sorted([name for name in ch_names])

    marker_labels = []
    last_organ = ""
    colors = []
    for i, channel in enumerate(ch_names):
        current_organ = channel[:-1]
        if current_organ != last_organ:
            last_organ = current_organ
            marker_labels.append(current_organ)
        else:
            marker_labels.append("")
        colors.append('black')

        if channel in stimulation_locations:
            colors[i] = 'red'
            if marker_labels[i] == '':
                marker_labels[i] = 'S'
            else:
                marker_labels[i] += '-S'

    cords = [ch_to_cord[name] for name in ch_names]
    view = plotting.view_markers(cords, marker_size=5, marker_labels=marker_labels, marker_color=colors)

    view.save_as_html(os.path.join(subject.paths.subject_electrodes_dir_path, 'electrodes.html'))


# Generate a color gradient between red and yellow and return the RGB color as a dictionary
# with the value as a key and the RGB color as a value
def generate_color_gradient(values: list) -> dict:

    def get_color(value, min_value, max_value):
        # Normalize the value between 0 and 1
        normalized_value = 1 - (value - min_value) / (max_value - min_value)

        # Interpolate the hue value between red (0) and yellow (60) in the HSL color space
        hue = 0 + normalized_value * 240

        # Convert HSL color to RGB color
        return mcolors.hsv_to_rgb((hue / 360, 1, 1))

    values = values.copy()
    values.sort()

    # Generate colors for each value
    return {value: get_color(value, min(values), max(values)) for value in values}


# feature must be in format: {'channel name': {'value': '', 'cords': ''}}
def plot_feature_on_electrodes(subject: Subject, features: dict, name: str, unit: str = ''):
    values = [d['value'] for ch, d in features.items()]
    colors_plate = generate_color_gradient(values)

    colors = []
    cords = []
    marker_labels = []
    last_organ = ""
    for channel in sorted(features.keys()):
        current_organ = channel[:-1]
        if current_organ != last_organ:
            last_organ = current_organ
            marker_labels.append(current_organ)
        else:
            marker_labels.append("")
        colors.append(colors_plate[features[channel]['value']])
        cords.append(features[channel]['cords'])

    # Add legend to the 3D brain with the min and max values and their colors
    cords.extend([(-60, -90, 85), (-60, -90, 70), (-60, -90, 55)])
    colors.extend([colors_plate[max(values)], mcolors.hsv_to_rgb((120 / 360, 1, 1)), colors_plate[min(values)]])
    if type(values[0]) == float or type(values[0]) == np.float64:
        marker_labels.extend([f"{max(values):.1f}{unit}", "", f"{min(values):.1f}{unit}"])
    else:
        marker_labels.extend([f"{max(values)}{unit}", "", f"{min(values)}{unit}"])

    view = plotting.view_markers(cords, colors, marker_size=10, marker_labels=marker_labels)
    view.save_as_html(os.path.join(subject.paths.subject_features_3d_plots_dir_path, f'{name}.html'))


# Generate a color gradient between red and yellow based on the avarage spike amplitude
# Gets a dict of channels and their spikes features
def plot_avg_spike_amplitude_by_electrode(subject: Subject, channels_spikes_features: Dict[str, np.ndarray]):
    ch_avrage_amp = {}
    for ch, features in channels_spikes_features.items():
        if features[0, CORD_X_INDEX] == np.NAN:
            continue
        ch_avrage_amp[ch] = {}
        ch_avrage_amp[ch]['value'] = np.average(features[:, AMPLITUDE_INDEX])
        ch_avrage_amp[ch]['cords'] = (features[0, CORD_X_INDEX], features[0, CORD_Y_INDEX], features[0, CORD_Z_INDEX])
    plot_feature_on_electrodes(subject, ch_avrage_amp, "avrage_amplitude", 'σ')


# Generate a color gradient between red and yellow based on the avarage spike amplitude
# Gets a dict of channels and their spikes features
def plot_number_of_spikes_by_electrode(subject: Subject, channels_spikes_features: Dict[str, np.ndarray]):
    ch_count_amp = {}
    for ch, features in channels_spikes_features.items():
        if features[0, CORD_X_INDEX] == np.NAN:
            continue
        ch_count_amp[ch] = {}
        ch_count_amp[ch]['value'] = features[:, CHANNEL_INDEX].size
        ch_count_amp[ch]['cords'] = (features[0, CORD_X_INDEX], features[0, CORD_Y_INDEX], features[0, CORD_Z_INDEX])
    plot_feature_on_electrodes(subject, ch_count_amp, "number_of_spikes")


def create_raincloud_plot(figure_path: str, list_of_arrys: List[np.array], feature_name: str, description, xticks: list, yticklabels: list = None, yticks: list = None, is_discrete: bool = False):
    fig, ax = plt.subplots(figsize=(8, 4))


    if is_discrete:
        plt.hist(list_of_arrys[0], bins=range(int(min(list_of_arrys[0])), int(max(list_of_arrys[0]) + 2)), edgecolor='black', orientation='horizontal', alpha=0.5, label=xticks[0], density=True)
        plt.hist(list_of_arrys[1], bins=range(int(min(list_of_arrys[1])), int(max(list_of_arrys[1]) + 2)), edgecolor='black', orientation='horizontal',  alpha=0.5, label=xticks[1], density=True)
        plt.legend(loc='upper right')
        plt.xlabel('Density')
    else:

        # Create a list of colors for the boxplots based on the number of features you have
        boxplots_colors = ['yellowgreen', 'olivedrab', 'darkolivegreen', 'darkseagreen', 'lightgreen']

        # Boxplot data
        bp = ax.boxplot(list_of_arrys, patch_artist=True,
                        notch='True', showfliers=False)

        # Change to the desired color and add transparency
        for patch, color in zip(bp['boxes'], boxplots_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.1)

        # Create a list of colors for the violin plots based on the number of features you have
        violin_colors = ['darksalmon', 'orchid', 'skyblue', 'lightgreen', 'gold']

        # Violinplot data
        vp = ax.violinplot(list_of_arrys, points=100, widths=0.7, showmeans=False, showextrema=False, showmedians=False)

        for idx, b in enumerate(vp['bodies']):
            # Change the color of each violin
            b.set_color(violin_colors[idx])

        # Create a list of colors for the scatter plots based on the number of features you have
        scatter_colors = ['tomato', 'darksalmon', 'orchid', 'skyblue', 'lightgreen', 'gold']

        # Scatterplot data
        for idx, features in enumerate(list_of_arrys):
            # Add jitter effect so the features do not overlap on the y-axis
            offset = 0.4 if idx % 2 == 0 else 1.6
            x = np.full(len(features), idx + offset)
            idxs = np.arange(len(x))
            out = x.astype(float)
            out.flat[idxs] += np.random.uniform(low=-.15, high=.15, size=len(idxs))
            x = out
            plt.scatter(x, features, s=.08, c=scatter_colors[idx])
        plt.xticks(np.arange(1, len(list_of_arrys) + 1, 1), xticks)  # Set text labels.

    if yticks:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    plt.ylabel(feature_name)
    plt.title(f"{feature_name} raincloud plot")
    plt.text(0, -0.2, description, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    fig.savefig(os.path.join(figure_path, feature_name), bbox_inches='tight', dpi=300)


def get_scalp_intracranial_correlation_raincloud_text(list_of_arrys: List[np.array], is_group: bool):
    t_p_val, _ = utils.t_test(list_of_arrys[0], list_of_arrys[1])
    return f'{len(list_of_arrys[0])}/{len(list_of_arrys[0]) + len(list_of_arrys[1])} {"groups" if is_group else "spikes"} detected by scalp model | t test p-value{t_p_val}'


def create_raincloud_plot_for_all_spikes_features(subject: Subject, flat_features: np.ndarray, index_to_channel_name: Dict[int, str]):
    sum(flat_features[:, IS_IN_SCALP_INDEX])

    group_ids = flat_features[:, GROUP_INDEX]
    unique_indices = np.unique(group_ids, return_index=True)[1]
    unique_group_flat = flat_features[unique_indices]

    xticks = ['Scalp Detection', 'No Scalp Detection']
    path = subject.paths.subject_raincloud_plots_dir_path

    ampliudes_of_detected = flat_features[flat_features[:, IS_IN_SCALP_INDEX] == 1][:, AMPLITUDE_INDEX]
    ampliudes_of_not_detected = flat_features[flat_features[:, IS_IN_SCALP_INDEX] == 0][:, AMPLITUDE_INDEX]
    create_raincloud_plot(
        figure_path=path,
        list_of_arrys=[ampliudes_of_detected, ampliudes_of_not_detected],
        feature_name="Spike Amplitude (σ)",
        description=get_scalp_intracranial_correlation_raincloud_text([ampliudes_of_detected, ampliudes_of_not_detected], is_group=False),
        xticks=xticks,
    )

    durations_of_detected = flat_features[flat_features[:, IS_IN_SCALP_INDEX] == 1][:, DURATION_INDEX]
    durations_of_not_detected = flat_features[flat_features[:, IS_IN_SCALP_INDEX] == 0][:, DURATION_INDEX]
    create_raincloud_plot(
        figure_path=path,
        list_of_arrys=[durations_of_detected, durations_of_not_detected],
        feature_name="Spike Duration (ms)",
        description=get_scalp_intracranial_correlation_raincloud_text([durations_of_detected, durations_of_not_detected], is_group=False),
        xticks=xticks,
    )

    event_focal_of_detected = unique_group_flat[unique_group_flat[:, IS_IN_SCALP_INDEX] == 1][:, GROUP_FOCAL_INDEX]
    event_focal_of_not_detected = unique_group_flat[unique_group_flat[:, IS_IN_SCALP_INDEX] == 0][:, GROUP_FOCAL_INDEX]
    yticklabels = []
    yticks = []
    for index in index_to_channel_name.keys():
        if index_to_channel_name[index][:-1] not in yticklabels:
            yticklabels.append(index_to_channel_name[index][:-1])
            yticks.append(index)
    print(yticklabels)
    print(yticks)

    create_raincloud_plot(
        figure_path=path,
        list_of_arrys=[event_focal_of_detected, event_focal_of_not_detected],
        feature_name="Group Event Focal Electrode",
        description=get_scalp_intracranial_correlation_raincloud_text([event_focal_of_detected, event_focal_of_not_detected], is_group=True),
        xticks=xticks,
        yticklabels=yticklabels,
        yticks=yticks,
        is_discrete=True,
    )

    event_electrodes_of_detected = flat_features[flat_features[:, IS_IN_SCALP_INDEX] == 1][:, CHANNEL_INDEX]
    event_electrodes_of_not_detected = flat_features[flat_features[:, IS_IN_SCALP_INDEX] == 0][:, CHANNEL_INDEX]
    create_raincloud_plot(
        figure_path=path,
        list_of_arrys=[event_electrodes_of_detected, event_electrodes_of_not_detected],
        feature_name="Spikes Electrodes",
        description=get_scalp_intracranial_correlation_raincloud_text([event_electrodes_of_detected, event_electrodes_of_not_detected], is_group=False),
        xticks=xticks,
        yticklabels=yticklabels,
        yticks=yticks,
        is_discrete=True,
    )

    event_duration_of_detected = unique_group_flat[unique_group_flat[:, IS_IN_SCALP_INDEX] == 1][:, GROUP_EVENT_DURATION_INDEX]
    event_duration_of_not_detected = unique_group_flat[unique_group_flat[:, IS_IN_SCALP_INDEX] == 0][:, GROUP_EVENT_DURATION_INDEX]
    create_raincloud_plot(
        figure_path=path,
        list_of_arrys=[event_duration_of_detected, event_duration_of_not_detected],
        feature_name="Group Event Spreading Duration (ms)",
        description=get_scalp_intracranial_correlation_raincloud_text([event_duration_of_detected, event_duration_of_not_detected], is_group=True),
        xticks=xticks,
    )

    event_size_of_detected = unique_group_flat[unique_group_flat[:, IS_IN_SCALP_INDEX] == 1][:, GROUP_EVENT_SIZE_INDEX]
    event_size_of_not_detected = unique_group_flat[unique_group_flat[:, IS_IN_SCALP_INDEX] == 0][:, GROUP_EVENT_SIZE_INDEX]
    create_raincloud_plot(
        figure_path=path,
        list_of_arrys=[event_size_of_detected, event_size_of_not_detected],
        feature_name="Group Event Size (n electrodes)",
        description=get_scalp_intracranial_correlation_raincloud_text([event_size_of_detected, event_size_of_not_detected], is_group=True),
        xticks=xticks,
    )

    event_deepest_index_of_detected = unique_group_flat[unique_group_flat[:, IS_IN_SCALP_INDEX] == 1][:, GROUP_EVENT_DEEPEST_INDEX]
    event_deepest_index_of_not_detected = unique_group_flat[unique_group_flat[:, IS_IN_SCALP_INDEX] == 0][:, GROUP_EVENT_DEEPEST_INDEX]
    create_raincloud_plot(
        figure_path=path,
        list_of_arrys=[event_deepest_index_of_detected, event_deepest_index_of_not_detected],
        feature_name="Group Event Deepest Electrode Index",
        description=get_scalp_intracranial_correlation_raincloud_text([event_deepest_index_of_detected, event_deepest_index_of_not_detected], is_group=True),
        xticks=xticks,
        is_discrete=True,
    )

    event_shallowest_index_of_detected = unique_group_flat[unique_group_flat[:, IS_IN_SCALP_INDEX] == 1][:, GROUP_EVENT_SHALLOWEST_INDEX]
    event_shallowest_index_of_not_detected = unique_group_flat[unique_group_flat[:, IS_IN_SCALP_INDEX] == 0][:, GROUP_EVENT_SHALLOWEST_INDEX]
    create_raincloud_plot(
        figure_path=path,
        list_of_arrys=[event_shallowest_index_of_detected, event_shallowest_index_of_not_detected],
        feature_name="Group Event Shallowest Electrode Index",
        description=get_scalp_intracranial_correlation_raincloud_text([event_shallowest_index_of_detected, event_shallowest_index_of_not_detected], is_group=True),
        xticks=xticks,
        is_discrete=True,
    )

    spatial_spread_index_of_detected = unique_group_flat[unique_group_flat[:, IS_IN_SCALP_INDEX] == 1][:, GROUP_EVENT_SPATIAL_SPREAD_INDEX]
    spatial_spread_index_of_detected = spatial_spread_index_of_detected[spatial_spread_index_of_detected > 0]
    spatial_spread_index_of_not_detected = unique_group_flat[unique_group_flat[:, IS_IN_SCALP_INDEX] == 0][:, GROUP_EVENT_SPATIAL_SPREAD_INDEX]
    spatial_spread_index_of_not_detected = spatial_spread_index_of_not_detected[spatial_spread_index_of_not_detected > 0]
    create_raincloud_plot(
        figure_path=path,
        list_of_arrys=[spatial_spread_index_of_detected, spatial_spread_index_of_not_detected],
        feature_name="Group Event Spatial Spread",
        description=get_scalp_intracranial_correlation_raincloud_text([spatial_spread_index_of_detected, spatial_spread_index_of_not_detected], is_group=True),
        xticks=xticks,
    )


def get_stimuli_effects_raincloud_text(list_of_arrys: List[np.array], is_group: bool = False):
    t_p_val1, _ = utils.t_test(list_of_arrys[0], list_of_arrys[1])
    t_p_val2, _ = utils.t_test(list_of_arrys[1], list_of_arrys[2])
    t_p_val3, _ = utils.t_test(list_of_arrys[0], list_of_arrys[2])

    return f'Before: {len(list_of_arrys[0])} | During: {len(list_of_arrys[1])} | After: {len(list_of_arrys[2])} {"groups" if is_group else "spikes"} detected by scalp model' \
           f'\nBefore - During t test p-value{t_p_val1}' \
           f'\nDuring - After t test p-value{t_p_val2}' \
           f'\nBefore - After t test p-value{t_p_val3}'


def stimuli_effects_raincloud_plots(subject: Subject, flat_features: np.ndarray, index_to_channel: Dict[int, str]):
    before, during, after = utils.stimuli_effects(subject, flat_features)

    create_raincloud_plot(
        figure_path=subject.paths.subject_raincloud_plots_dir_path,
        list_of_arrys=[before[:, AMPLITUDE_INDEX], during[:, AMPLITUDE_INDEX], after[:, AMPLITUDE_INDEX]],
        feature_name='Stimuli Effects on Amplitude (σ)',
        description=get_stimuli_effects_raincloud_text([before[:, AMPLITUDE_INDEX], during[:, AMPLITUDE_INDEX], after[:, AMPLITUDE_INDEX]]),
        yticks=['Before', 'During', 'After'],
    )

    create_raincloud_plot(
        figure_path=subject.paths.subject_raincloud_plots_dir_path,
        list_of_arrys=[before[:, DURATION_INDEX], during[:, DURATION_INDEX], after[:, DURATION_INDEX]],
        feature_name='Stimuli Effects on Spike Duration (ms)',
        description=get_stimuli_effects_raincloud_text([before[:, DURATION_INDEX], during[:, DURATION_INDEX], after[:, DURATION_INDEX]]),
        yticks=['Before', 'During', 'After'],
    )

    group_ids = flat_features[:, GROUP_INDEX]
    unique_indices = np.unique(group_ids, return_index=True)[1]
    unique_group_flat = flat_features[unique_indices]
    before, during, after = utils.stimuli_effects(subject, unique_group_flat)

    xticklabels = []
    xticks = []
    for index in reversed(index_to_channel.keys()):
        if index_to_channel[index][:-1] not in xticklabels:
            xticklabels.append(index_to_channel[index][:-1])
            xticks.append(index)

    create_raincloud_plot(
        figure_path=subject.paths.subject_raincloud_plots_dir_path,
        list_of_arrys=[before[:, GROUP_FOCAL_INDEX], during[:, GROUP_FOCAL_INDEX], after[:, GROUP_FOCAL_INDEX]],
        feature_name='Stimuli Effects on Group Event Focal Electrode',
        description=get_stimuli_effects_raincloud_text([before[:, GROUP_FOCAL_INDEX], during[:, GROUP_FOCAL_INDEX], after[:, GROUP_FOCAL_INDEX]], is_group=True),
        yticks=['Before', 'During', 'After'],
        xticks=xticks,
        xticklabels=xticklabels,
        is_discrete=True,
    )

    create_raincloud_plot(
        figure_path=subject.paths.subject_raincloud_plots_dir_path,
        list_of_arrys=[before[:, GROUP_EVENT_DURATION_INDEX], during[:, GROUP_EVENT_DURATION_INDEX],
                       after[:, GROUP_EVENT_DURATION_INDEX]],
        feature_name='Stimuli Effects on Group Event Spreading Duration (ms)',
        description=get_stimuli_effects_raincloud_text([before[:, GROUP_EVENT_DURATION_INDEX], during[:, GROUP_EVENT_DURATION_INDEX], after[:, GROUP_EVENT_DURATION_INDEX]], is_group=True),
        yticks=['Before', 'During', 'After'],
    )

    create_raincloud_plot(
        figure_path=subject.paths.subject_raincloud_plots_dir_path,
        list_of_arrys=[before[:, GROUP_EVENT_SIZE_INDEX], during[:, GROUP_EVENT_DURATION_INDEX],
                       after[:, GROUP_EVENT_DURATION_INDEX]],
        feature_name='Stimuli Effects on Group Event Size (n electrodes)',
        description=get_stimuli_effects_raincloud_text([before[:, GROUP_EVENT_SIZE_INDEX], during[:, GROUP_EVENT_DURATION_INDEX], after[:, GROUP_EVENT_DURATION_INDEX]], is_group=True),
        yticks=['Before', 'During', 'After'],
    )

    create_raincloud_plot(
        figure_path=subject.paths.subject_raincloud_plots_dir_path,
        list_of_arrys=[before[:, GROUP_EVENT_DEEPEST_INDEX], during[:, GROUP_EVENT_DEEPEST_INDEX],
                       after[:, GROUP_EVENT_DEEPEST_INDEX]],
        feature_name='Stimuli Effects on Group Group Event Deepest Electrode Index',
        description=get_stimuli_effects_raincloud_text([before[:, GROUP_EVENT_DEEPEST_INDEX], during[:, GROUP_EVENT_DEEPEST_INDEX], after[:, GROUP_EVENT_DEEPEST_INDEX]], is_group=True),
        yticks=['Before', 'During', 'After'],
    )

    create_raincloud_plot(
        figure_path=subject.paths.subject_raincloud_plots_dir_path,
        list_of_arrys=[before[:, GROUP_EVENT_SHALLOWEST_INDEX], during[:, GROUP_EVENT_SHALLOWEST_INDEX],
                       after[:, GROUP_EVENT_SHALLOWEST_INDEX]],
        feature_name='Stimuli Effects on Group Group Event Shallowest Electrode Index',
        description=get_stimuli_effects_raincloud_text([before[:, GROUP_EVENT_SHALLOWEST_INDEX], during[:, GROUP_EVENT_SHALLOWEST_INDEX], after[:, GROUP_EVENT_SHALLOWEST_INDEX]], is_group=True),
        yticks=['Before', 'During', 'After'],
    )

    create_raincloud_plot(
        figure_path=subject.paths.subject_raincloud_plots_dir_path,
        list_of_arrys=[before[:, GROUP_EVENT_SPATIAL_SPREAD_INDEX], during[:, GROUP_EVENT_SPATIAL_SPREAD_INDEX],
                       after[:, GROUP_EVENT_SPATIAL_SPREAD_INDEX]],
        feature_name='Stimuli Effects on Group Event Spatial spread',
        description=get_stimuli_effects_raincloud_text([before[:, GROUP_EVENT_SPATIAL_SPREAD_INDEX], during[:, GROUP_EVENT_SPATIAL_SPREAD_INDEX], after[:, GROUP_EVENT_SPATIAL_SPREAD_INDEX]], is_group=True),
        yticks=['Before', 'During', 'After'],
    )
