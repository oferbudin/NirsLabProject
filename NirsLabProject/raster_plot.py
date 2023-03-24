import numpy as np
import matplotlib.pyplot as plt

from NirsLabProject.consts import *


COLOR_CODES = np.array([[0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 1]]
)


def get_channel_names_and_color(channels):
    colors = {}
    channels = dict.fromkeys(c[:-1] for c in channels)
    for i, channel in enumerate(channels):
        colors[channel] = COLOR_CODES[i % len(COLOR_CODES)]
    return colors


def add_histogram_to_fig(ax, channels_data):
    # Histogram of every channel
    y = []
    ax_histogram_y = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    ax_histogram_y.tick_params(axis="y", labelleft=False)
    bins = np.arange(0, len(channels_data), 1)
    for i, c in enumerate(channels_data):
        y.extend(len(c) * [i])
    ax_histogram_y.hist(y, bins=bins, orientation='horizontal')

    # Histogram of events in time
    binwidth = 10
    ax_histogram_x = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histogram_x.tick_params(axis="x", labelbottom=False)
    x = np.concatenate(channels_data)
    xymax = np.max(np.abs(x))
    lim = (int(xymax / binwidth) + 1) * binwidth
    bins = np.arange(0, lim + binwidth, binwidth)
    ax_histogram_x.hist(x, bins=bins)


# based of https://pythontic.com/visualization/charts/spikerasterplot
def show_raster_plot(subject_name, spikes, show_histogram=True, show_plot=True):
    channels_name = list(spikes.keys())
    channels_name.reverse()
    channels_data = list(spikes.values())
    channels_data.reverse()

    channels_color = get_channel_names_and_color(channels_name)
    colors = [channels_color[channel[:-1]] for channel in channels_name]

    fig = plt.figure(layout='constrained')
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    plt.title(subject_name)

    # adds the main plot - the raster
    ax.eventplot(
        positions=channels_data,
        color=colors,
        linelengths=[0.3]*len(spikes),
        linewidth=[2]*len(spikes)
    )

    # Set y axis labels
    yticks = [c[:-1] if c[-1] == '1' else '' for c in channels_name]
    ax.set_yticklabels(yticks)
    ax.set_yticks(np.arange(len(channels_name)))
    plt.yticks(fontsize=8)

    # Set x axis labels values to minutes
    xticks = ax.get_xticklabels()
    [tick.set_text(int(tick.get_text()) // 60) for tick in xticks[1:]]
    ax.set_xticklabels(xticks)
    plt.xlabel('Minutes')

    fig.set_figwidth(14)
    fig.set_figheight(7)


    if show_histogram:
        add_histogram_to_fig(ax, channels_data)

    plt.savefig(subject_raster_plot_path, dpi=1000)

    if show_plot:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()


if __name__ == '__main__':
    spikes = np.load(subject_spikes_path)
    show_raster_plot(SUBJECT, spikes)