import numpy as np
import matplotlib.pyplot as plot

from NirsLabProject.consts import *


COLOR_CODES = np.array([[0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 0],
                        [1, 0, 1],
                        [0, 1, 1],
                        [1, 0, 1]]
)


# based of https://pythontic.com/visualization/charts/spikerasterplot
def show_raster_plot(spikes):
    fig, ax = plot.subplots()
    plot.eventplot(spikes.values(), color=COLOR_CODES[:len(spikes)], linelengths=[0.2]*len(spikes))
    ax.set_yticks(np.arange(len(spikes.keys())))
    ax.set_yticklabels(spikes.keys())
    plot.show()

if __name__ == '__main__':
    spikes = np.load(subject_spikes_path)
    show_raster_plot(spikes)