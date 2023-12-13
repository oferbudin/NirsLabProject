import os
import numpy as np
import pandas as pd
from yasa import hypno
import scipy.io as sio
from typing import Tuple

from NirsLabProject.config.consts import *
from NirsLabProject.config.subject import Subject

SLEEPING_CYCLE = {WAKE, N1, N2, N3, REM}
OPTIONAL_STAGES = {WAKE, N1}


def get_hypnogram_indexes_of_first_rem_sleep(subject: Subject) -> Tuple[int, int]:
    hypnogram = np.loadtxt(subject.paths.subject_hypnogram_path)
    sleeping_stages = hypno.hypno_find_periods(hypnogram, sf_hypno=HYPNOGRAM_SF, threshold='5min')
    first_rem = sleeping_stages[sleeping_stages['values'] == REM].min()
    last_wakes = sleeping_stages[np.logical_and(sleeping_stages['start'] < first_rem['start'],
                                sleeping_stages['values'] == WAKE)]
    last_wake = last_wakes.iloc[-1]
    return int(last_wake['start'] + last_wake['length']), int(first_rem['start'] + first_rem['length'])


# adds TIME_IN_MINUTES_BEFORE_SLEEP_START before and TIME_IN_MINUTES_AFTER_REM_END after
def get_timestamps_in_seconds_of_first_rem_sleep(subject: Subject) -> Tuple[int, int]:
    start_index, end_index = get_hypnogram_indexes_of_first_rem_sleep(subject)
    start_timestamp = start_index*HYPNOGRAM_SAMPLES_INTERVAL_IN_SECONDS - TIME_IN_MINUTES_BEFORE_SLEEP_START*60
    end_timestamp = end_index*HYPNOGRAM_SAMPLES_INTERVAL_IN_SECONDS + TIME_IN_MINUTES_AFTER_REM_END*60
    return start_timestamp, end_timestamp


def get_hypnogram_changes_in_miliseconds(subject: Subject):
    if os.path.exists(subject.paths.subject_hypnogram_path):
        hypnogram = np.loadtxt(subject.paths.subject_hypnogram_path, dtype=int)
        hypnogram = pd.Series(hypnogram).map(
            {
                N1: HYPNOGRAM_FLAG_NREM,
                N2: HYPNOGRAM_FLAG_NREM,
                N3: HYPNOGRAM_FLAG_NREM,
                REM: HYPNOGRAM_FLAG_REM_OR_WAKE,
                WAKE: HYPNOGRAM_FLAG_REM_OR_WAKE
            }
        ).values
        indices = np.where(hypnogram[:-1] != hypnogram[1:])[0] + 1
        indices = np.insert(indices, 0, 0)
        changing_points = indices * HYPNOGRAM_SAMPLES_INTERVAL_IN_SECONDS * SR  # Assuming each sample represents 30 seconds
        values = hypnogram[indices]
    elif os.path.exists(subject.paths.subject_sleep_scoring_path):
        f = sio.loadmat(subject.paths.subject_sleep_scoring_path)
        data = f['sleep_score']
        hypnogram = np.array(data)[0]
        hypnogram = pd.Series(hypnogram).map(
            {
                0: HYPNOGRAM_FLAG_REM_OR_WAKE,
                -1: HYPNOGRAM_FLAG_REM_OR_WAKE,
                1: HYPNOGRAM_FLAG_NREM,
            }
        ).values
        changing_points = np.where(hypnogram[:-1] != hypnogram[1:])[0] + 1
        changing_points = np.insert(changing_points, 0, 0)
        values = hypnogram[changing_points]
    else:
        raise Exception(f'No hypnogram file found for subject {subject.name}')
    return changing_points, values


def get_sleep_start_end_indexes(subject: Subject):
    changing_points, values = get_hypnogram_changes_in_miliseconds(subject)
    first_rem_index = np.where(values == HYPNOGRAM_FLAG_NREM)[0][0]
    last_wake_index = np.where(values == HYPNOGRAM_FLAG_REM_OR_WAKE)[0][-1]
    return changing_points[first_rem_index], changing_points[last_wake_index]

