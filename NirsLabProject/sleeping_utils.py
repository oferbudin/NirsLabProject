import numpy as np
from yasa import hypno
from typing import Tuple

from NirsLabProject.config.consts import *
from NirsLabProject.config.subject import Subject

SLEEPING_CYCLE = {WAKE, N1, N2, N3, REM}
OPTIONAL_STAGES = {WAKE, N1}


# TODO: improve - handle
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
