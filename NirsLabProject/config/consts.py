# -------------------------------------------- General -----------------------------------------------------------------

SUBJECT = 'p396'

SR = 1000
DIVISION_FACTOR = 4

SPIKE_RANGE_SECONDS = .25

LOW_THRESHOLD_FREQUENCY = 0.1
HIGH_THRESHOLD_FREQUENCY = 50

# -------------------------------------------- Sleeping ----------------------------------------------------------------

WAKE = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4

TIME_IN_MINUTES_BEFORE_SLEEP_START = 5
TIME_IN_MINUTES_AFTER_REM_END = 5

SLEEP_STATES = ['wake', 'rem', 'n1', 'n2', 'n3']

HYPNOGRAM_SAMPLES_INTERVAL_IN_SECONDS = 30
HYPNOGRAM_SR = 60 // HYPNOGRAM_SAMPLES_INTERVAL_IN_SECONDS
HYPNOGRAM_SF = HYPNOGRAM_SR / 60

# ----------------------------------------------------------------------------------------------------------------------
