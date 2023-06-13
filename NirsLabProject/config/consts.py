# -------------------------------------------- General -----------------------------------------------------------------

SUBJECT = 'p406'

SR = 1000
DIVISION_FACTOR = 4

SPIKE_RANGE_SECONDS = .25

LOW_THRESHOLD_FREQUENCY = 5
HIGH_THRESHOLD_FREQUENCY = 250

# based on https://academic.oup.com/brain/article/146/5/1903/7024726
MAX_SPIKE_LENGTH_MILLISECONDS = 70

# In stimuli project might be 1 instead of 3
MIN_AMPLITUDE_Z_SCORE = 3

# https://academic.oup.com/brain/article/146/5/1903/7024726
SPIKES_GROUPING_WINDOW_SIZE = 100

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
# ----------------------------------------------- Plots ----------------------------------------------------------------

STIMULI_COLOR = '#0f0f0f50'
REM_COLOR = '#CBE9F8'
WAKE_COLOR = '#DCDCDC'
NREM_COLOR = 'white'

# ----------------------------------------------------------------------------------------------------------------------
NUM_OF_FEATURES = 7
TIMESTAMP_INDEX = 0
CHANNEL_INDEX = 1
AMPLITUDE_INDEX = 2
DURATION_INDEX = 3
CORD_X_INDEX = 4
CORD_Y_INDEX = 5
CORD_Z_INDEX = 6

# Additional features - added late
GROUP_INDEX = 7
