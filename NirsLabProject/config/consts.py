# -------------------------------------------- RUN CONFIG -----------------------------------------------------------------

FORCE_LOAD_EDF = False
FORCE_DETECT_SPIKES = False
FORCE_CALCULATE_SPIKES_FEATURES = False

# -------------------------------------------- General -----------------------------------------------------------------

STIMULI_PROJECT_FIRST_P_NUMBER = 485
SOURASKY_PROJECT_LAST_P_NUMBER = 300


SR = 1000
DIVISION_FACTOR = 4

SCALP_MODEL_PROBABILITIES = 0.8
SCALP_MODEL_NAME = 'lgbm_full_origin_70_19.pkl'

SPIKE_RANGE_SECONDS = .25

LOW_THRESHOLD_FREQUENCY = 5
HIGH_THRESHOLD_FREQUENCY = 250

# based on https://academic.oup.com/brain/article/146/5/1903/7024726
MAX_SPIKE_LENGTH_MILLISECONDS = 70
MIN_SPIKE_LENGTH_MILLISECONDS = 20


# In stimuli project might be 1 instead of 3
MIN_AMPLITUDE_Z_SCORE = 1
MAX_AMPLITUDE_Z_SCORE = 30

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
GROUP_FOCAL_INDEX = 8
GROUP_EVENT_DURATION_INDEX = 9
GROUP_EVENT_SIZE_INDEX = 10
GROUP_EVENT_DEEPEST_INDEX = 11
GROUP_EVENT_SHALLOWEST_INDEX = 12
GROUP_EVENT_SPATIAL_SPREAD_INDEX = 13
IS_IN_SCALP_INDEX = 14
STIMULI_FLAG_INDEX = 15
HYPNOGRAM_FLAG_INDEX = 16
SUBJECT_NUMBER = 17

STIMULI_FLAG_BEFORE_FIRST_STIMULI_SESSION = 0
STIMULI_FLAG_STIMULI_SESSION = 1  # during stimuli session that have multiple windows
STIMULI_FLAG_DURING_STIMULI_BLOCK = 2  # during a stimuli window
STIMULI_FLAG_AFTER_STIMULI_SESSION = 3

HYPNOGRAM_FLAG_REM_OR_WAKE = 0
HYPNOGRAM_FLAG_NREM = 1

# ----------------------------------------------------------------------------------------------------------------------

DETECTION_PROJECT_INTERSUBJECTS_SUBJECT_NAME = 'd2'
STIMULI_PROJECT_INTERSUBJECTS_SUBJECT_NAME = 's1'


# ----------------------------------------------------------------------------------------------------------------------

SHARED_GOOGLE_DRIVE_PATH = 'https://drive.google.com/drive/u/1/folders/15kVUjay5dT-u4yyNMsXExfvolwErD-78'
GOOGLE_DRIVE_LINK = 'https://drive.google.com/drive/u/1/folders/1Ujjzk1WRMnNM48ttfm5VO9A_fqZ-8fyL'
DETECTION_PROJECT_GOOGLE_FRIVE_LINK = 'https://drive.google.com/drive/u/1/folders/1cLBAXj-eKIwxsE81gjt2MUwAur5xRjrq'
