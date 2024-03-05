# -------------------------------------------- RUN CONFIG -----------------------------------------------------------------

DOWNLOAD_FROM_GOOGLE_DRIVE = False
FORCE_LOAD_EDF = False
FORCE_DETECT_SPIKES = True
FORCE_CALCULATE_SPIKES_FEATURES = True

# -------------------------------------------- General -----------------------------------------------------------------

STIMULI_PROJECT_FIRST_P_NUMBER = 485
SOURASKY_PROJECT_LAST_P_NUMBER = 300


SR = 1000
DIVISION_FACTOR = 4

SCALP_MODEL_PROBABILITIES = 0.8
SCALP_MODEL_NAME = 'xgb_full_origin_19_fix_prob.pkl'

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
SPIKES_GROUPING_WINDOW_SIZE = 200

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
TIMESTAMP_INDEX = 0
CHANNEL_INDEX = TIMESTAMP_INDEX+1
AMPLITUDE_INDEX = CHANNEL_INDEX+1
DURATION_INDEX = AMPLITUDE_INDEX+1
CORD_X_INDEX = DURATION_INDEX+1
CORD_Y_INDEX = CORD_X_INDEX+1
CORD_Z_INDEX = CORD_Y_INDEX+1

# Additional features - added late
GROUP_INDEX = CORD_Z_INDEX+1
GROUP_FOCAL_INDEX = GROUP_INDEX+1
GROUP_EVENT_DURATION_INDEX = GROUP_FOCAL_INDEX+1
GROUP_EVENT_SIZE_INDEX = GROUP_EVENT_DURATION_INDEX+1
GROUP_EVENT_DEEPEST_INDEX = GROUP_EVENT_SIZE_INDEX+1
GROUP_EVENT_SHALLOWEST_INDEX = GROUP_EVENT_DEEPEST_INDEX+1
GROUP_EVENT_SPATIAL_SPREAD_INDEX = GROUP_EVENT_SHALLOWEST_INDEX+1
IS_IN_SCALP_INDEX = GROUP_EVENT_SPATIAL_SPREAD_INDEX+1
STIMULI_FLAG_INDEX = IS_IN_SCALP_INDEX+1
HYPNOGRAM_FLAG_INDEX = STIMULI_FLAG_INDEX+1
SUBJECT_NUMBER = HYPNOGRAM_FLAG_INDEX+1
GROUP_FOCAL_AMPLITUDE_INDEX = SUBJECT_NUMBER+1
ANGLE_INDEX = GROUP_FOCAL_AMPLITUDE_INDEX+1
RELATIVE_SPIKE_AMPLITUDE_INDEX = ANGLE_INDEX+1
RELATIVE_SPIKE_DURATION_INDEX = RELATIVE_SPIKE_AMPLITUDE_INDEX+1

NUM_OF_FEATURES = RELATIVE_SPIKE_DURATION_INDEX+1

FEATURES_NAMES = {
    TIMESTAMP_INDEX: 'timestamp',
    CHANNEL_INDEX: 'channel',
    AMPLITUDE_INDEX: 'amplitude',
    DURATION_INDEX: 'duration',
    CORD_X_INDEX: 'cord_x',
    CORD_Y_INDEX: 'cord_y',
    CORD_Z_INDEX: 'cord_z',
    GROUP_INDEX: 'group',
    GROUP_FOCAL_INDEX: 'group_focal',
    GROUP_EVENT_DURATION_INDEX: 'group_event_duration',
    GROUP_EVENT_SIZE_INDEX: 'group_event_size',
    GROUP_EVENT_DEEPEST_INDEX: 'group_event_deepest',
    GROUP_EVENT_SHALLOWEST_INDEX: 'group_event_shallowest',
    GROUP_EVENT_SPATIAL_SPREAD_INDEX: 'group_event_spatial_spread',
    IS_IN_SCALP_INDEX: 'is_in_scalp',
    STIMULI_FLAG_INDEX: 'stimuli_flag',
    HYPNOGRAM_FLAG_INDEX: 'hypnogram_flag',
    SUBJECT_NUMBER: 'subject_number',
    GROUP_FOCAL_AMPLITUDE_INDEX: 'group_focal_amplitude',
    ANGLE_INDEX: 'angle',
    RELATIVE_SPIKE_AMPLITUDE_INDEX: 'relative_spike_amplitude',
    RELATIVE_SPIKE_DURATION_INDEX: 'relative_spike_duration'
}


STIMULI_FLAG_BEFORE_FIRST_STIMULI_SESSION = 0
STIMULI_FLAG_STIMULI_SESSION = 1  # during stimuli session that have multiple windows
STIMULI_FLAG_DURING_STIMULI_BLOCK = 2  # during a stimuli window
STIMULI_FLAG_AFTER_STIMULI_SESSION = 3
STIMULI_FLAGS_NAMES = {
    STIMULI_FLAG_BEFORE_FIRST_STIMULI_SESSION: 'before_first_stimuli_session',
    STIMULI_FLAG_STIMULI_SESSION: 'pause_block',
    STIMULI_FLAG_DURING_STIMULI_BLOCK: 'stimuli_block',
    STIMULI_FLAG_AFTER_STIMULI_SESSION: 'after_last_stimuli_block'
}

HYPNOGRAM_FLAG_REM_OR_WAKE = 0
HYPNOGRAM_FLAG_NREM = 1
HYPNOGRAM_FLAG_REM = 2
HYPNOGRAM_FLAG_WAKE = 3
HYPNOGRAM_FLAGS_NAMES = {
    HYPNOGRAM_FLAG_REM_OR_WAKE: 'rem/wake',
    HYPNOGRAM_FLAG_NREM: 'nrem',
}

DETECTION_HYPNOGRAM_FALGS_NAMES = {
    HYPNOGRAM_FLAG_NREM: 'nrem',
    HYPNOGRAM_FLAG_WAKE: 'wake',
    HYPNOGRAM_FLAG_REM: 'rem'
}

# ----------------------------------------------------------------------------------------------------------------------

DETECTION_PROJECT_INTERSUBJECTS_SUBJECT_NAME = 'd2'
STIMULI_PROJECT_INTERSUBJECTS_SUBJECT_NAME = 's1'


# ----------------------------------------------------------------------------------------------------------------------

SHARED_GOOGLE_DRIVE_PATH = 'https://drive.google.com/drive/u/1/folders/15kVUjay5dT-u4yyNMsXExfvolwErD-78'
GOOGLE_DRIVE_LINK = 'https://drive.google.com/drive/u/1/folders/1Ujjzk1WRMnNM48ttfm5VO9A_fqZ-8fyL'
DETECTION_PROJECT_GOOGLE_FRIVE_LINK = 'https://drive.google.com/drive/u/1/folders/1cLBAXj-eKIwxsE81gjt2MUwAur5xRjrq'


