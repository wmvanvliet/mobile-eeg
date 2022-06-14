"""
Explore the current mobile-EEG dataset a bit that was send to me by: 
    Juha Salmitaival <juha.salmitaival@aalto.fi>
    Mire Ilomaki <miro.ilomaki@aalto.fi>

Two sessions of an oddball paradigm were performed. During the first session,
the participant was in a calm and relaxed position. During the second session,
the participants was performing different types of movement ( and I don't know
whether they even payed any attention to the oddball paradigm).

This script loads in the data and performs some initial exploratory analysis
using MNE-Python (https://mne.tools).
"""
# Required imports
import mne  # Main MNE-Python analysis package
import pandas as pd  # For loading the .csv files
import numpy as np  # When do you not need numpy?
from scipy.io import loadmat  # For loading .mat files
from scipy.signal import resample, decimate  # Signal resampling

# Set this to where the data is stored on your machine
data_path = './data'


###############################################################################
# Pure EEG analysis of segment 1: oddball paradigm with no movements

# Load the EDF file
raw = mne.io.read_raw_edf(f'{data_path}/Test_alto_2022-06-01_15-03-36_Segment_0.edf', preload=True)

# The channels are named according to the scheme: 'EEG Cz - Cpz'. Strip this
# down to just 'Cz'.
raw.rename_channels(lambda x: x[4:-4])

# Visual inspection of the data showed these two channels to be badly
# connected. Mark them "bad" so they are not taken into account during the
# analysis.
raw.info['bads'] = ['M1', 'T8']

# Initially, the EEG was references against the CPz electrode. This is right
# where we also expect the P300 oddball effect to occur. Let's re-refrence this
# to get a better view of the P300. Normally, I would re-reference to the
# average of the mastoid electrodes M1 and M2, but since M1 is marked "bad",
# let's re-reference to the average of all "good" channels instead.
raw.add_reference_channels('CPz')
raw.set_eeg_reference('average', projection=True)

# Since the channel names follow the standard 10-05 system, we can assign
# default positions for them. Is makes it possible to make topograph plots.
raw.set_montage(mne.channels.make_standard_montage('standard_1005'))


# Apply some bandpass frequency filtering to get rid of slow drifts. Also get
# rid of high frequency stuff (including any 50Hz powerline), since the P300 we
# are after is a slow phenomenon and any high frequency stuff is probably
# irrelevant.
raw.filter(0.5, 40)

# To figure out which stimuli was shown when, load in the Presentation log file.
metadata = pd.read_csv('data\Pilot1-Active Visual Oddball P3.txt', sep='\t')

# For this analysis, we don't really care exactly which stimulus was presented,
# only whether it was a "target" or a "distractor". First find the original
# port codes for all targets and all distractors. Next, mark targets with the
# code "1" and distractors with the code "2".
port_codes = dict()
for port_code, desc in metadata.groupby('Port Code').first()['Target Type'].items():
    port_codes[f' {port_code}'] = 1 if desc == 'Target' else 2
events, _ = mne.events_from_annotations(raw, port_codes)
event_id = dict(target=1, distractor=2)

# Cut epochs around the presentation of each stimulus.
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=1.0, preload=True)

# Annotate the epochs with the information learned from the Presentation log
# file. Just for bookkeeping really. The first 10 entries in the log file were
# practise trials during which the EEG was not running, so skip those.
epochs.metadata = metadata[10:]

# Average epochs together to do event-related potential analysis. Make separate
# ERPs for the "target" and "distractor" conditions.
evokeds = [epochs['target'].average(), epochs['distractor'].average()]

# Plot the ERPs. Layout the channels according to their approximate position on
# the head.
mne.viz.plot_evoked_topo(evokeds, ylim=dict(eeg=(-2, 5)), legend=False)


###############################################################################
# Load EMG signals for segment 1.

# We need to dive pretty deep into the MATLAB # data structure to get at the
# actual data.
external_sensors = (
    loadmat(f'{data_path}/2022-06-01-16-02_EMG11_ECG1.mat')
       ['record_2022_06_01_16_02_EMG11_ECG1']
       ['movements'][0, 0]
       ['sources'][0, 0]
       ['signals'][0, 0]
)

# The first 7 sensors are EMG sensors, sampled at 2000 Hz.
# Again, we need to dive deep into the datastructure to get the actual data
# matrix.
emg = np.vstack([external_sensors[f'signal_{i + 1}'][0, 0]['data'][0, 0].T
                 for i in range(7)])

# The next 5 sensors are sampled at 500 Hz
misc = np.vstack([external_sensors[f'signal_{i + 1}'][0, 0]['data'][0, 0].T
                 for i in range(7, 12)])

# The final channel contains the trigger onsets, sampled at 2000 Hz
triggers = external_sensors['signal_13'][0, 0]['data'][0, 0].T

# Let's print the data dimensions, so we know what we are dealing with.
print('Loaded external sensors. Data dimensions are:')
print('EMG:     ', emg.shape)
print('MISC:    ', misc.shape)
print('Triggers:', triggers.shape)

# The 2nd EMG channel contains NaNs. That could give problems perhaps.
# Let's replace all NaNs with zeros.
emg = np.nan_to_num(emg, nan=0)

# EEG was sampled at 500 Hz. Lets resample the EMG to match. We go from 2000 Hz
# to 500 Hz, so a downsampling of 4 times. The MISC sensors were already
# sampled at 500 Hz, so we can leave them alone. We also won't resample the
# trigger channel yet, as we need the high temporal precision it provides for a
# little longer.
emg = resample(emg, emg.shape[1] // 4, axis=1)
print('Data dimensions after resampling:')
print('EMG:     ', emg.shape)
print('MISC:    ', misc.shape)
print('Triggers:', triggers.shape)

# Use the trigger channel to compute the time offset between the external
# sensors and the EEG. First, compute the onset of the triggers by checking
# where we go from 0 to 1 (the upwards flank).
trigger_onsets = np.where(np.diff(triggers) > 0)[1]

# Looking at the triggers, it looks like the second trigger matches the first EEG event.
first_eeg_marker_time = raw.times[events[0, 0]]
first_emg_marker_time = trigger_onsets[1] / 2000
time_offset = first_emg_marker_time - first_eeg_marker_time
print('Time offset between EEG and external electrodes:', time_offset, 'seconds')
# It looks like the EEG started around 2.2455 seconds after the EMG measurement.

# Now, we can resample the trigger channel so we can add it in with the rest of the data.
#triggers = decimate(triggers, 4)
triggers = resample(triggers, triggers.shape[1] // 4, axis=1)

# Lets cut the external sensor data to match the EEG exactly.
sample_offset = int(time_offset * 500)
emg = emg[:, sample_offset:sample_offset + len(raw)]
misc = misc[:, sample_offset:sample_offset + len(raw)]
triggers = triggers[:, sample_offset:sample_offset + len(raw)]

print('Final data dimensions:')
print('EEG:     ', raw.get_data().shape)
print('EMG:     ', emg.shape)
print('MISC:    ', misc.shape)
print('Triggers:', triggers.shape)

# Pack the data into a MNE-Python continuous data "Raw" object for easy
# manipulation and visualization. First, we need to create an "Info" dictionary
# object that contains information about the sensors.
info = mne.create_info(
    # Channel names
    ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EOG', 'EMG6', 'EMG7',
     'Respiration', 'ECG', 'Respiration Rate', 'Heart Rate', 'R-R Interval',
     'triggers'],

    # Sensor types. Respiration sensor is simply tagged as "bio". Other sensors are "misc".
    ch_types=['emg', 'emg','emg','emg','eog','emg','emg', 'bio', 'ecg', 'misc', 'misc', 'misc', 'stim'],

    # Sampling frequency of 500 Hz
    sfreq=500,
)

# The unit of measurement in MNE-Python is always the SI unit. So, micro-Volts
# need to be converted to Volts.
emg *= 1E-6
misc[:2] *= 1E-6

# Build the "Raw" object
raw_external = mne.io.RawArray(np.vstack((emg, misc, triggers)), info)

# Filter the external sensors to match the filtering of the EEG
raw_external.filter(0.5, 40, picks=['emg', 'ecg', 'eog', 'bio'])

# Now we can concatenate the EEG with the external sensors
raw = raw.add_channels([raw_external], force_update_info=True)

# Plot everything together as one big happy family of sensors. I'm tweaking the
# scalings of the different channels types a bit so it all looks nice.
raw.plot(scalings=dict(eeg=20E-6, ecg=1000E-6, emg=100E-6))


###############################################################################
# ICA analysis on EEG during segment 2. Participant was moving around.

# Load the data in the same manner as we did for segment 1
raw2 = mne.io.read_raw_edf(f'{data_path}/Test_alto_2022-06-01_15-03-36_Segment_1.edf', preload=True)
raw2.rename_channels(lambda x: x[4:-4])
raw2.info['bads'] = ['M1', 'T8']
raw2.add_reference_channels('CPz')
raw2.set_eeg_reference('average', projection=True)
raw2.set_montage(mne.channels.make_standard_montage('standard_1005'))

# For the ICA decomposition, we need to get rid of slow drifts. A highpass
# filter will do nicely.
raw2.filter(1.0, None)

# Do the actual ICA decomposition
ica2 = mne.preprocessing.ICA().fit(raw2)

# Plot the ICA components
ica2.plot_sources(raw2)
