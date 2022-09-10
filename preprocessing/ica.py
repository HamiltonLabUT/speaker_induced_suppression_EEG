# ica.py
# written by Garret Kurteff, for the Hamilton Lab
# June 2019
# Only works in Python 3.x due to use of input() over raw_input().

# imports
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet as tfr_morlet
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import mne
from glob import glob

# Paths (change these for local use)
eeg_data_path = '/path/to/dataset/' # downloadable from OSF: https://doi.org/10.17605/OSF.IO/FNRD9

subj = input("Which subject? > ")

# Load data
blocks = np.sort([b[-2:] for b in glob(f'{eeg_data_path}/{subj}/*') if '_B' in b])
if len(blocks) == 1:
	raw_fpath = f'{eeg_data_path}{subj}/{subj}_{blocks[0]}/{subj}_{blocks[0]}_downsampled.vhdr'
	raw = mne.io.read_raw_brainvision(raw_fpath,preload=True,verbose=False)
elif len(blocks) > 1:
	raws = []
	for b in blocks:
		raw_fpath = f'{eeg_data_path}{subj}/{subj}_{b}/{subj}_{b}_downsampled.vhdr'
		raws.append(mne.io.read_raw_brainvision(raw_fpath,preload=True,verbose=False))
	raw = mne.concatenate_raws(raws,preload=True,verbose=False)
else:
	raise Exception('Could not find any blocks of data. Check your path (line 17 of script)')

# Apply montage
pos = raw.get_montage().get_positions()
biosem=mne.channels.make_standard_montage('biosemi64')
new_montage = mne.channels.make_dig_montage(
    ch_pos=pos['ch_pos'],
    nasion=biosem.get_positions()['nasion'],
    lpa=biosem.get_positions()['lpa'],
    rpa=biosem.get_positions()['rpa'],
    hsp=None,
    hpi=None,
    coord_frame='head',
)
raw.set_montage(new_montage);

# Apply linked mastoid reference (i.e., TP9 & TP10)
if subj == 'OP0017': # TP9 is a bad ch for OP0017 so we interpolate first
	raw.info['bads'] = ['TP9']
	raw.interpolate_bads(reset_bads=False)
raw.set_eeg_reference(['TP9','TP10'],verbose=False)

# Plot PSD
picks = mne.pick.types(raw.info,eeg=True,meg=False,eog=False)
raw.plot_psd(picks=picks)

# Filter the data (notch and bandpass)
raw.notch_filter(60)
raw.filter(l_freq=1,h_freq=30)

# Annotate raw data, rejecting bad segments and channels
raw.plot()

# Save annotations
raw.save(f'{eeg_data_path}{subj}/{subj}_{blocks[0]}/{subj}_{blocks[0]}_downsampled_with_annotations.fif',overwrite=True)

# Fit ICA on the EEG channels
ica = ICA(n_components=len(picks),method='infomax')
ica.fit(raw,picks=picks,reject_by_annotation=True)

# Find epochs where EOG is large
if 'vEOG' in raw.ch_names:
	eog_epochs = create_eog_epochs(raw, ch_name='vEOG', tmin=-0.5, tmax=0.5, l_freq=1, h_freq=10) 
	if len(eog_epochs) > 0:
		eog_inds, scores = ica.find_bads_eog(eog_epochs, ch_name='vEOG', threshold=3.0)
		if len(eog_inds) > 0:
			ica.plot_properties(eog_epochs, picks=eog_inds)

# Inspect components. Reject artifact components here
ica.plot_components()
ica.plot_sources(raw)
ica.plot_properties(raw)

# Apply ICA to the data
# First, reload the raw data -- we are not applying ICA to bandpassed data because CCA requires high frequency info
if len(blocks) == 1:
	raw_fpath = f'{eeg_data_path}{subj}/{subj}_{blocks[0]}/{subj}_{blocks[0]}_downsampled.vhdr'
	raw = mne.io.read_raw_brainvision(raw_fpath,preload=True,verbose=False)
elif len(blocks) > 1:
	raws = []
	for b in blocks:
		raw_fpath = f'{eeg_data_path}{subj}/{subj}_{b}/{subj}_{b}_downsampled.vhdr'
		raws.append(mne.io.read_raw_brainvision(raw_fpath,preload=True,verbose=False))
	raw = mne.concatenate_raws(raws,preload=True,verbose=False)
if subj == 'OP0017':
	raw.info['bads'] = ['TP9']
	raw.interpolate_bads(reset_bads=False)
raw.set_eeg_reference(['TP9','TP10'],verbose=False)
raw.notch_filter(60)
ica.apply(raw)

# Filter for viewing corrected data
viz_raw = raw.copy()
viz_raw.filter(l_freq=1,h_freq=15)
viz_raw.plot()

# Lastly, save ICA'd data to file
raw.save(f'{eeg_data_path}{subj}/{subj}_{blocks[0]}/{subj}_{blocks[0]}_ica.fif',overwrite=True)
