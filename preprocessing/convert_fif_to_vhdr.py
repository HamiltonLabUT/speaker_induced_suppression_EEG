# convert_fif_to_vhdr.py
# written by Garret Kurteff, for the Hamilton Lab
# June 2019
# Only works in Python 3.x due to use of input() over raw_input().

# imports
import philistine as ph
from glob import glob
import numpy as np
import mne
# Paths (change these for local use)
eeg_data_path = '/path/to/dataset/' # downloadable from OSF: https://doi.org/10.17605/OSF.IO/FNRD9

subj = input("Which subject? > ")
blocks = np.sort([b[-2:] for b in glob(f'{eeg_data_path}/{subj}/*') if '_B' in b])

fif_fname = f'{eeg_data_path}{subj}/{subj}_{blocks[0]}/{subj}_{blocks[0]}_ica.fif'
vhdr_fname = f'{eeg_data_path}{subj}/{subj}_{blocks[0]}/{subj}_{blocks[0]}_ica.vhdr'

# Load raw and filter
raw = mne.io.read_raw_fif(fif_fname,preload=True,verbose=False)
raw.filter(l_freq=0.16,h_freq=None)

# Convert and save via philistine
ph.mne.write_raw_brainvision(raw,vhdr_fname)