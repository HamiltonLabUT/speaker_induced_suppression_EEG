#!/usr/bin/env python


import scipy.io # For .mat files
import h5py # For loading hf5 files
import mne # For loading BrainVision files (EEG)
import numpy as np
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt # For plotting
from matplotlib import cm, rcParams
import random
import itertools as itools
import csv
import logging
import sys

sys.path.append('./path/to/git/speaker_induced_suppression_EEG/preprocessing/')
from utils import ridge, utils

def strf(resp, stim,
	delay_min=0, delay_max=0.6, wt_pad=0.0, alphas=np.hstack((0, np.logspace(-3,5,20))),
	use_corr=True, single_alpha=True, nboots=20, sfreq=128, vResp=[],vStim=[], flip_resp=False,return_pred=False):
	'''
	Run the STRF model.
	* wt_pad: Amount of padding for delays, since edge artifacts can make weights look weird
	* use_corr: Use correlation between predicted and validation set as metric for goodness of fit
	* single_alpha: Use the same alpha value for all electrodes (helps with comparing across sensors)
	* Logspace was previously -1 to 7, changed for smol stim strf in may 20
	'''
	# Populate stim and resp lists (these will be used to get tStim and tResp, or vStim and vResp)
	stim_list = []
	stim_sum= []
	train_or_val = [] # Mark for training or for validation set
	np.random.seed(6655321)
	if flip_resp == True:
		resp = resp.T
		if len(vResp) >= 1:
			vResp = vResp.T
	# Load stimulus and response
	if resp.shape[1] != stim.shape[0]:
		logging.warning("Resp and stim do not match! This is a problem")
	nchans, ntimes = resp.shape
	print(nchans, ntimes)
	# RUN THE STRFS
	# For logging compute times, debug messages
	logging.basicConfig(level=logging.DEBUG)
	delays = np.arange(np.floor((delay_min-wt_pad)*sfreq), np.ceil((delay_max+wt_pad)*sfreq), dtype=int) 
	nalphas = len(alphas)
	all_wts = []
	all_corrs = []
	# Train on 80% of the trials, test on 
	# the remaining 20%.
	# Z-scoring function (assumes time is the 0th dimension)
	resp = zs(resp.T).T
	if len(vResp) >= 1:
		vResp = zs(vResp.T).T
	if len(vResp) == 0 and len(vStim) == 0:
		print("Creating vResp and vStim using an automated 80-20 split...")
		# Create training and validation response matrices.
		# Time must be the 0th dimension.
		tResp = resp[:,:int(0.8*ntimes)].T
		vResp = resp[:,int(0.8*ntimes):].T
		# Create training and validation stimulus matrices
		tStim_temp = stim[:int(0.8*ntimes),:]
		vStim_temp = stim[int(0.8*ntimes):,:]
	else: # if vResp and vStim were passed into the function
		print("Using training/validation split passed into the func...")
		tResp = resp.T
		vResp = vResp.T
		tStim_temp = stim
		vStim_temp = vStim
	tStim = utils.make_delayed(tStim_temp, delays)
	vStim = utils.make_delayed(vStim_temp, delays)
	chunklen = int(len(delays)*4) # We will randomize the data in chunks 
	nchunks = np.floor(0.2*tStim.shape[0]/chunklen).astype(int)
	nchans = tResp.shape[1] # Number of electrodes/sensors

	# get a strf
	print(tStim.shape, vStim.shape)
	print(tResp.shape, vResp.shape)
	wt, corrs, valphas, allRcorrs, valinds, pred, Pstim = ridge.bootstrap_ridge(tStim, tResp, vStim, vResp, 
																		  alphas, nboots, chunklen, nchunks, 
																		  use_corr=use_corr,  single_alpha = single_alpha, 
																		  use_svd=False, corrmin = 0.05,
																		  joined=[np.array(np.arange(nchans))])
	print("wt shape:")
	print(wt.shape)
	
	# If we decide to add some padding to our model to account for edge artifacts, 
	# get rid of it before returning the final strf
	if wt_pad>0:
		print("Reshaping weight matrix to get rid of padding on either side")
		orig_delays = np.arange(np.floor(delay_min*sfreq), np.ceil(delay_max*sfreq), dtype=np.int) 

		# This will be a boolean mask of only the "good" delays (not part of the extra padding)
		good_delays = np.zeros((len(delays), 1), dtype=np.bool)
		int1, int2, good_inds = np.intersect1d(orig_delays,delays,return_indices=True)
		for g in good_inds:
			good_delays[g] = True	#wt2 = wt.reshape((len(delays), -1, wt.shape[1]))[len(np.where(delays<0)[0]):-(len(np.where(delays<0)[0])),:,:]
		print(delays)
		print(orig_delays)
		# Reshape the wt matrix so it is now only including the original delay_min to delay_max time period instead
		# of delay_min-wt_pad to delay_max+wt_pad
		wt2 = wt.reshape((len(delays), -1, wt.shape[1])) # Now this will be ndelays x nfeat x nchans
		wt2 = wt2[good_delays.ravel(), :, :].reshape(-1, wt2.shape[2])
	else:
		wt2 = wt
	print(wt2.shape)
	all_wts.append(wt2)
	all_corrs.append(corrs)
	if return_pred:
		return(all_corrs,all_wts,tStim,tResp,vStim,vResp,valphas,pred)
	else:
		return(all_corrs, all_wts, tStim, tResp, vStim, vResp, valphas)	


def get_feats(model_number='model1',mode='eeg',return_dict=False,extend_labels=False):
	'''
	onsetProd helper function. Returns a list of features given model number.
	'''
	if extend_labels == False:
		task_labels = ['spkr','mic','el','sh']
	else:
		task_labels = ['perception','production','predictable','unpredictable']
	if model_number in ['model8', 'model9','model10', 'model11']:
		if model_number in ['model8', 'model10']: # Manner only
			features_dict = {
				'plosive': ['p','pcl','t','tcl','k','kcl','b','bcl','d','dcl','g','gcl','q'],
				'fricative': ['f','v','th','dh','s','sh','z','zh','hh','hv','ch','jh'],
				'syllabic': ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux'],
				'nasal': ['m','em','n','en','ng','eng','nx'],
				'voiced':   ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux','w','y','el','l','r','dh','z','v','b','bcl','d','dcl','g','gcl','m','em','n','en','eng','ng','nx','q','jh','zh'],
				'obstruent': ['b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx','f', 'g', 'gcl', 'hh', 'hv','jh', 'k', 'kcl', 'p', 'pcl', 'q', 's', 'sh','t', 'tcl', 'th','v','z', 'zh','q'],
				'sonorant': ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux','w','y','el','l','r','m', 'n', 'ng', 'eng', 'nx','en','em'],
			}
		if model_number in ['model9', 'model11']: # Place only
			features_dict = {
				'dorsal': ['y','w','k','kcl', 'g','gcl','eng','ng'],
				'coronal': ['ch','jh','sh','zh','s','z','t','tcl','d','dcl','n','th','dh','l','r'],
				'labial': ['f','v','p','pcl','b','bcl','m','em','w'],
				'high': ['uh','ux','uw','iy','ih','ix','ey','eh','oy'],
				'front': ['iy','ih','ix','ey','eh','ae','ay'],
				'low': ['aa','ao','ah','ax','ae','aw','ay','axr','ow','oy'],
				'back': ['aa','ao','ow','ah','ax','ax-h','uh','ux','uw','axr','aw'],
				'voiced':   ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux','w','y','el','l','r','dh','z','v','b','bcl','d','dcl','g','gcl','m','em','n','en','eng','ng','nx','q','jh','zh'],
			}
	else:
		features_dict = {
			'dorsal': ['y','w','k','kcl', 'g','gcl','eng','ng'],
			'coronal': ['ch','jh','sh','zh','s','z','t','tcl','d','dcl','n','th','dh','l','r'],
			'labial': ['f','v','p','pcl','b','bcl','m','em','w'],
			'high': ['uh','ux','uw','iy','ih','ix','ey','eh','oy'],
			'front': ['iy','ih','ix','ey','eh','ae','ay'],
			'low': ['aa','ao','ah','ax','ae','aw','ay','axr','ow','oy'],
			'back': ['aa','ao','ow','ah','ax','ax-h','uh','ux','uw','axr','aw'],
			'plosive': ['p','pcl','t','tcl','k','kcl','b','bcl','d','dcl','g','gcl','q'],
			'fricative': ['f','v','th','dh','s','sh','z','zh','hh','hv','ch','jh'],
			'syllabic': ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux'],
			'nasal': ['m','em','n','en','ng','eng','nx'],
			'voiced':   ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux','w','y','el','l','r','dh','z','v','b','bcl','d','dcl','g','gcl','m','em','n','en','eng','ng','nx','q','jh','zh'],
			'obstruent': ['b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx','f', 'g', 'gcl', 'hh', 'hv','jh', 'k', 'kcl', 'p', 'pcl', 'q', 's', 'sh','t', 'tcl', 'th','v','z', 'zh','q'],
			'sonorant': ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux','w','y','el','l','r','m', 'n', 'ng', 'eng', 'nx','en','em'],
		}
	features = [f for f in features_dict.keys()]
	if mode == 'ecog':
		emg = False
	elif model_number[-1] == 'e':
		emg = False
		model_number = model_number[:-1]
	else:
		emg = True
	if model_number in ['model1', 'model8', 'model9']:
		y_labels = features + task_labels
	elif model_number in ['model2', 'model10', 'model11']:
		y_labels = features + [f'spkr_{s}' for s in features] + [f'mic_{s}' for s in features] + ['spkr','mic','el','sh']
	elif model_number == 'model3':
		y_labels = features + task_labels[:2]
	elif model_number == 'model4':
		y_labels = features + task_labels[2:]
	elif model_number == 'model5':
		y_labels = features + task_labels + ['spkr_onset','mic_onset']
	elif model_number == 'model6':
		y_labels = y_labels = features + [f'spkr_{s}' for s in features] + [f'mic_{s}' for s in features] + ['spkr','mic','el','sh','spkr_onset','mic_onset']
	elif model_number == 'model7':
		y_labels = features
	elif model_number == 'model12':
		y_labels = task_labels
	elif model_number == 'model13':
		y_labels = features
	if emg == True:
		y_labels.append('emg')
	if return_dict == False:
		return y_labels
	else:
		return features_dict

def load_model_inputs(model_input_h5_fpath,model_number):
	'''
	Loads model from hdf5 and indexes it accordingly for a given model number.
	'''
	all_features = get_feats('model2')
	all_phnfeat = [all_features.index(f) for f in all_features[:-5] if "spkr_" not in f and "mic_" not in f]
	spkr_phnfeat = [all_features.index(f) for f in all_features[:-5] if "spkr_" in f]
	mic_phnfeat = [all_features.index(f) for f in all_features[:-5] if "mic_" in f]
	spkr_mic_task_feats = [all_features.index(f) for f in all_features[-5:] if f in ['spkr','mic']]
	el_sh_task_feats = [all_features.index(f) for f in all_features[-5:] if f in ['el','sh']]
	emg_task_feat = [all_features.index('emg')]
	# Load data
	if os.path.isfile(model_input_h5_fpath):
		with h5py.File(model_input_h5_fpath,'r') as f:
			tStim = np.array(f.get('tStim'))
			tResp = np.array(f.get('tResp'))
			vStim = np.array(f.get('vStim'))
			vResp = np.array(f.get('vResp'))
	else:
		raise Exception(f"File does not exist: {model_input_h5_fpath}")
	# Index specific features according to model number
	if 'model1' in model_number:
		# 18(+1) features: 14 phnfeat + 4 task (+ emg)
		feat_idxs = all_phnfeat + spkr_mic_task_feats + el_sh_task_feats
	if 'model2' in model_number:
		# 46(+1) features: 14*3 phnfeat + 4 task (+ emg)
		feat_idxs = all_phnfeat + spkr_phnfeat + mic_phnfeat + spkr_mic_task_feats + el_sh_task_feats
	if 'model3' in model_number:
		# 16(+1) features: 14 phnfeat + 2 task (+ emg)
		feat_idxs = all_phnfeat + spkr_mic_task_feats
	if 'model4' in model_number:
		# 16(+1) features: 14 phnfeat +2 task (+ emg)
		feat_idxs = all_phnfeat + el_sh_task_feats
	if model_number[-1] != 'e':
		# Add the EMG (if the model includes it)
		feat_idxs = feat_idxs + emg_task_feat
	tStim = tStim[:,feat_idxs]
	vStim = vStim[:,feat_idxs]
	return tStim, tResp, vStim, vResp

def predict_response(wt, vStim, vResp):
	''' 
	Predict the response to [vStim] given STRF weights [wt],
	compare to the actual response [vResp], and return the correlation
	between predicted and actual response.
	Inputs:
	    wt: [features x delays] x electrodes, your STRF weights
	    vStim: time x [features x delays], your delayed stimulus matrix
	    vResp: time x electrodes, your true response to vStim
	Outputs:
	    corr: correlation between predicted and actual response
	    pred: prediction for each electrode [time x electrodes]
	'''
	nchans = wt.shape[1]
	# print('Calculating prediction...')
	pred = np.dot(vStim, wt)

	# print('Calculating correlation')
	corr = np.array([np.corrcoef(vResp[:,i], pred[:,i])[0,1] for i in np.arange(nchans)])

	return corr, pred