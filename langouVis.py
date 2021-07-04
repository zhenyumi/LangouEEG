# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os

import numpy as np
from scipy import stats as stats

import mne
from mne.epochs import equalize_epoch_counts
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.datasets import sample
import pickle
from langouEEG import *

import os.path as op
import numpy as np

import mne
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse


# %%
# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')


# %%
subject_name='S03'
dataRoot = '/data/home/viscent/Light'
with open(dataRoot+'/clean_data/'+subject_name+'_clean.lgeeg','rb') as f:
    raw=pickle.load(f)
events, event_dict=extractEvents(raw)
initLayout(raw)
picks = np.arange(64)
epoch_RR,epoch_RF,epoch_4R,epoch_4F = extractEpochs(raw,events,picks)
epoch_4F.set_eeg_reference(projection=True)
epochs = epoch_RR,epoch_RF,epoch_4R,epoch_4F


# %%
noise_cov = mne.compute_covariance(
    epoch_4F, tmax=100, method=['shrunk', 'empirical'], rank=None, verbose=True)

fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw.info)


# %%
fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=1)
print(fwd)
inverse_operator = make_inverse_operator(
    epoch_4F.average().info, fwd, noise_cov, loose=0.2, depth=0.8)
del fwd


# %%
method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2
stc, residual = apply_inverse(epoch_4F.average(), inverse_operator, lambda2,
                              method=method, pick_ori=None,
                              return_residual=True, verbose=True)


# %%
vertno_max, time_max = stc.get_peak(hemi='rh')

subjects_dir = '/data/home/viscent/mne_data/MNE-fsaverage-data'
surfer_kwargs = dict(
    hemi='rh', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral', backend='matplotlib',
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
               font_size=14)
brain.save_movie(..., tmin=0.05, tmax=10, interpolation='linear',
                 time_dilation=20, framerate=10, time_viewer=True)

