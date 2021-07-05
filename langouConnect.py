# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# %%
# get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# 
# # Compute source space connectivity and visualize it using a circular graph
# 
# This example computes the all-to-all connectivity between 68 regions in
# source space based on dSPM inverse solutions and a FreeSurfer cortical
# parcellation. The connectivity is visualized using a circular graph which
# is ordered based on the locations of the regions in the axial plane.
# 

# %%
# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Nicolas P. Rougier (graph code borrowed from his matplotlib gallery)
#
# License: BSD (3-clause)

import numpy as np
import os.path as op
import os
import matplotlib.pyplot as plt
from langouEEG import *

import mne
import pickle
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
import mne
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.minimum_norm import write_inverse_operator
dataRoot = "/data/home/viscent/Light"
# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
result_dir = op.join(dataRoot,'result')
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
print(__doc__)

# %% [markdown]
# ## Load our data
# 
# First we'll load the data we'll use in connectivity estimation. We'll use
# the sample MEG data provided with MNE.
# 
# 

# %%
for subject_name in range(1,21):
    if subject_name<10:
        subject_name='S0'+str(subject_name)
    else:
        subject_name='S'+str(subject_name)
    with open(dataRoot+'/clean_data/'+subject_name+'_clean.lgeeg','rb') as f:
        raw=pickle.load(f)
    raw.set_channel_types({'Trigger':'stim','VEO':'eog'})
    raw.set_eeg_reference(projection=True)
    events, event_dict=extractEvents(raw)
    if not op.exists(os.path.join(dataRoot,'fwd_solutions',subject_name+'_fwd.lgeeg')):
        fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                        bem=bem, eeg=True, mindist=5.0, n_jobs=1)
        print(fwd)
        mne.write_forward_solution(os.path.join(dataRoot,'fwd_solutions',subject_name+'_fwd.lgeeg'),fwd,overwrite=True)
    else:
        fwd = mne.read_forward_solution(os.path.join(dataRoot,'fwd_solutions',subject_name+'_fwd.lgeeg'))


    # %%
    data_path = sample.data_path()
    # subjects_dir = data_path + '/subjects'
    fname_inv = os.path.join(dataRoot,'inv_operators',subject_name+'_inv.lgeeg')
    # subject = 'sample'
    # fname_raw = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    # fname_event = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'


    # Pick MEG channels
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=True,
                        exclude='bads')

    epoch_RR,epoch_RF,epoch_4R,epoch_4F = extractEpochs(raw,events,picks)
    # evoked_4F = epoch_4F.average().pick('eeg')
    # Define epochs for left-auditory condition
    # event_id, tmin, tmax = 1, -0.2, 0.5
    # epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        # baseline=(None, 0), reject=dict(mag=4e-12, grad=4000e-13,
                        #                                 eog=150e-6))


    noise_cov = mne.compute_covariance(
        epoch_4F, tmax=20., method=['shrunk', 'empirical'], rank=None, verbose=True)
    inverse_operator = make_inverse_operator(
        epoch_4F.info, fwd, noise_cov, loose=0.2, depth=0.8)
    if not os.path.exists(fname_inv):  
        write_inverse_operator(os.path.join(dataRoot,'inv_operators',subject_name+'_inv.lgeeg'),inverse_operator)
    else:   
    # Load data
        inverse_operator = read_inverse_operator(fname_inv)
    # raw = mne.io.read_raw_fif(fname_raw)
    # events = mne.read_events(fname_event)

    # %% [markdown]
    # ## Compute inverse solutions and their connectivity
    # 
    # Next, we need to compute the inverse solution for this data. This will return
    # the sources / source activity that we'll use in computing connectivity. We'll
    # compute the connectivity in the alpha band of these sources. We can specify
    # particular frequencies to include in the connectivity with the ``fmin`` and
    # ``fmax`` flags. Notice from the status messages how mne-python:
    # 
    # 1. reads an epoch from the raw file
    # 2. applies SSP and baseline correction
    # 3. computes the inverse to obtain a source estimate
    # 4. averages the source estimate to obtain a time series for each label
    # 5. includes the label time series in the connectivity computation
    # 6. moves to the next epoch.
    # 
    # This behaviour is because we are using generators. Since we only need to
    # operate on the data one epoch at a time, using a generator allows us to
    # compute connectivity in a computationally efficient manner where the amount
    # of memory (RAM) needed is independent from the number of epochs.
    # 
    # 

    # %%
    # Compute inverse solution and for each epoch. By using "return_generator=True"
    # stcs will be a generator object instead of a list.

    snr = 1.0  # use lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
    stcs = apply_inverse_epochs(epoch_4F, inverse_operator, lambda2, method,
                                pick_ori="normal", return_generator=True)

    # Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
    labels = mne.read_labels_from_annot('fsaverage', parc='aparc',
                                        subjects_dir=subjects_dir)[:68]
    label_colors = [label.color for label in labels]

    # Average the source estimates within each label using sign-flips to reduce
    # signal cancellations, also here we return a generator
    src = inverse_operator['src']
    label_ts = mne.extract_label_time_course(
        stcs, labels,  src,allow_empty=False, mode='mean_flip', return_generator=True)

    fmin = 8.
    fmax = 13.
    sfreq = raw.info['sfreq']  # the sampling frequency
    con_methods = ['pli', 'wpli2_debiased', 'ciplv']
    con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        label_ts, method=con_methods, mode='multitaper', sfreq=sfreq, fmin=fmin,
        fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)

    # con is a 3D array, get the connectivity for the first (and only) freq. band
    # for each method
    con_res = dict()
    for method, c in zip(con_methods, con):
        con_res[method] = c[:, :, 0]

    # %% [markdown]
    # ## Make a connectivity plot
    # 
    # Now, we visualize this connectivity using a circular graph layout.
    # 
    # 

    # %%
    # First, we reorder the labels based on their location in the left hemi
    label_names = [label.name for label in labels]

    lh_labels = [name for name in label_names if name.endswith('lh')]

    # Get the y-location of the label
    label_ypos = list()
    for name in lh_labels:
        idx = label_names.index(name)
        ypos = np.mean(labels[idx].pos[:, 1])
        label_ypos.append(ypos)

    # Reorder the labels based on their location
    lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

    # For the right hemi
    rh_labels = [label[:-2] + 'rh' for label in lh_labels]

    # Save the plot order and create a circular layout
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)
    # node_order = node_order[:69]

    node_angles = circular_layout(label_names, node_order, start_pos=90,
                                group_boundaries=[0, len(label_names) / 2])

    # Plot the graph using node colors from the FreeSurfer parcellation. We only
    # show the 300 strongest connections.
    fig,ax=plot_connectivity_circle(con_res['pli'], label_names, n_lines=300,
                            node_angles=node_angles, node_colors=label_colors,
                            title='All-to-All Connectivity 40 Hz '
                                'Condition (PLI)')
    fig.savefig(op.join(result_dir,subject_name+'connectivity.png'))


    # %%
    ax

    # %% [markdown]
    # ## Make two connectivity plots in the same figure
    # 
    # We can also assign these connectivity plots to axes in a figure. Below we'll
    # show the connectivity plot using two different connectivity methods.
    # 
    # 

    # %%
    fig = plt.figure(num=None, figsize=(8, 4), facecolor='black')
    no_names = [''] * len(label_names)
    for ii, method in enumerate(con_methods):
        plot_connectivity_circle(con_res[method], no_names, n_lines=300,
                                node_angles=node_angles, node_colors=label_colors,
                                title=method, padding=0, fontsize_colorbar=6,
                                fig=fig, subplot=(1, 3, ii + 1))

    plt.show()

    # %% [markdown]
    # ## Save the figure (optional)
    # 
    # By default matplotlib does not save using the facecolor, even though this was
    # set when the figure was generated. If not set via savefig, the labels, title,
    # and legend will be cut off from the output png file.
    # 
    # 

    # %%
    # fname_fig = data_path + '/MEG/sample/plot_inverse_connect.png'
    # fig.savefig(fname_fig, facecolor='black')


