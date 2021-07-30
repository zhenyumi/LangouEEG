# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%


# %% [markdown]
# # Initialize

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
from scipy.stats import *
from IPython.display import clear_output as clear

import mne
import pickle
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity, envelope_correlation
from mne.viz import circular_layout, plot_connectivity_circle
import mne
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.minimum_norm import write_inverse_operator

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import mne

import pyvista
pyvista.start_xvfb()

sample_data_folder = mne.datasets.sample.data_path()
dataRoot = "/data/home/viscent/Light"
# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
isMale = False
isAll = True
isBlind = False
if not isAll:
    result_dir = op.join(dataRoot,'result','male' if isMale else 'female')
else:
    result_dir = op.join(dataRoot,'result','all')
if isBlind:
    result_dir = op.join(result_dir,'Blind')
subjects_dir = op.dirname(fs_dir)
if not op.exists(result_dir):
    os.mkdir(result_dir)
# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
print(__doc__)

# %% [markdown]
# # Load data

# %%
# %%capture
epochs_4F = dict()
epochs_RF = dict()
epochs_4R = dict()
epochs_RR = dict()
for subject_name in range(1,21):
    if not isAll:
        if not (isMale ^ (subject_name in [7,8,11,17])):
            continue
    if subject_name<10:
        subject_name='S0'+str(subject_name)
    else:
        subject_name='S'+str(subject_name)
    with open(dataRoot+'/clean_data_av/'+subject_name+'_clean.lgeeg','rb') as f:
        raw=pickle.load(f)
    events, event_dict=extractEvents(raw)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=True,
                        exclude='bads')
    if isBlind:
        epoch_RR,epoch_RF,epoch_4R,epoch_4F = extractEpochsBlind(raw,events,picks)
    else:
        epoch_RR,epoch_RF,epoch_4R,epoch_4F = extractEpochs(raw,events,picks)
    epochs_4F[subject_name]=epoch_4F
    epochs_RF[subject_name]=epoch_RF
    epochs_RR[subject_name]=epoch_RR
    epochs_4R[subject_name]=epoch_4R
    if not op.exists(os.path.join(dataRoot,'fwd_solution.lgeeg')):
        fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                        bem=bem, eeg=True, mindist=5.0, n_jobs=1)
        print(fwd)
        mne.write_forward_solution(os.path.join(dataRoot,'fwd_solution.lgeeg'),fwd,overwrite=True)
    else:
        fwd = mne.read_forward_solution(os.path.join(dataRoot,'fwd_solution.lgeeg'))
    data_path = sample.data_path()

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=True,
                        exclude='bads')

 


# %% [markdown]
# # Source Estimation
# 

# %%
# %%capture
cons_40 = []
cons_rand = []
act_40 = []
act_40_paired = []
act_rand = []
act_rand_paired = []
avg_stc_40 = None
avg_stc_rand = None
result_root= result_dir
fname_avg_stc_4F = op.join(result_dir,'avg_stc_4F.lgeeg')
fname_avg_stc_RF = op.join(result_dir,'avg_stc_RF.lgeeg')
for subject_name,_ in epochs_4F.items():
    # Compute inverse solution and for each epoch. By using "return_generator=True"
    # stcs will be a generator object instead of a list.
    result_dir = op.join(result_root,subject_name)
    epoch_4F = epochs_4F[subject_name]
    epoch_RF = epochs_RF[subject_name]
    fname_stc_4F = op.join(result_dir,subject_name+'_stc_4F.lgeeg')
    fname_stc_RF = op.join(result_dir,subject_name+'_stc_RF.lgeeg')
    fname_tl_4F = op.join(result_dir,subject_name+'_tl_4F.lgeeg')
    fname_tl_RF = op.join(result_dir,subject_name+'_tl_RF.lgeeg')
    fname_inv = os.path.join(dataRoot,'inv_operators',subject_name+'_inv.lgeeg')
    

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)



    fname_inv_r = os.path.join(dataRoot,'inv_operators',subject_name+'_r_inv.lgeeg')
    fname_cov_r = os.path.join(dataRoot,'noise_covariance',subject_name+'_r_cov.lgeeg')
    fname_inv_4 = os.path.join(dataRoot,'inv_operators',subject_name+'_4_inv.lgeeg')
    fname_cov_4 = os.path.join(dataRoot,'noise_covariance',subject_name+'_4_cov.lgeeg')



    if not os.path.exists(fname_cov_4):  
        noise_cov = mne.compute_covariance(
            epochs_4R[subject_name], tmax=80., method=['shrunk', 'empirical'], rank=None, verbose=True)
        mne.write_cov(fname_cov_4,noise_cov)
    else:   
    # Load data
        noise_cov = mne.read_cov(fname_cov_4)
    
    if not os.path.exists(fname_inv_4):  
        inverse_operator = make_inverse_operator(
            raw.info, fwd, noise_cov, loose=0.2, depth=0.8)
        write_inverse_operator(fname_inv_4,inverse_operator)
    inverse_operator = read_inverse_operator(fname_inv_4)



    snr = 1.0  # use lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
    # Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
    labels = mne.read_labels_from_annot('fsaverage', parc='aparc',
                                        subjects_dir=subjects_dir)[:68]
    label_colors = [label.color for label in labels]
    # Average the source estimates within each label using sign-flips to reduce
    # signal cancellations, also here we return a generator
    src = inverse_operator['src']
#=====STC======

    if op.exists(fname_stc_4F):
        with open(fname_stc_4F,'rb') as f :
            stcs = pickle.load(f)
        print(subject_name+' loaded')
    else:
        stcs = apply_inverse_epochs(epoch_4F, inverse_operator, lambda2, method,
                                    pick_ori="normal", return_generator=False)
        with open(fname_stc_4F,'wb') as f:
            pickle.dump(stcs,f)
    if not op.exists(fname_avg_stc_4F):
        if avg_stc_40 is None:
            avg_stc_40 = np.mean(stcs)
        else:
            avg_stc_40 = np.mean([np.mean(stcs),avg_stc_40])
#======Time Label=====
    if op.exists(fname_tl_4F):
        with open(fname_tl_4F,'rb') as f:
            label_ts = pickle.load(f)
        print(subject_name+' loaded')

    else:
        
        label_ts = mne.extract_label_time_course(
            stcs, labels,  src,allow_empty=False, mode='mean_flip', return_generator=False)
        with open(fname_tl_4F,'wb') as f:
            pickle.dump(label_ts,f)
    fmin = 8.
    fmax = 13.
    sfreq = raw.info['sfreq']  # the sampling frequency
    con_methods = ['pli', 'wpli2_debiased', 'ciplv']
    if not os.path.exists(op.join(result_dir,'cons')):
        os.mkdir(op.join(result_dir,'cons'))
    if not op.exists(op.join(result_dir,'cons','40_con.lgeeg')):
        # con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        #     label_ts, method=con_methods, mode='multitaper', sfreq=sfreq, fmin=fmin,
        #     fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)
        print("computing correlation")
        con = envelope_correlation(
            label_ts,combine = None,absolute=True,verbose=True)
        with open(op.join(result_dir,'cons','40_con.lgeeg'),'wb') as f:
            pickle.dump(con,f)
    else:
        with open(op.join(result_dir,'cons','40_con.lgeeg'),'rb') as f:
            con = pickle.load(f)
    label_names = [label.name for label in labels]
    # con = np.squeeze(con)
    con = np.mean(con,axis=0)
    cons_40.append(con)
    pd_40=pd.DataFrame(con)
    pd_40.columns = label_names
    pd_40.index = label_names
    pd_40.to_excel(op.join(result_dir,'40_conn.xlsx'))
    con_40 = con.copy()
    pd_40=pd.DataFrame(np.mean(label_ts,axis=2))
    pd_40.columns = label_names
    pd_40.to_excel(op.join(result_dir,'40_activation.xlsx'))
    act_40.append(np.mean(np.array(label_ts),axis=2))
    act_40_paired.append(np.mean(np.mean(np.array(label_ts),axis=2),axis=0))

    if not os.path.exists(fname_cov_r):  
        noise_cov = mne.compute_covariance(
            epochs_RR[subject_name], tmax=80., method=['shrunk', 'empirical'], rank=None, verbose=True)
        mne.write_cov(fname_cov_r,noise_cov)
    else:   
    # Load data
        noise_cov = mne.read_cov(fname_cov_r)
    
    if not os.path.exists(fname_inv_r):  
        inverse_operator = make_inverse_operator(
            raw.info, fwd, noise_cov, loose=0.2, depth=0.8)
        write_inverse_operator(fname_inv_r,inverse_operator)
    inverse_operator = read_inverse_operator(fname_inv_r)


    lambda2 = 1.0 / snr ** 2
    method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)



    #=====STC======

    if op.exists(fname_stc_RF):
        with open(fname_stc_RF,'rb') as f :
            stcs = pickle.load(f)
        print(subject_name+" loaded")
    else:
        stcs = apply_inverse_epochs(epoch_RF, inverse_operator, lambda2, method,
                                    pick_ori="normal", return_generator=False)
        with open(fname_stc_RF,'wb') as f:
            pickle.dump(stcs,f)
    if not op.exists(fname_avg_stc_RF):
        if avg_stc_rand is None:
            avg_stc_rand = np.mean(stcs)
        else:
            avg_stc_rand = np.mean([np.mean(stcs),avg_stc_rand])
#======Time Label=====
    if op.exists(fname_tl_RF):
        with open(fname_tl_RF,'rb') as f:
            label_ts = pickle.load(f)
        print(subject_name+" loaded")
    else:
        label_ts = mne.extract_label_time_course(
            stcs, labels,  src,allow_empty=False, mode='mean_flip', return_generator=False)
        with open(fname_tl_RF,'wb') as f:
            pickle.dump(label_ts,f)

    
    if not op.exists(op.join(result_dir,'cons','rand_con.lgeeg')):
        # con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        #     label_ts, method=con_methods, mode='multitaper', sfreq=sfreq, fmin=fmin,
        #     fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)
        print("computing correlation")
        con = envelope_correlation(
            label_ts,combine = None, absolute=True,verbose=True)
        with open(op.join(result_dir,'cons','rand_con.lgeeg'),'wb') as f:
            pickle.dump(con,f)
    else:
        print(subject_name+" loaded")
        with open(op.join(result_dir,'cons','rand_con.lgeeg'),'rb') as f:
            con = pickle.load(f)
    # con is a 3D array, get the connectivity for the first (and only) freq. band
    # for each method

    # con = np.squeeze(con)
    con = np.mean(con,axis=0)
    cons_rand.append(con)
    con_rand = con.copy()
    pd_rand=pd.DataFrame(con)
    pd_rand.columns = label_names
    pd_rand.index = label_names
    pd_rand.to_excel(op.join(result_dir,'rand_conn.xlsx'))
    pd_delta=pd.DataFrame(con_40-con_rand)
    pd_delta.columns = label_names
    pd_delta.index = label_names
    pd_delta.to_excel(op.join(result_dir,'delta_conn.xlsx'))
    pd_rand=pd.DataFrame(np.mean(label_ts,axis=2))
    pd_rand.columns = label_names
    pd_rand.to_excel(op.join(result_dir,'rand_activation.xlsx'))
    act_rand.append(np.mean(np.array(label_ts),axis=2))
    act_rand_paired.append(np.mean(np.mean(np.array(label_ts),axis=2),axis=0))
if not op.exists(fname_avg_stc_4F):
    with open(fname_avg_stc_4F,'wb') as f:
        pickle.dump(avg_stc_40,f)
else:
    with open(fname_avg_stc_4F,'rb') as f:
        avg_stc_40 = pickle.load(f)
if not op.exists(fname_avg_stc_RF):
    with open(fname_avg_stc_RF,'wb') as f:
        pickle.dump(avg_stc_rand,f)
else:
    with open(fname_avg_stc_RF,'rb') as f:
        avg_stc_rand = pickle.load(f)
cons_40 = np.array(cons_40)
cons_rand = np.array(cons_rand)
act_40 = np.vstack(act_40)
act_rand = np.vstack(act_rand)
act_40_paired = np.array(act_40_paired)
act_rand_paired = np.array(act_rand_paired)


# %%
subjects_dir = data_path + '/subjects'
vertno_max, time_max = avg_stc_40.get_peak(hemi='rh')
surfer_kwargs = dict(
    hemi='both', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)
brain = avg_stc_40.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
               font_size=14)


# %%
