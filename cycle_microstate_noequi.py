## Generally setup
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
from langouEEG import *
from scipy.stats import *
from eeg_microstates3 import *
import pandas as pd
import time
import mne
import pickle
from mne.datasets import sample
from mne.datasets import fetch_fsaverage
from mpl_toolkits.mplot3d import Axes3D  # noqa
# from langouMicrostates import *
from IPython.display import clear_output as clear
import logging
# import easyEEG

for i in range(0,6):
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

    ## Input EEG data and extarct epochs

    epochs_4F = dict()
    epochs_RF = dict()
    epochs_4R = dict()
    epochs_RR = dict()

    tmin_rest, tmax_rest, tmin_flick,tmax_flick = 40, 60, 10, 30
    tmin_R_flick, tmax_R_flick = 20, 25

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
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                            exclude='bads')
        if isBlind:
            epoch_RR,epoch_RF,epoch_4R,epoch_4F = extractEpochsBlind(raw,events,picks, 
            tmin_rest = 60,tmax_rest = 87,tmin_flick = 3,tmax_flick = 30)
        else:
            epoch_RR,epoch_RF,epoch_4R,epoch_4F = extractEpochs_id(raw,events,picks, 
            tmin_rest = tmin_rest,tmax_rest = tmax_rest,tmin_flick = tmin_flick,tmax_flick = tmax_flick)
            _,epoch_RF,_,_ = extractEpochs_id(raw,events,picks, 
            tmin_rest = tmin_rest,tmax_rest = tmax_rest,tmin_flick = tmin_R_flick,tmax_flick = tmax_R_flick)
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

        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                            exclude='bads')
        # Clear output display
        clear()

    ## Concact epochs
    epoch_4F = mne.concatenate_epochs(list(epochs_4F.values()))
    epoch_RF = mne.concatenate_epochs(list(epochs_RF.values()))
    epoch_4R = mne.concatenate_epochs(list(epochs_4R.values()))
    # mne.epochs.equalize_epoch_counts([epoch_4F, epoch_RF, epoch_RR])

    # Clear output display
    clear()

    result_dir = "/data/home/viscent/Light/result/microstates"
    tm = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    time_augs=[['min_rest','max_rest','min_flick','max_flick','min_random_flick','max_random_flick']
    ,[tmin_rest, tmax_rest, tmin_flick, tmax_flick, tmin_R_flick, tmax_R_flick]]
    to_save_cache = False

    """ vars_4R = display_maps(epoch_4R, n_maps=2, save=True, dpi=600, filename='40Hz_rest',
    fmt='.png', to_save_cache=to_save_cache, time_augs=time_augs, tm=tm, result_dir=result_dir,
    calc_lzc=True, epochs=epochs_4R,save_log=True) """
    save_logs(epoch_4R, tm=tm, n_maps=20, filename='40Hz_rest',result_dir=result_dir, save_time=True, save_p=False, save_t=False, save_state=True, save_GEV=True, save_RTT=False)

    """ vars_4F = display_maps(epoch_4F, n_maps=2, save=True, dpi=600, filename='40Hz_flicker', 
    fmt='.png', to_save_cache=to_save_cache, time_augs=time_augs, tm=tm, result_dir=result_dir,
    calc_lzc=True, epochs=epochs_4F,save_log=True) """
    save_logs(epoch_4F, tm=tm, n_maps=20, filename='40Hz_flicker',result_dir=result_dir, save_time=True, save_p=False, save_t=False, save_state=True, save_GEV=True, save_RTT=False)