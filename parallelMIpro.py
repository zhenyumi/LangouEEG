import os
import numpy as np
import mne
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from copy import deepcopy
from mne.preprocessing import create_ecg_epochs, create_eog_epochs, read_ica
import sys
from langouEEG import *
from tensorpac import Pac, EventRelatedPac, PreferredPhase
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude
import pickle
from tqdm import trange
dataRoot = "/data/home/viscent/Light"


subject_name=4
if subject_name<10:
    subject_name='S0'+str(subject_name)
else:
    subject_name='S'+str(subject_name)
print(subject_name)
with open(dataRoot+'/clean_data/'+subject_name+'_clean.lgeeg','rb') as f:
    raw=pickle.load(f)
events, event_dict=extractEvents(raw)
epochs_4F=[]
epochs=[]
for i in range(64):
    epoch_RR,epoch_RF,epoch_4R,epoch_4F = extractEpochs(raw,events,i)
    epoch_4F_raw=epoch_4F.to_data_frame()
    epoch_4F_np=epoch_4F_raw.values[:,3:6].flatten()
    epochs_4F.append(epoch_4F_np)
    epochs.append([epoch_RR,epoch_RF,epoch_4R,epoch_4F])

p_objs=[]
for i in trange(64):
    for j in trange(64):
        sf = 500
        pha = epochs_4F[i]
        amp = epochs_4F[j]
        p_obj = Pac(idpac=(6, 0, 0), f_pha=np.arange(4,12,0.1), f_amp=np.arange(30,200,2))
        pha_p = p_obj.filter(sf, pha, ftype='phase')
        amp_p = p_obj.filter(sf, amp, ftype='amplitude')
        time_exec = slice(5000, 6000)
        pha_exec, amp_exec = pha_p[..., time_exec], amp_p[..., time_exec]
        pac_exec = p_obj.gpufit(pha_exec, amp_exec).mean(-1)
        p_objs.append(p_obj)
with open(dataRoot+'/'+subject_name+'_pac.lgeeg','wb') as f:
    pickle.dump(p_objs,f)
