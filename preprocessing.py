import os
import numpy as np
import pickle
import mne
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from copy import deepcopy
from mne.preprocessing import create_ecg_epochs, create_eog_epochs, read_ica
import sys
#mne.utils.set_config('MNE_USE_CUDA', 'true')  
from langouEEG import *
from tensorpac import Pac, EventRelatedPac, PreferredPhase
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude
dataRoot = "/data/home/viscent/Light"
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
subject_name=int(rank)+1
if subject_name<10:
    subject_name='S0'+str(subject_name)
else:
    subject_name='S'+str(subject_name)
print(subject_name)
raw,picks,picks_str = initData(subject_name)
initLayout(raw)
events, event_dict=extractEvents(raw)
filterRaw(raw, picks, ref_set_average=True, ref_channels=['M1', 'M2'])
runICA(raw)
with open(dataRoot+'/clean_data/'+subject_name+'_clean.lgeeg','wb') as f:
    pickle.dump(raw,f)