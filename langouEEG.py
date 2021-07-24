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
from tensorpac import Pac, EventRelatedPac, PreferredPhase
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude
from scipy.integrate import simps
#mne.utils.set_config('MNE_USE_CUDA', 'true')  
dataRoot = "/data/home/viscent/Light"
# dataRoot = sys.path[0]

def init_prog():
    global ratio_TD_all_r, ratio_TU_all_r, ratio_DU_all_r
    ratio_TD_all_r, ratio_TU_all_r, ratio_DU_all_r = [],[],[]

    global ratio_TD_all_f, ratio_TU_all_f, ratio_DU_all_f
    ratio_TD_all_f, ratio_TU_all_f, ratio_DU_all_f = [],[],[]
    
    global ratioMA_TD_all_r, ratioMA_TU_all_r, ratioMA_DU_all_r
    ratioMA_TD_all_r, ratioMA_TU_all_r, ratioMA_DU_all_r = [],[],[]
    
    global ratioMA_TD_all_f, ratioMA_TU_all_f, ratioMA_DU_all_f
    ratioMA_TD_all_f, ratioMA_TU_all_f, ratioMA_DU_all_f = [],[],[]
    
    global rel_power_R, rel_power_F, subj_number_R, subj_number_F, rel_power_R_ran, rel_power_F_ran
    rel_power_R, rel_power_F, subj_number_R, subj_number_F, rel_power_R_ran, rel_power_F_ran = [],[],[],[],[],[]
    
    return
def save_ratios(filefolder = dataRoot + '/Light'):
    global ratio_TD_all_r, ratio_TU_all_r, ratio_DU_all_r
    global ratio_TD_all_f, ratio_TU_all_f, ratio_DU_all_f
    global ratioMA_TD_all_r, ratioMA_TU_all_r, ratioMA_DU_all_r
    global ratioMA_TD_all_f, ratioMA_TU_all_f, ratioMA_DU_all_f
    
    df_r = pd.DataFrame()
    df_r['ratio_TD'] = ratio_TD_all_r
    df_r['ratio_TU'] = ratio_TU_all_r
    df_r['ratio_DU'] = ratio_DU_all_r
    df_r.to_csv(filefolder + '/ratios_rest.csv')
    df_f = pd.DataFrame()
    df_f['ratio_TD'] = ratio_TD_all_f
    df_f['ratio_TU'] = ratio_TU_all_f
    df_f['ratio_DU'] = ratio_DU_all_f
    df_f.to_csv(filefolder + '/ratios_flicker.csv')
    dfMA_r = pd.DataFrame()
    dfMA_r['ratio_TD'] = ratioMA_TD_all_r
    dfMA_r['ratio_TU'] = ratioMA_TU_all_r
    dfMA_r['ratio_DU'] = ratioMA_DU_all_r
    dfMA_r.to_csv(filefolder + '/ratiosMA_rest.csv')
    dfMA_f = pd.DataFrame()
    dfMA_f['ratio_TD'] = ratioMA_TD_all_f
    dfMA_f['ratio_TU'] = ratioMA_TU_all_f
    dfMA_f['ratio_DU'] = ratioMA_DU_all_f
    dfMA_f.to_csv(filefolder + '/ratiosMA_flicker.csv')
    return
def csv_transformat(filefolder = dataRoot + '/Light', type='flicker'):
    filefolder = dataRoot + '/Light'
    df=pd.read_csv(filefolder + '/ratios_{0}.csv'.format(type))
    shape = df.shape
    print(shape)
    rows = shape[0]
    column = shape[1]
    col_names = df.columns.values.tolist()
    ratios = []
    labels = []
    print(col_names)
    for i in range(1, column):
        ratio = df.iloc[0:rows,i].tolist()
        type_label = [col_names[i]] * len(ratio)
        ratios += ratio
        labels += type_label
    save = pd.DataFrame()
    save['ratios'] = ratios
    save['labels'] = labels
    save.to_csv(filefolder + '/ratios_{0}_all.csv'.format(type))
    print('{0} CSV transformation complete'.format(type))
    return
def csv_transformat_MA(filefolder = dataRoot + '/Light', type='flicker'):
    filefolder = dataRoot + '/Light'
    df=pd.read_csv(filefolder + '/ratiosMA_{0}.csv'.format(type))
    shape = df.shape
    print(shape)
    rows = shape[0]
    column = shape[1]
    col_names = df.columns.values.tolist()
    ratios = []
    labels = []
    print(col_names)
    for i in range(1, column):
        ratio = df.iloc[0:rows,i].tolist()
        type_label = [col_names[i]] * len(ratio)
        ratios += ratio
        labels += type_label
    save = pd.DataFrame()
    save['ratios'] = ratios
    save['labels'] = labels
    save.to_csv(filefolder + '/ratiosMA_{0}_all.csv'.format(type))
    print('{0} CSV transformation complete'.format(type))
    return
def initData(subject_name,picks_str=['O1','O2','OZ']):
    file_path = dataRoot+'/Light/'+subject_name + " Data.cnt"
    print(dataRoot)
    print(file_path)
    raw = mne.io.read_raw_cnt(file_path, preload=True)
    # picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=True, exclude='bads')
    picks = mne.pick_channels(raw.info["ch_names"], picks_str)  
    raw = raw.set_channel_types({'Trigger':'stim','VEO':'eog'})
    if not os.path.exists(dataRoot+('/Light')):
        os.mkdir('Light')
    return raw,picks,picks_str
def initData_clean(subject_name,picks_str=['O1','O2','OZ']):
    file_path = dataRoot+'/Light/'+subject_name + " Data.cnt"
    print(dataRoot)
    print(file_path)
    raw = mne.io.read_raw_cnt(file_path, preload=True)
    # picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=True, exclude='bads')
    picks = picks_str
    if not os.path.exists(dataRoot+('/Light')):
        os.mkdir('Light')
    return raw,picks,picks_str
def initLayout(raw):
    # layout = pd.read_csv(dataRoot + '/channel_dict.txt', sep = '\t')
    # layout.columns = layout.columns.str.strip()
    # layout["labels"] = layout["labels"].str.strip()
    # layout = layout.set_index('labels')
    # layout = layout.to_dict(orient = "index")
    # for channel in layout.keys():
    #     yxz = np.array([layout[channel]["Y"], layout[channel]["X"], layout[channel]["Z"]])
    #     layout[channel] = yxz
    # layout = mne.channels.make_dig_montage(layout, coord_frame='head')
    # mne.viz.plot_montage(layout)
    # raw.set_montage(layout)
    quikMontage = mne.channels.read_custom_montage(os.path.join(dataRoot,'LangouEEG','quikCap.elc'))
    raw = raw.copy().set_montage(quikMontage)
    return raw
def extractEvents(raw):
# cnt file describe
    # print("file info:")
    # print(raw.info)
    # print("channel names:")
    # print(raw.info["ch_names"])
    # print("time period:")
    # print(raw.n_times)
    # #print("time points:")
    # #print(raw.times)
    # print("events:")
    events, event_dict = mne.events_from_annotations(raw)
    # event_dict = {'random_flicker-60s':1, 'random_rest-300s':2, '40Hz_rest-300s':3, '40Hz_flicker-60s':4}
    print(event_dict)
    return events, event_dict
def filterRaw(raw,picks, ref_set_average=False, ref_channels=['M1', 'M2']):
    raw = raw.filter(0.1, None, fir_design='firwin')
    if ref_set_average: 
        # 可以考虑使用所有通道的平均值作为参考
        raw = raw.copy().set_eeg_reference(ref_channels='average', projection=True)
    else:
        # 使用特定的参考电极
        raw = raw.copy().set_eeg_reference(ref_channels=ref_channels, projection=True)
    raw = raw.apply_proj()
    raw = raw.notch_filter(freqs=50,method='spectrum_fit')
    # raw.plot_psd(area_mode='range', tmax=10.0, picks=picks, average=False)
    return raw
def dbgPlot(raw):
    # Plot raw data
    img_raw_psd = raw.plot_psd()
    # Print
    scale = dict(mag=1e-12, grad=4e-11, eeg=128e-6, eog=150e-6, ecg=5e-4,
         emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
         resp=1, chpi=1e-4, whitened=1e2)
    # Set fig size
    img_raw_plot = raw.plot(duration = 40, n_channels=65, scalings=scale,start=288)
    img_raw_plot.set_size_inches([20,20])
    # img_raw_plot.savefig(dataRoot + '/img_raw_plot3.png', dpi=300)
def runICA(raw):
# set up and fit the ICA
    ica = mne.preprocessing.ICA(n_components=20, random_state=0)
    ica.fit(raw)
    # ica.plot_components()
    bad_ica = ica.detect_artifacts(raw).exclude
    raw = ica.apply(raw.copy(), exclude=bad_ica)
    return raw
def extractEpochsBlind(raw,events,picks,tmin_rest = -20,tmax_rest = -10,tmin_flick = 3,tmax_flick = 30):
# Get epoch for each event
    custom_event_ids = {'LB':5, 'RB':7,'4F':9,'RF':12,'4R':8,'RR':11}
    events, event_dict = mne.events_from_annotations(raw)
    print(event_dict)
    tmin_rest = tmin_rest
    tmax_rest = tmax_rest
    tmin_flick = tmin_flick
    tmax_flick = tmax_flick
    reject=dict()
    ## Epoch: Random flicker
    event_id = custom_event_ids['RB']
    event_id = event_dict[str(event_id)]
    tmin = tmin_flick
    tmax = tmax_flick
    epoch_RF = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks,reject=reject, baseline=(tmin_flick, tmin_flick), preload=True)
    evoked_RF = epoch_RF.average()
    #evoked_RF.plot(time_unit='s')
    ## Epoch: Random rest
    event_id = custom_event_ids['RB']
    event_id = event_dict[str(event_id)]
    tmin = tmin_rest
    tmax = tmax_rest
    epoch_RR = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks,reject=reject, baseline=(tmin_rest, tmin_rest), preload=True)
    evoked_RR = epoch_RR.average()
    #evoked_RR.plot(time_unit='s')
    ## Epoch: 40 Hz rest
    event_id = custom_event_ids['LB']
    event_id = event_dict[str(event_id)]
    tmin = tmin_rest
    tmax = tmax_rest
    epoch_4R = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks,reject=reject, baseline=(tmin_rest, tmin_rest), preload=True)
    #epoch_4R.drop([0,1])
    evoked_4R = epoch_4R.average()
    #evoked_4R.plot(time_unit='s')
    ## Epoch: 40 Hz flicker
    event_id = custom_event_ids['LB']
    event_id = event_dict[str(event_id)]
    tmin = tmin_flick
    tmax = tmax_flick
    epoch_4F = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks,reject=reject,baseline=(tmin_flick, tmin_flick), preload=True)
    epoch_4F_all = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        baseline=(tmin_flick, tmin_flick), preload=True, 
                        reject=dict())
    #epoch_4F.drop([0,1])
    #epoch_4F_all.drop([0,1])
    evoked_4F = epoch_4F.average()
    #evoked_4F.plot(time_unit='s')
    return epoch_RR,epoch_RF,epoch_4R,epoch_4F
def extractEpochs(raw,events,picks,tmin_rest = -20,tmax_rest = -10,tmin_flick = 10,tmax_flick = 20):
# Get epoch for each event
    custom_event_ids = {'LB':5, 'RB':7,'4F':9,'RF':12,'4R':8,'RR':11}
    tmin_rest = tmin_rest
    tmax_rest = tmax_rest
    tmin_flick = tmin_flick
    tmax_flick = tmax_flick
    reject=dict()
    ## Epoch: Random flicker
    events, event_dict = mne.events_from_annotations(raw)
    event_id = event_dict[str(custom_event_ids['RF'])]
    tmin = tmin_flick
    tmax = tmax_flick
    epoch_RF = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks,reject=reject, baseline=(tmin_flick, tmin_flick), preload=True)
    evoked_RF = epoch_RF.average()
    #evoked_RF.plot(time_unit='s')
    ## Epoch: Random rest
    event_id = event_dict[str(custom_event_ids['RF'])]
    tmin = tmin_rest
    tmax = tmax_rest
    epoch_RR = mne.Epochs(raw, events, event_id, tmin_rest, tmax_rest, proj=True,
                        picks=picks,reject=reject, baseline=None, preload=True)
    evoked_RR = epoch_RR.average()
    #evoked_RR.plot(time_unit='s')
    ## Epoch: 40 Hz rest
    event_id = event_dict[str(custom_event_ids['4F'])]
    tmin = tmin_rest
    tmax = tmax_rest
    epoch_4R = mne.Epochs(raw, events, event_id, tmin_rest, tmax_rest, proj=True,
                        picks=picks,reject=reject, baseline=None, preload=True)
    #epoch_4R.drop([0,1])
    evoked_4R = epoch_4R.average()
    #evoked_4R.plot(time_unit='s')
    ## Epoch: 40 Hz flicker
    event_id = event_dict[str(custom_event_ids['4F'])]
    tmin = tmin_flick
    tmax = tmax_flick
    epoch_4F = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks,reject=reject,baseline=(tmin_flick, tmin_flick), preload=True)
    epoch_4F_all = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        baseline=(tmin_flick, tmin_flick), preload=True, 
                        reject=dict())
    #epoch_4F.drop([0,1])
    #epoch_4F_all.drop([0,1])
    evoked_4F = epoch_4F.average()
    #evoked_4F.plot(time_unit='s')
    return epoch_RR,epoch_RF,epoch_4R,epoch_4F

def extractEpochs_id(raw,events,picks,tmin_rest = 60,tmax_rest = 120,tmin_flick = 10,tmax_flick = 20):
# Get epoch for each event
    custom_event_ids = {'LB':5, 'RB':7,'4F':9,'RF':12,'4R':8,'RR':11}
    tmin_rest = tmin_rest
    tmax_rest = tmax_rest
    tmin_flick = tmin_flick
    tmax_flick = tmax_flick
    reject=dict()
    ## Epoch: Random flicker
    events, event_dict = mne.events_from_annotations(raw)
    event_id = event_dict[str(custom_event_ids['RF'])]
    tmin = tmin_flick
    tmax = tmax_flick
    epoch_RF = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks,reject=reject, baseline=(tmin_flick, tmin_flick), preload=True)
    evoked_RF = epoch_RF.average()
    #evoked_RF.plot(time_unit='s')
    ## Epoch: Random rest
    event_id = event_dict[str(custom_event_ids['RR'])]
    tmin = tmin_rest
    tmax = tmax_rest
    epoch_RR = mne.Epochs(raw, events, event_id, tmin_rest, tmax_rest, proj=True,
                        picks=picks,reject=reject, baseline=None, preload=True)
    evoked_RR = epoch_RR.average()
    #evoked_RR.plot(time_unit='s')
    ## Epoch: 40 Hz rest
    event_id = event_dict[str(custom_event_ids['4R'])]
    tmin = tmin_rest
    tmax = tmax_rest
    epoch_4R = mne.Epochs(raw, events, event_id, tmin_rest, tmax_rest, proj=True,
                        picks=picks,reject=reject, baseline=None, preload=True)
    #epoch_4R.drop([0,1])
    evoked_4R = epoch_4R.average()
    #evoked_4R.plot(time_unit='s')
    ## Epoch: 40 Hz flicker
    event_id = event_dict[str(custom_event_ids['4F'])]
    tmin = tmin_flick
    tmax = tmax_flick
    epoch_4F = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks,reject=reject,baseline=(tmin_flick, tmin_flick), preload=True)
    epoch_4F_all = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        baseline=(tmin_flick, tmin_flick), preload=True, 
                        reject=dict())
    #epoch_4F.drop([0,1])
    #epoch_4F_all.drop([0,1])
    evoked_4F = epoch_4F.average()
    #evoked_4F.plot(time_unit='s')
    return epoch_RR,epoch_RF,epoch_4R,epoch_4F

def doMA(psds,n=20):
    for i in range(psds.shape[0]):
        tempSum=0
        for j in range(int(n)):
            if j+i < psds.shape[0]:
                tempSum+=psds[j+i]
            else:
                n=j
                break
        psds[i]=tempSum/n
    return psds
def doMA3D(psds):
    for i in range(psds.shape[0]):
        for j in range(psds.shape[1]):
            psds[i][j] = doMA(psds[i][j])
    return psds
def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate((
        np.ones(noise_n_neighbor_freqs),
        np.zeros(2 * noise_skip_neighbor_freqs + 1),
        np.ones(noise_n_neighbor_freqs)))
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode='valid'),
        axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(
        mean_noise, pad_width=pad_width, constant_values=np.nan
    )

    return psd / mean_noise
def add_arrows(axes):
    # add some arrows at 60 Hz and its harmonics
    for ax in axes:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()
        freq = 40
        idx = np.searchsorted(freqs, freq)
        # get ymax of a small region around the freq. of interest
        y = psds[(idx - 4):(idx + 5)].max()
        ax.arrow(x=freqs[idx], y=y + 18, dx=0, dy=-12, color='red',
                 width=0.1, head_width=3, length_includes_head=True)
def specPlot(epochs):
    # MA: 是否做滑动平均
    for epoch in epochs:
        epoch.plot_psd(fmin=0.1, fmax=100., average=True, spatial_colors=False)
def plot_psd_sub(epoch,ax, fmin=.1, fmax=100, n_jobs=8, color='k', alpha=.5, label='Default', isma=False):
    psds, freqs = psd_multitaper(epoch, fmin=fmin, fmax=fmax, n_jobs=n_jobs)
    # MA: 是否做滑动平均
    if isma:
        psds = doMA3D(psds)
    psds = 10. * np.log10(psds)
    psds_mean = psds.mean(0).mean(0)
    psds_std = psds.mean(0).std(0)
    ax.plot(freqs, psds_mean, color=color, label = label)
    ax.fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
                color=color, alpha=alpha)
    ax.legend()
def superposGamma(epoch_4R,epoch_4F,epoch_RF,subject_name,MA=False):
    fmin = 35
    fmax = 45
    alpha = .2
    f, ax = plt.subplots(figsize=(15,7))
    if MA:
        plot_psd_sub(ax=ax,epoch = epoch_4R, color='y', fmin=fmin, fmax=fmax, alpha=alpha, label='Rest State', isma=True)
        plot_psd_sub(ax=ax,epoch = epoch_4F, color='k', fmin=fmin, fmax=fmax, alpha=alpha, label='40 Hz Light Stimulation', isma=True)
        plot_psd_sub(ax=ax,epoch = epoch_RF, color='r', fmin=fmin, fmax=fmax, alpha=alpha, label='Random Hz Light Stimulation', isma=True)
    else:
        plot_psd_sub(ax=ax,epoch = epoch_4R, color='y', fmin=fmin, fmax=fmax, alpha=alpha, label='Rest State')
        plot_psd_sub(ax=ax,epoch = epoch_4F, color='k', fmin=fmin, fmax=fmax, alpha=alpha, label='40 Hz Light Stimulation')
        plot_psd_sub(ax=ax,epoch = epoch_RF, color='r', fmin=fmin, fmax=fmax, alpha=alpha, label='Random Hz Light Stimulation')
    plt.xlabel("Frequency")
    plt.ylabel("Power spectral density (PSD) in log")
    plt.savefig(dataRoot + '/Light/Light_figures_non/' + subject_name + '_35_45.png')
def superposFull(epoch_4R,epoch_4F,epoch_RF,subject_name,MA=False):
    fmin = 0
    fmax = 120
    alpha = .2
    f, ax = plt.subplots(figsize=(15,7))
    if MA:
        plot_psd_sub(ax=ax,epoch = epoch_4R, color='y', fmin=fmin, fmax=fmax, alpha=alpha, label='Rest State', isma=True)
        plot_psd_sub(ax=ax,epoch = epoch_4F, color='k', fmin=fmin, fmax=fmax, alpha=alpha, label='40 Hz Light Stimulation', isma=True)
        plot_psd_sub(ax=ax,epoch = epoch_RF, color='r', fmin=fmin, fmax=fmax, alpha=alpha, label='Random Hz Light Stimulation', isma=True)
    else:
        plot_psd_sub(ax=ax,epoch = epoch_4R, color='y', fmin=fmin, fmax=fmax, alpha=alpha, label='Rest State')
        plot_psd_sub(ax=ax,epoch = epoch_4F, color='k', fmin=fmin, fmax=fmax, alpha=alpha, label='40 Hz Light Stimulation')
        plot_psd_sub(ax=ax,epoch = epoch_RF, color='r', fmin=fmin, fmax=fmax, alpha=alpha, label='Random Hz Light Stimulation')
    plt.xlabel("Frequency")
    plt.ylabel("Power spectral density (PSD) in log")
    plt.savefig(dataRoot + '/Light/Light_figures_non/' + subject_name + '_0_120.png')
def superpos85(epoch_4R,epoch_4F,epoch_RF,subject_name,MA=False):
    fmin = 40
    fmax = 85
    alpha = .2
    f, ax = plt.subplots(figsize=(15,7))
    if MA:
        plot_psd_sub(ax=ax,epoch = epoch_4R, color='y', fmin=fmin, fmax=fmax, alpha=alpha, label='Rest State', isma=True)
        plot_psd_sub(ax=ax,epoch = epoch_4F, color='k', fmin=fmin, fmax=fmax, alpha=alpha, label='40 Hz Light Stimulation', isma=True)
        plot_psd_sub(ax=ax,epoch = epoch_RF, color='r', fmin=fmin, fmax=fmax, alpha=alpha, label='Random Hz Light Stimulation', isma=True)
    else:
        plot_psd_sub(ax=ax,epoch = epoch_4R, color='y', fmin=fmin, fmax=fmax, alpha=alpha, label='Rest State')
        plot_psd_sub(ax=ax,epoch = epoch_4F, color='k', fmin=fmin, fmax=fmax, alpha=alpha, label='40 Hz Light Stimulation')
        plot_psd_sub(ax=ax,epoch = epoch_RF, color='r', fmin=fmin, fmax=fmax, alpha=alpha, label='Random Hz Light Stimulation')
    plt.xlabel("Frequency")
    plt.ylabel("Power spectral density (PSD) in log")
    plt.savefig(dataRoot + '/Light/Light_figures_non/' + subject_name + '_40_85.png')
def getRatio_rest(epoch,fmin=35.0,fmax=45.0,picks=['O1', 'OZ', 'O2']):
    psds, freqs = psd_multitaper(epoch,fmin=fmin, fmax=fmax, n_jobs=8,picks=picks) 
    # psd.shape: (number of epoch, number of channel, frequency)
    f_down = 35.0
    f_low = 39.0
    f_high = 41.0
    f_upstream = 45.0
    print(psds.shape)
    num_of_epoch = psds.shape[0]
    num_of_channel = psds[0].shape[0]
    print("{0} epochs in total".format(num_of_epoch))
    print("{0} channels in total".format(num_of_channel))
    print(psds.shape)
    # average all channels
    psds = np.mean(psds, axis=1)
    # extract power in selected frequency bands
    
    downstream_mean_power, target_mean_power, upstream_mean_power = [],[],[]
    for i in range(0, num_of_epoch):
        a,b,c = [],[],[]
        for j in range(0, len(freqs)):
            freq = freqs[j]
            power = psds[i][j]
            if freq < f_low and freq > f_down:
                a.append(power)
            if freq < f_high and freq > f_low:
                b.append(power)
            if freq < f_upstream and freq > f_high:
                c.append(power)
        downstream_mean_power.append(np.max(a))
        target_mean_power.append(np.max(b))
        upstream_mean_power.append(np.max(c))
    print(r'The downstream mean power is:')
    print(downstream_mean_power)
    print(r'The taget band mean power is:')
    print(target_mean_power)
    print(r'The upstream mean power is:')
    print(upstream_mean_power)
    ratio_TD = []
    ratio_TU = []
    ratio_DU = []
    for i in range(0, len(target_mean_power)):
        TD = target_mean_power[i]/downstream_mean_power[i]
        TU = target_mean_power[i]/upstream_mean_power[i]
        DU = downstream_mean_power[i]/upstream_mean_power[i]
        ratio_TD.append(TD)
        global ratio_TD_all_r
        ratio_TD_all_r.append(TD)
        ratio_TU.append(TU)
        global ratio_TU_all_r
        ratio_TU_all_r.append(TU)
        ratio_DU.append(DU)
        global ratio_DU_all_r
        ratio_DU_all_r.append(DU)
    print(r'The target/downstream is:')
    print(ratio_TD)
    print(r'The target/upstream is:')
    print(ratio_TU)
    print(r'The downstream/upstream is:')
    print(ratio_DU)
    return downstream_mean_power,target_mean_power,upstream_mean_power
def getRatio_flicker(epoch,fmin=35.0,fmax=45.0,picks=['O1', 'OZ', 'O2']):
    psds, freqs = psd_multitaper(epoch,fmin=fmin, fmax=fmax, n_jobs=8,picks=picks) 
    # psd.shape: (number of epoch, number of channel, frequency)
    f_down = 35.0
    f_low = 39.0
    f_high = 41.0
    f_upstream = 45.0
    print(psds.shape)
    num_of_epoch = psds.shape[0]
    num_of_channel = psds[0].shape[0]
    print("{0} epochs in total".format(num_of_epoch))
    print("{0} channels in total".format(num_of_channel))
    print(psds.shape)
    # average all channels
    psds = np.mean(psds, axis=1)
    # extract power in selected frequency bands
    downstream_mean_power, target_mean_power, upstream_mean_power = [],[],[]
    for i in range(0, num_of_epoch):
        a,b,c = [],[],[]
        for j in range(0, len(freqs)):
            freq = freqs[j]
            power = psds[i][j]
            if freq < f_low and freq > f_down:
                a.append(power)
            if freq < f_high and freq > f_low:
                b.append(power)
            if freq < f_upstream and freq > f_high:
                c.append(power)
        downstream_mean_power.append(np.max(a))
        target_mean_power.append(np.max(b))
        upstream_mean_power.append(np.max(c))
    print(r'The downstream mean power is:')
    print(downstream_mean_power)
    print(r'The taget band mean power is:')
    print(target_mean_power)
    print(r'The upstream mean power is:')
    print(upstream_mean_power)
    ratio_TD = []
    ratio_TU = []
    ratio_DU = []
    for i in range(0, len(target_mean_power)):
        TD = target_mean_power[i]/downstream_mean_power[i]
        TU = target_mean_power[i]/upstream_mean_power[i]
        DU = downstream_mean_power[i]/upstream_mean_power[i]
        ratio_TD.append(TD)
        global ratio_TD_all_f
        ratio_TD_all_f.append(TD)
        ratio_TU.append(TU)
        global ratio_TU_all_f
        ratio_TU_all_f.append(TU)
        ratio_DU.append(DU)
        global ratio_DU_all_f
        ratio_DU_all_f.append(DU)
    print(r'The target/downstream is:')
    print(ratio_TD)
    print(r'The target/upstream is:')
    print(ratio_TU)
    print(r'The downstream/upstream is:')
    print(ratio_DU)
    return downstream_mean_power,target_mean_power,upstream_mean_power
def get_minima(data):
    center = np.argmax(data)
    print(center)
    min_right = data[center]
    min_left = data[center]
    for i in range(center, data.shape[0]-1):
        if data[i+1] <= min_right:
            min_right = data[i+1]
        else:
            break
    for i in range(center, 0, -1):
        if data[i-1] <= min_left:
            min_left = data[i-1]
        else:
            break
    return min_left, min_right
def getRatio_flicker_MA(epoch,fmin=38.0,fmax=42.0,picks=['O1', 'OZ', 'O2']):
    psds, freqs = psd_multitaper(epoch,fmin=fmin, fmax=fmax, n_jobs=8,picks=picks) 
    # MA: 做滑动平均
    psds = doMA3D(psds)
    # psd.shape: (number of epoch, number of channel, frequency)
    f_down = 35.0
    f_low = 39.0
    f_high = 41.0
    f_upstream = 45.0
    print(psds.shape)
    num_of_epoch = psds.shape[0]
    num_of_channel = psds[0].shape[0]
    print("{0} epochs in total".format(num_of_epoch))
    print("{0} channels in total".format(num_of_channel))
    print(psds.shape)
    # average all channels
    psds = np.mean(psds, axis=1)
    # extract power in selected frequency bands
    downstream_mean_power, target_mean_power, upstream_mean_power = [],[],[]
    for i in range(0, num_of_epoch): 
        a,c = get_minima(psds[i])
        b = np.max(psds[i])
        downstream_mean_power.append(a)
        target_mean_power.append(b)
        upstream_mean_power.append(c)
    print(r'The downstream min power is:')
    print(downstream_mean_power)
    print(r'The taget band max power is:')
    print(target_mean_power)
    print(r'The upstream min power is:')
    print(upstream_mean_power)
    ratio_TD = []
    ratio_TU = []
    ratio_DU = []
    for i in range(0, len(target_mean_power)):
        TD = target_mean_power[i]/downstream_mean_power[i]
        TU = target_mean_power[i]/upstream_mean_power[i]
        DU = downstream_mean_power[i]/upstream_mean_power[i]
        ratio_TD.append(TD)
        global ratioMA_TD_all_f
        ratioMA_TD_all_f.append(TD)
        ratio_TU.append(TU)
        global ratioMA_TU_all_f
        ratioMA_TU_all_f.append(TU)
        ratio_DU.append(DU)
        global ratioMA_DU_all_f
        ratioMA_DU_all_f.append(DU)
    print(r'The target/downstream is:')
    print(ratio_TD)
    print(r'The target/upstream is:')
    print(ratio_TU)
    print(r'The downstream/upstream is:')
    print(ratio_DU)
    return downstream_mean_power,target_mean_power,upstream_mean_power
def getRatio_rest_MA(epoch,fmin=38.0,fmax=42.0,picks=['O1', 'OZ', 'O2']):
    psds, freqs = psd_multitaper(epoch,fmin=fmin, fmax=fmax, n_jobs=8,picks=picks) 
    # MA: 做滑动平均
    psds = doMA3D(psds)
    # psd.shape: (number of epoch, number of channel, frequency)
    f_down = 35.0
    f_low = 39.0
    f_high = 41.0
    f_upstream = 45.0
    print(psds.shape)
    num_of_epoch = psds.shape[0]
    num_of_channel = psds[0].shape[0]
    print("{0} epochs in total".format(num_of_epoch))
    print("{0} channels in total".format(num_of_channel))
    print(psds.shape)
    # average all channels
    psds = np.mean(psds, axis=1)
    # extract power in selected frequency bands
    downstream_mean_power, target_mean_power, upstream_mean_power = [],[],[]
    for i in range(0, num_of_epoch): 
        a,c = get_minima(psds[i])
        b = np.max(psds[i])
        downstream_mean_power.append(a)
        target_mean_power.append(b)
        upstream_mean_power.append(c)
    print(r'The downstream min power is:')
    print(downstream_mean_power)
    print(r'The taget band max power is:')
    print(target_mean_power)
    print(r'The upstream min power is:')
    print(upstream_mean_power)
    ratio_TD = []
    ratio_TU = []
    ratio_DU = []
    for i in range(0, len(target_mean_power)):
        TD = target_mean_power[i]/downstream_mean_power[i]
        TU = target_mean_power[i]/upstream_mean_power[i]
        DU = downstream_mean_power[i]/upstream_mean_power[i]
        ratio_TD.append(TD)
        global ratioMA_TD_all_r
        ratioMA_TD_all_r.append(TD)
        ratio_TU.append(TU)
        global ratioMA_TU_all_r
        ratioMA_TU_all_r.append(TU)
        ratio_DU.append(DU)
        global ratioMA_DU_all_r
        ratioMA_DU_all_r.append(DU)
    print(r'The target/downstream is:')
    print(ratio_TD)
    print(r'The target/upstream is:')
    print(ratio_TU)
    print(r'The downstream/upstream is:')
    print(ratio_DU)
    return downstream_mean_power,target_mean_power,upstream_mean_power

def calc_abs_power_simp(epoch, fmin, fmax, fbottom, ftop, sfreq):
    epoch = np.asarray(epoch)
    band_power_all = []
    # psds, freqs = mne.time_frequency.psd_array_multitaper(epoch, n_jobs=8, sfreq=500.0) 
    psds, freqs = mne.time_frequency.psd_array_multitaper(epoch, fmin=fbottom, fmax=ftop, n_jobs=8, sfreq=500.0)
    # Average all channels
    psds = np.mean(psds, axis=1)
    # Resolution of calculation
    freq_res = freqs[1] - freqs[0]
    # Select frequency band
    band = [fmin, fmax]
    idx_band = ((freqs >= band[0]) & (freqs <= band[1]))
    print(freqs.shape)
    print(psds.shape)
    # integrate the PSD
    num = (fmax-fmin)/freq_res
    for i in range(psds.shape[0]):
        band_power = simps(psds[i][idx_band], dx=freq_res)
        band_power_all.append(band_power)
        # band_power_all.append(band_power/(num))
    print(band)
    print(band_power_all)
    return band_power_all

def cal_abs_power_allch(epoch, fmin, fmax, fbottom, ftop, sfreq):
    epoch = np.asarray(epoch)
    band_power_all = []
    # psds, freqs = mne.time_frequency.psd_array_multitaper(epoch, n_jobs=8, sfreq=500.0) 
    psds, freqs = mne.time_frequency.psd_array_multitaper(epoch, fmin=fbottom, fmax=ftop, n_jobs=8, sfreq=500.0)
    # Average all channels
    psds = np.mean(psds, axis=0)
    # Resolution of calculation
    freq_res = freqs[1] - freqs[0]
    # Select frequency band
    band = [fmin, fmax]
    idx_band = ((freqs >= band[0]) & (freqs <= band[1]))
    print(freqs.shape)
    print(psds.shape)
    # integrate the PSD
    num = (fmax-fmin)/freq_res
    for i in range(psds.shape[0]):
        band_power = simps(psds[i][idx_band], dx=freq_res)
        band_power_all.append(band_power)
        # band_power_all.append(band_power/(num))
    print(band)
    print(band_power_all)
    return band_power_all

def get_all_abs_power(epoch_R, epoch_F, f_bottom = 35.0, f_low = 39.0, f_high = 41.0, f_top = 50.0, sfreq=500.0):
    power_all_R = calc_abs_power_simp(epoch=epoch_R, fmin=f_bottom, fmax=f_top, fbottom=f_bottom, ftop=f_top, sfreq=500.0)
    power_R = calc_abs_power_simp(epoch=epoch_R, fmin=f_low, fmax=f_high, fbottom=f_bottom, ftop=f_top, sfreq=500.0)
    power_all_F = calc_abs_power_simp(epoch=epoch_F, fmin=f_bottom, fmax=f_top, fbottom=f_bottom, ftop=f_top, sfreq=500.0)
    power_F = calc_abs_power_simp(epoch=epoch_F, fmin=f_low, fmax=f_high, fbottom=f_bottom, ftop=f_top, sfreq=500.0)
    return power_all_R, power_R, power_all_F, power_F

def get_allch_abs_power(epoch_R, epoch_F, f_bottom = 35.0, f_low = 39.0, f_high = 41.0, f_top = 50.0, sfreq=500.0):
    power_all_R = cal_abs_power_allch(epoch=epoch_R, fmin=f_bottom, fmax=f_top, fbottom=f_bottom, ftop=f_top, sfreq=500.0)
    power_R = cal_abs_power_allch(epoch=epoch_R, fmin=f_low, fmax=f_high, fbottom=f_bottom, ftop=f_top, sfreq=500.0)
    power_all_F = cal_abs_power_allch(epoch=epoch_F, fmin=f_bottom, fmax=f_top, fbottom=f_bottom, ftop=f_top, sfreq=500.0)
    power_F = cal_abs_power_allch(epoch=epoch_F, fmin=f_low, fmax=f_high, fbottom=f_bottom, ftop=f_top, sfreq=500.0)
    return power_all_R, power_R, power_all_F, power_F

def cal_rel_power_R(power_target, power_all):
    global rel_power_R
    for i in range(min(len(power_target), len(power_all))):
        rel_power_R.append(power_target[i]/power_all[i])
    print(rel_power_R)
    return rel_power_R, min(len(power_target), len(power_all))

def cal_rel_power_F(power_target, power_all):
    global rel_power_F
    for i in range(min(len(power_target), len(power_all))):
        rel_power_F.append(power_target[i]/power_all[i])
    print(rel_power_F)
    return rel_power_F, min(len(power_target), len(power_all))

def cal_rel_power_R_ran(power_target, power_all):
    global rel_power_R_ran
    for i in range(min(len(power_target), len(power_all))):
        rel_power_R_ran.append(power_target[i]/power_all[i])
    print(rel_power_R_ran)
    return rel_power_R_ran, min(len(power_target), len(power_all))

def cal_rel_power_F_ran(power_target, power_all):
    global rel_power_F_ran
    for i in range(min(len(power_target), len(power_all))):
        rel_power_F_ran.append(power_target[i]/power_all[i])
    print(rel_power_F_ran)
    return rel_power_F_ran, min(len(power_target), len(power_all))

def save_rel_powers_subj(length_R, length_F, filefolder = dataRoot + '/Light/csvs', subject_name="Undefined"):
    # 添加样本序号索引
    global rel_power_R, rel_power_F, subj_number_R, subj_number_F
    subj_number_R = subj_number_R + list([subject_name])*length_R
    subj_number_F = subj_number_F + list([subject_name])*length_F
    df_R, df_F,df_subj_R, df_subj_F = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_R['rel_power_rest'] = rel_power_R
    df_F['rel_power_flicker'] = rel_power_F
    df_subj_R["Subject Number Rest"] = subj_number_R
    df_subj_F["Subject Number Flicker"] = subj_number_F
    df_csv = pd.concat([df_R, df_subj_R, df_F, df_subj_F], axis=1)
    df_csv.to_csv(filefolder + '/rel_powers.csv')

def save_rel_powers(filefolder = dataRoot + '/Light/csvs'):
    df_R, df_F= pd.DataFrame(), pd.DataFrame()
    df_R['rel_power_rest'] = rel_power_R
    df_F['rel_power_flicker'] = rel_power_F
    df_csv = pd.concat([df_R, df_F], axis=1)
    df_csv.to_csv(filefolder + '/rel_powers.csv')

def calc_psds(epoch, fmin, fmax, n_jobs=1, type='multitaper'):
    if type=='multitaper':
        psds, freqs = psd_multitaper(epoch, fmin=fmin, fmax=fmax, n_jobs=n_jobs)
        return psds, freqs
    return 
    
def save_psd(psd, freqs, filepath='default.csv', pad=5):
    # psd = np.mean(psd, axis=1）
    psd = psd[:, ::pad]
    freqs = freqs[::pad]
    psd = np.insert(psd, 0, values=freqs, axis=0)
    np.savetxt(filepath, psd, delimiter=',')
    return