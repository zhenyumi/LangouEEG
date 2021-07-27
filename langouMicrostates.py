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
from mne.viz import plot_topomap
from mpl_toolkits.mplot3d import Axes3D  # noqa
from lempel_ziv_complexity import lempel_ziv_complexity

def plot_substate(epoch, maps, n_maps, dpi=300, save=False, filename='Default', fmt='.png', result_dir=''):
    fig, axis = plt.subplots(1, n_maps, dpi=dpi)
    for i in range(0,n_maps):
        axis[i].set_title('state{0}'.format(i+1))
        plot_topomap(maps[i], epoch.info, axes=axis[i], show=False)
    fig.show()
    if save:
        fig.savefig(result_dir + '/' + filename + fmt)
    return fig

def save_sub_stateplots(epoch, maps, n_maps, dpi=300, save=False, filename='Default', fmt='.png', result_dir=''):
    result_dir = result_dir + '/' + fmt
    if not os.path.exists(result_dir):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(result_dir)
    for i in range(0,n_maps):
        fig = plt.figure(dpi = dpi)
        plot_topomap(maps[i], epoch.info, show=False)
        fig.savefig(result_dir + '/' + filename + "_{0}".format(i) + fmt)
        fig.clf()
    return

def display_gfp_peaks(gfp_peaks, x, fs):
    pps = len(gfp_peaks) / (len(x)/fs)  # peaks per second
    print(f"\nGFP peaks per sec.: {pps:.2f}")
    return

def display_gev(gev):
    print("\nGlobal explained variance (GEV) per map:")
    for i, g in enumerate(gev): print(f"GEV(ms-{i:d}) = {gev[i]:.2f}")
    print(f"\ntotal GEV: {gev.sum():.3f}")
    return

def display_info(x, n_maps, gfp_peaks, gev, fs):
    p_hat = p_empirical(x, n_maps)
    T_hat = T_empirical(x, n_maps)
    print("\nEmpirical symbol distribution (RTT):\n")
    for i in range(n_maps):
        print(f"p_{i:d} = {p_hat[i]:.3f}")
    print("\nEmpirical transition matrix:\n")
    print_matrix(T_hat)

    display_gfp_peaks(gfp_peaks=gfp_peaks,x=x, fs=fs)
    display_gev(gev)

    h_hat = H_1(x, n_maps)
    h_max = max_entropy(n_maps)
    print(f"\nEmpirical entropy H = {h_hat:.2f} (max. entropy: {h_max:.2f})")
    h_rate, _ = excess_entropy_rate(x, n_maps, kmax=8, doplot=True)
    h_mc = mc_entropy_rate(p_hat, T_hat)
    print(f"\nEmpirical entropy rate h = {h_rate:.2f}")
    print(f"Theoretical MC entropy rate h = {h_mc:.2f}")
    return

def display_states(x, pca1):
    fig, ax = plt.subplots(2, 1, figsize=(15,4), sharex=True)
    ax[0].plot(x[0:3000])
    ax[1].plot(pca1[0:3000])
    return

def save_to_file(array, filepath):
    np.asarray(array)
    df = pd.DataFrame (array)
    df.to_csv(filepath, index=False)
    return 

def save_cache(folder_path, x, maps, pca, gfp_peaks, gev, state, fig, time_augs, fmt='.png'):
    if not os.path.exists(folder_path):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder_path)
    save_to_file(time_augs,  folder_path + '/4R4R4F4FRFRF_time.csv')
    save_to_file(x, folder_path + '/x_{0}.csv'.format(state))
    # save_to_file(maps, folder_path + '/maps_{0}.csv'.format(state))
    save_to_file(pca, folder_path + '/pca_{0}.csv'.format(state))
    save_to_file(gfp_peaks, folder_path + '/gfp_peaks_{0}.csv'.format(state))
    save_to_file(gev, folder_path + '/gev_{0}.csv'.format(state))
    fig.savefig(folder_path + '/' + state + fmt)
    return

'''
LZc
'''
def LZC(x, epochs):
    strx = ''
    for a in x:
        strx += str(a)
    epochs_count = len(list(epochs.values()))
    epoch_len = int(x.shape[0]/epochs_count)
    lzc = []
    for i in range(0,x.shape[0],epoch_len):
        lzc.append(lempel_ziv_complexity(strx[i:i+epoch_len]))
    return lzc

def display_lzc(lzc):
    print("The lzc:")
    print(np.shape(lzc))
    print(lzc)
    return

def display_maps(epoch, tm, n_maps=4, save=False, dpi=300, filename='Default', fmt='.png', to_save_cache=False, time_augs=[0,0,0,0], result_dir='', calc_lzc=False, epochs=None):
    data_raw = np.hstack(epoch.get_data()).T
    fs = 500
    data = bp_filter(data_raw, f_lo=2, f_hi=20, fs=fs)
    print(data.shape)
    pca = PCA(copy=True, n_components=1, whiten=False)
    pca1 = pca.fit_transform(data)[:,0]

    plot_data(pca1, fs)
    t0, t1 = 1000, 3000
    plot_data(pca1[t0:t1], fs)
    mode = ["aahc", "kmeans", "kmedoids", "pca", "ica"][1]
    print(f"Clustering algorithm: {mode:s}")
    n_maps = n_maps
    chs = 64
    locs = []
    maps, x, gfp_peaks, gev = clustering(data, fs, chs, locs, mode, n_maps, interpol=False, doplot=False)
    display_info(x, n_maps, gfp_peaks, gev, fs)
    display_states(x, pca1)
    if calc_lzc:
        lzc = LZC(x, epochs)
        display_lzc(lzc)
    fig = plot_substate(epoch=epoch, maps=maps, n_maps=n_maps, 
                        save=save, dpi=dpi, filename=filename, fmt=fmt, result_dir=result_dir)
    # Save cache
    if to_save_cache:
        folder_path = result_dir + '/cache/' + tm
        save_cache(time_augs = time_augs, folder_path=folder_path, x=x, 
        maps=maps, pca=pca1, gfp_peaks=gfp_peaks, gev=gev, state=filename, fig=fig)
        save_sub_stateplots(epoch=epoch, maps=maps, n_maps=n_maps, 
                            save=save, dpi=dpi, filename=filename, fmt='.png', result_dir=folder_path)
        save_sub_stateplots(epoch=epoch, maps=maps, n_maps=n_maps, 
                            save=save, dpi=dpi, filename=filename, fmt='.svg', result_dir=folder_path)
        save_maps = pd.DataFrame()
        for i in range(0, n_maps):
            save_maps[str(i)] = maps[i]
        save_maps.to_csv(folder_path + '/maps_{0}.csv'.format(filename))
    return


