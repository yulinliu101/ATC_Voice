# -*- coding: utf-8 -*-
# @Author: Yulin Liu, Lu Dai
# @Date:   2018-08-13 16:20:17
# @Last Modified by:   Yulin Liu
# @Last Modified time: 2018-08-13 17:39:56

from utils import combine_to_range, Movingavg
import numpy as np
# from itertools import groupby, count
# from operator import itemgetter
# import matplotlib.pyplot as plt # plot func disabled

def voice_activity_detector(sec_to_bin, time_ins, Pxx, power_threshold = 0, silence_sec = 0.1, mvg_point = 5, **kwarg):
	# power_threshold is a value that greater than 0. Usually is 1
    # silence_sec_s is the minimum duration of a silence, in seconds. Usually is 0.1.

    Pxx += Pxx.mean()
    Zxx = np.flipud(10. * np.log10(Pxx))
    # Power Spectrum Density V**2/Hz

    # High frequency band & Low frequency band
    Hxx = Zxx[0:Zxx.shape[0]//2, :Zxx.shape[1]]
    Lxx = Zxx[Zxx.shape[0]//2:, :Zxx.shape[1]]

    # Apply moving average filter
    hxxf = Movingavg(Hxx.sum(axis = 0), n = mvg_point)
    lxxf = Movingavg(Lxx.sum(axis = 0), n = mvg_point)
    Fxx = lxxf - np.mean(hxxf)
    # if PLOT:
    #     plt.figure(figsize = (18,4))
    #     if kwarg['tmin'] is None:
    #         tmin = self.time_ins.min()
    #     if kwarg['tmax'] is None:
    #         tmax = tmin + 100
    #     t = np.linspace(kwarg['tmin'], kwarg['tmax'], (kwarg['tmax'] - kwarg['tmin']) * self.sec_to_bin + 1)
    #     # print(t.shape)
    #     # print(Fxx[np.where((self.time_ins >= kwarg['tmin']) & (self.time_ins < kwarg['tmax']))[0]].shape)
    #     plt.plot(t, Fxx[np.where((self.time_ins >= kwarg['tmin']) & (self.time_ins < kwarg['tmax']))[0]])
    #     plt.hlines(power_threshold, t.min(), t.max(), linestyles = '--', colors = 'r')
    #     # plt.xticks
    
    silence_seg_2d, _, idx_act = combine_to_range(Fxx, power_threshold, sec_to_bin, silence_sec)
    #si_time_duration = out[1]
    # active_rate = 1 - out[1] / sound_length
    silence_seg = silence_seg_2d.flatten()
    # self.Pxx_act = Pxx[:, self.idx_act]
    # idx_act is the index vector that has voice activity, when input the diarz_segment index "idx_act[idx_diarz]",
    # Pxx_act is the spectral matrix that has already remove the silence part.
    return silence_seg, silence_seg_2d, idx_act