# coding: utf-8

"""
@author: Yulin Liu, Lu Dai
@ITS Berkeley
"""
from __future__ import division
import os
import numpy as np
from itertools import groupby, count
from operator import itemgetter
import matplotlib.pyplot as plt

class AudioActDet:
    def __init__(self, AudioFeatures):
        
        self.sec_to_bin, self.time_ins, self.Pxx = AudioFeatures.sec_to_bin, AudioFeatures.time_ins.copy(), AudioFeatures.Pxx.copy()

    def Movingavg(self, x, n = 5):
        padarray = np.lib.pad(x, n//2, 'edge')
        csum = np.cumsum(np.insert(padarray, 0, 0))
        y = (csum[n:] - csum[:-n])/n
        return y


    def combine_to_range(self, power, power_threshold, sec_to_bin, silence_sec = 0.5):
        # power is the matrix produced by specgram
        # power_threshold usually is 0
        # sec_to_bin is usally bin.shape[0] / audio length(ms)/1000
        
        idx_range = []

        groups = groupby(np.where(power <= power_threshold)[0], key = lambda i,x = count():i-next(x))
        tmp = [list(g) for k, g in groups]

        for item in tmp:
            if len(item) == 1:
                idx_range.append([item[0], item[0]])
            else:
                idx_range.append([item[0], item[-1]])


        # for k, g in groupby(enumerate(np.where(power <= power_threshold)[0]), key = lambda (i,x):i-x):
        #     group = map(itemgetter(1), g)
        #     idx_range.append([group[0], group[-1]])

        idx_range = np.array(idx_range)
        silence_time_idx = idx_range[np.where((idx_range[:,1] 
                                                  - idx_range[:,0]) > silence_sec * sec_to_bin), :][0]
        silence_time_range = idx_range[np.where((idx_range[:,1] 
                                                  - idx_range[:,0]) > silence_sec * sec_to_bin), :][0]/sec_to_bin
        idx_nums = ()
        for i in silence_time_idx:
            idx_nums = np.append(idx_nums,range(int(i[0]), int(i[1]+1)))
        idx_act = np.sort(np.array(list(set(range(power.shape[0])).difference(set(idx_nums)))))
        
        silence_time_duration = silence_time_range[:,1] - silence_time_range[:,0]
        silence_time = sum(silence_time_duration)
        return silence_time_idx, silence_time, idx_act

    def detect_silence(self, power_threshold = 0, silence_sec = 0.5, mvg_point = 5, PLOT = False, **kwarg):
        # power_threshold is a value that greater than 0. Usually is 1
        # silence_sec_s is the minimum duration of a silence, in seconds. Usually is 0.5.

        self.Pxx += self.Pxx.mean()
        Zxx = np.flipud(10. * np.log10(self.Pxx))
        # Power Spectrum Density V**2/Hz

        # High frequency band & Low frequency band
        Hxx = Zxx[0:Zxx.shape[0]//2, :Zxx.shape[1]]
        Lxx = Zxx[Zxx.shape[0]//2:, :Zxx.shape[1]]

        # Apply moving average filter
        hxxf = self.Movingavg(Hxx.sum(axis=0), n = mvg_point)
        lxxf = self.Movingavg(Lxx.sum(axis = 0), n = mvg_point)
        Fxx = lxxf - np.mean(hxxf)
        if PLOT:
            plt.figure(figsize = (18,4))
            if kwarg['tmin'] is None:
                tmin = self.time_ins.min()
            if kwarg['tmax'] is None:
                tmax = tmin + 100
            t = np.linspace(kwarg['tmin'], kwarg['tmax'], (kwarg['tmax'] - kwarg['tmin']) * self.sec_to_bin + 1)
            # print(t.shape)
            # print(Fxx[np.where((self.time_ins >= kwarg['tmin']) & (self.time_ins < kwarg['tmax']))[0]].shape)
            plt.plot(t, Fxx[np.where((self.time_ins >= kwarg['tmin']) & (self.time_ins < kwarg['tmax']))[0]])
            plt.hlines(power_threshold, t.min(), t.max(), linestyles = '--', colors = 'r')
            # plt.xticks
        
        self.silence_seg_2d, _, self.idx_act = self.combine_to_range(Fxx, power_threshold, self.sec_to_bin, silence_sec)
        #si_time_duration = out[1]
        # active_rate = 1 - out[1] / sound_length
        self.silence_seg = self.silence_seg_2d.flatten()
        # self.Pxx_act = Pxx[:, self.idx_act]
        # idx_act is the index vector that has voice activity, when input the diarz_segment index "idx_act[idx_diarz]",
        # Pxx_act is the spectral matrix that has already remove the silence part.
        return self.silence_seg, self.silence_seg_2d, self.idx_act