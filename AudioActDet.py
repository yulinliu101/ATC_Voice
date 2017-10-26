# coding: utf-8

"""
@author: Yulin Liu, Lu Dai
@ITS Berkeley
"""
from __future__ import division
import os
import numpy as np
from itertools import groupby, chain
from operator import itemgetter

class AudioActDet:
    def __init__(self, AudioLoad, AudioFeatures):
        self.AudioClass = AudioLoad
        self.AudioFeatures = AudioFeatures

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
        for k, g in groupby(enumerate(np.where(power <= power_threshold)[0]), lambda (i,x):i-x):
            group = map(itemgetter(1), g)
            idx_range.append([group[0], group[-1]])
        idx_range = np.array(idx_range)
        silence_time_range = idx_range[np.where((idx_range[:,1] 
                                                  - idx_range[:,0]) > silence_sec * sec_to_bin), :][0]
        
        idx_nums = ()
        for i in silence_time_range:
            idx_nums = np.append(idx_nums,range(int(i[0]), int(i[1]+1)))
        idx_act = list(set(range(time_ins.shape[0])).difference(set(idx_nums))) 
        
        silence_time_duration = silence_time_range[:,1] - silence_time_range[:,0]
        silence_time = sum(silence_time_duration)
        return silence_time_range, silence_time, idx_act

    def detect_silence(self, power_threshold = 0, silence_sec = 0.5, mvg_point = 5):
        # power_threshold is a value that greater than 0. Usually is 1
        # silence_sec_s is the minimum duration of a silence, in seconds. Usually is 3.
        
        # Get array of sound
        sample_rate = self.AudioClass.sample_rate
        sound_track = self.AudioClass.sound_track
        sound_length = self.AudioClass.sample_audio.duration_seconds
        
        # Plot specgram and get pxx, freq, bins, im
        # freqs, time_ins, Pxx = scipy.signal.spectrogram(np.array(sound_track), fs = sample_rate, window = 'hann', nperseg = 2048, noverlap = 2048/8, detrend = 'constant', scaling = 'density', mode = 'psd')
        freqs, time_ins, Pxx = self.AudioFeatures.freqs, self.AudioFeatures.time_ins, self.AudioFeatures.Pxx

        Pxx += Pxx.mean()
        Zxx = np.flipud(10. * np.log10(Pxx))
        # Power Spectrum Density V**2/Hz
        # Pxx, _, bins, _ = plt.specgram(sound_track, scale = 'linear', Fs = sample_rate, NFFT = 1024, noverlap = 1024/4)
        # plt.clf()
        # plt.close('all')

        # High frequency band & Low frequency band
        Hxx = Zxx[0:Zxx.shape[0]//2, :Zxx.shape[1]]
        Lxx = Zxx[Zxx.shape[0]//2:, :Zxx.shape[1]]

        # Apply moving average filter
        hxxf = self.Movingavg(Hxx.sum(axis=0), n = mvg_point)
        lxxf = self.Movingavg(Lxx.sum(axis = 0), n = mvg_point)
        Fxx = lxxf - np.mean(hxxf)
        
        sec_to_bin = time_ins.shape[0] / sound_length
        out = self.combine_to_range(Fxx,power_threshold,sec_to_bin, silence_sec)
        #si_time_duration = out[1]
        active_rate = 1 - out[1] / sound_length
        idx_act = out[2]
        Pxx_act = Pxx[:, idx_act]
        # idx_act is the index vector that has voice activity, when input the diarz_segment index "idx_act[idx_diarz]",
        # Pxx_act is the spectral matrix that has already remove the silence part.
        return active_rate, idx_act, Pxx_act
