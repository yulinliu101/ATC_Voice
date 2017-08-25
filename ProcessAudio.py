# coding: utf-8

"""
@author: Yulin Liu, Lu Dai
@ITS Berkeley
"""
from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub import scipy_effects
from operator import itemgetter
from itertools import groupby, chain
import scipy
import warnings
import time

# Useful Functions
def combine_to_range(power, power_threshold, sec_to_bin, min_silence_s):
    # power is the matrix produced by specgram
    # sec_to_bin is usally bin.shape[0] / audio length(ms)/1000
    
    idx_range = []
    for k, g in groupby(enumerate(np.where(power.sum(axis = 0) <= power_threshold)[0]), lambda (i,x):i-x):
        group = map(itemgetter(1), g)
        idx_range.append([group[0], group[-1]])
    idx_range = np.array(idx_range)
    silence_time_range = idx_range[np.where(((idx_range[:,1] 
                                              - idx_range[:,0])/sec_to_bin) > min_silence_s), :][0]/sec_to_bin
    silence_time_duration = silence_time_range[:,1] - silence_time_range[:,0]
    silence_time = sum(silence_time_duration)
    return silence_time_range, silence_time_duration, silence_time

def detect_silence(sound, power_threshold, min_silence_s):
    # power_threshold is a value that greater than 0. Usually is 1
    # min_silence_s is the minimum duration of a silence, in seconds. Usually is 3.
    
    # Get array of sound
    sample_rate = sound.frame_rate
    sound_track = sound.get_array_of_samples()
    sound_length = sound.duration_seconds
    
    # Plot specgram and get pxx, freq, bins, im
    # freqs, time_ins, Pxx = scipy.signal.spectrogram(np.array(sound_track), fs = sample_rate, window = 'hann', nperseg = 2048, noverlap = 2048/8, detrend = 'constant', scaling = 'density', mode = 'psd')
    freqs, time_ins, Pxx = scipy.signal.stft(sound_track, fs = sample_rate, nperseg = 2048, noverlap = 2048/8,\
                                 window = 'hann', return_onesided = True, padded = False, boundary = None, detrend = 'constant')
    Pxx = (np.abs(Pxx)/32678)**2 # PSD V**2
    # Pxx, _, bins, _ = plt.specgram(sound_track, scale = 'linear', Fs = sample_rate, NFFT = 1024, noverlap = 1024/4)
    # plt.clf()
    # plt.close('all')

    sec_to_bin = time_ins.shape[0] / sound_length
    out = combine_to_range(Pxx,power_threshold,sec_to_bin, min_silence_s)
    si_time_duration = out[1]
    active_rate = 1 - out[2] / sound_length
    return si_time_duration, active_rate

class DetectAudioActivity:
    def __init__(self, Airport = 'KJFK', root_dir = "/AudioDownload/Tower/", File_Type = 'Twr', Anal_Date = 'Apr-19-2017', Anal_Sample_Time = ['1800Z', '1830Z'], combine_sample = False):
        # File_Type: 'Twr', 'ROBER', 'Final', 'CAMRN'
        self.path = os.getcwd() + root_dir
        self.start_str = Airport + '-' + File_Type + '-' + Anal_Date
        print('Analyzed File Type and Date: %s'%self.start_str)
        self.daily_file_list = [filename for filename in os.listdir(self.path) if filename.startswith(self.start_str)]
        self.daily_file_list.sort()
        self.sample_audio_file_list = []
        self.combine_sample = combine_sample
        if self.combine_sample:
            self.sample_audio = AudioSegment.empty()
        else:
            self.sample_audio = []

        try:
            for sample_time in Anal_Sample_Time:
                self.sample_audio_file_list.append(self.start_str + '-' + sample_time + '.mp3')
                if self.combine_sample:
                    self.sample_audio += AudioSegment.from_mp3(self.path + self.start_str + '-' + sample_time + '.mp3')
                else:
                    self.sample_audio.append(AudioSegment.from_mp3(self.path + self.start_str + '-' + sample_time + '.mp3'))

            if combine_sample:
                print('Duration of the sample audio: %.2f'%self.sample_audio.duration_seconds)
                print('Sampling rate of the sample audio: %d'%self.sample_audio.frame_rate)
                self.sample_rate = self.sample_audio.frame_rate
            else:
                print('Sampling rate of the sample audio: %d'%self.sample_audio[0].frame_rate)
                self.sample_rate = self.sample_audio[0].frame_rate
        except:
            print('No sample audio loaded')
            pass
    def DetectActivity(self, anal_source = 'daily', pwr_thres = 1, min_sil_sec = 3):
        self.si_duration_agg = []
        self.active_rates = []
        self.audio_file_date = []
        if anal_source == 'daily':
            print('Detect audio activity on the daily level')
            st = time.time()
            for audio_file in self.daily_file_list:
                print('Processing %s, elapsed time %.2f'%(audio_file, time.time() - st))
                self.audio_file_date.append(audio_file)
                sound = AudioSegment.from_mp3(self.path + audio_file)
                si_duration, active_rate = detect_silence(sound, pwr_thres, min_sil_sec)
                self.si_duration_agg.append(si_duration.tolist())
                self.active_rates.append(active_rate)
            self.daily_gap = list(chain.from_iterable(self.si_duration_agg))

        elif anal_source == 'sample':
            print('Detect audio activity on the audio sample')
            if self.combine_sample:
                print('-----------Processing-------------')
                si_duration, active_rate = detect_silence(self.sample_audio, pwr_thres, min_sil_sec)
                self.si_duration_agg.append(si_duration.tolist())
                self.active_rates.append(active_rate)
                self.daily_gap = list(chain.from_iterable(self.si_duration_agg))
                print('Finished')
            else:
                i = 0
                for audio_file in self.sample_audio_file_list:
                    print('-----------Processing %s-------------'%audio_file)
                    si_duration, active_rate = detect_silence(self.sample_audio[i], pwr_thres, min_sil_sec)
                    self.si_duration_agg.append(si_duration.tolist())
                    self.active_rates.append(active_rate)
                    i += 1
                self.daily_gap = list(chain.from_iterable(self.si_duration_agg))
        return self.daily_gap, self.si_duration_agg, self.active_rates

    def Plot_Activity_Rate(self, max_gap_sec = 180, bin_size = 10):
        warnings.warn('This function currently only works well for results based on anal_source = \'daily\'')
        # Histogram of daily gap activity
        plt.figure(figsize = (16,12))
        plt.subplot(121)
        plt.hist(self.daily_gap, bins = range(0, min(int(max(self.daily_gap))+20, max_gap_sec), bin_size))
        plt.title('Histogram of silence duration for ' + self.start_str[9:])
        plt.xlabel('Silence Duration(second)')
        plt.ylabel('Count')
        plt.ylim(0,1200)

        # Scatter plot of active rates
        plt.subplot(122)
        plt.plot(np.linspace(0,24,len(self.active_rates)), self.active_rates)
        plt.title('Active Rate of ' + self.start_str[9:])
        plt.xticks(range(25), range(0,25))
        plt.xlabel('UTC Hour')
        plt.ylabel('Active Rate')
        plt.ylim(0,1)
        plt.show()

    def VisualizeSampleVoice(self, start_sec = 0, end_sec = 300, NFFT = 2048, overlap_rate = 8, sum_psd = False):
        if self.combine_sample:
            sound_track = np.array(self.sample_audio.get_array_of_samples())/32678
        else:
            print('plot audio clip for %s'%self.sample_audio_file_list[0])
            sound_track = np.array(self.sample_audio[0].get_array_of_samples())/32678
        if end_sec - start_sec > 300:
            warnings.warn('The audio clip to visualize is longer than 300 seconds, which might induce memory overflow. Try a shorter clip')
        t = np.linspace(start_sec, end_sec, (end_sec - start_sec) * self.sample_rate)
        plt.figure(figsize=(18,15))
        ax1 = plt.subplot(311)
        ax1.plot(t, sound_track[int(start_sec * self.sample_rate): int(end_sec * self.sample_rate)])
        ax1.set_xlabel('Elapsed time since audio starts/sec')
        ax1.set_ylabel('Audio/ Volt')

        ax2 = plt.subplot(312, sharex = ax1)
        freqs, time_ins, Pxx = scipy.signal.spectrogram((sound_track[int(start_sec * self.sample_rate): int(end_sec * self.sample_rate)]), 
                                                        fs = self.sample_rate, window = 'hann', nperseg = NFFT, noverlap = NFFT/overlap_rate, 
                                                        detrend = 'constant', scaling = 'density', mode = 'psd')

        Zxx = np.flipud(10. * np.log10(Pxx))

        # pad_xextent = (NFFT-noverlap) / Fs / 2
        # xextent = np.min(t) - pad_xextent, np.max(t) + pad_xextent
        xmin, xmax = (start_sec, end_sec)
        extent = xmin, xmax, freqs[0], freqs[-1]
        im = ax2.imshow(Zxx, aspect = 'auto', extent = extent)
        
        # Pxx, freqs, time_ins, im = ax2.specgram(sound_track[int(start_sec * self.sample_rate): int(end_sec * self.sample_rate)], 
        #                                     Fs = self.sample_rate, NFFT = NFFT, noverlap = NFFT/overlap_rate, xextent = (start_sec, end_sec),
        #                                     mode = 'psd', scale = 'dB')
        # # power, frequencies, times
        # # freqs: 0 Hz -- sample_rate/2 (i * sample_rate/NFFT, i from 0 to NFFT/2)
        # # time_ins: interval is (NFFT-noverlap)/sample_rate
        cb = plt.colorbar(im, orientation='horizontal')
        cb.set_label('Audio Cross Spectral Density (dB/Hz)')
        ax2.set_xlabel('Elapsed time since audio starts/sec')
        ax2.set_ylabel('Frequency (Hz)')

        if sum_psd:
            ax3 = ax2.twinx()
            ax3.plot(time_ins, np.sum(Pxx, axis = 0), 'b-')
            ax3.set_ylabel('Sum of PSD across all freq. (V**2/Hz)', color='b')
        else:
            pass
        plt.show()
        return Pxx, freqs, time_ins
