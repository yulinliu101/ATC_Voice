# coding: utf-8

"""
@author: Yulin Liu, Lu Dai
@ITS Berkeley
"""
from __future__ import division
import os
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt


# path = "/Users/dl/GSR/Audio/"
# files = [path + i for i in os.listdir(path)
#          if os.path.isfile(os.path.join(path,i)) and 'Apr' in i]

class AudioLoad:
    def __init__(self, file_list, verbose = False):
        
        # File_Type: 'Twr', 'ROBER', 'Final', 'CAMRN'
        # print('Analyzed File Type and Date: %s'%self.start_str)
        
        # self.daily_file_list = [filename for filename in os.listdir(self.path) if filename.startswith(self.start_str)]
        # self.daily_file_list.sort()
        # sample_audio_file_list = []
        sample_audio = AudioSegment.empty()
        # self.combine_sample = combine_sample
        # if self.combine_sample:
        #     sample_audio = AudioSegment.empty()
        # else:
        #     sample_audio = []

        try:
            for file_name in file_list:
                # sample_audio_file_list.append(self.start_str + '-' + sample_time + '.mp3')
                # if self.combine_sample:
                if verbose:
                    print('Analyzed File Type and Date: %s'%file_name)
                sample_audio += AudioSegment.from_mp3(file_name)
                # else:
                #     sample_audio.append(AudioSegment.from_mp3(self.path + self.start_str + '-' + sample_time + '.mp3'))

            # if combine_sample:
            if verbose:
                print('Duration of the sample audio: %.2f'%sample_audio.duration_seconds)
                print('Sampling rate of the sample audio: %d'%sample_audio.frame_rate)
            self.sample_rate = sample_audio.frame_rate
            self.sound_track = np.array(sample_audio.get_array_of_samples(), dtype = np.int16)/32678
            self.sound_length = sample_audio.duration_seconds
            # else:
            #     print('Sampling rate of the sample audio: %d'%sample_audio[0].frame_rate)
            #     self.sample_rate = sample_audio[0].frame_rate
        except:
            print('No sample audio loaded')
            self.sample_rate = 0
            self.sound_track = np.array([])
            self.sound_length = 0
            pass

    def getAudio(self):
        # if self.combine_sample:
        return self.sound_track

    def Visualizer(self, tmin, tmax):
        if tmax - tmin > 300:
            warnings.warn('The audio clip to visualize is longer than 300 seconds, which might induce overflow problem. Try a shorter clip')

        t = np.linspace(tmin, tmax, (tmax - tmin) * self.sample_rate)
        plt.figure(figsize=(18,6))
        ax1 = plt.subplot(111)
        ax1.plot(t, self.sound_track[int(tmin * self.sample_rate): int(tmax * self.sample_rate)])
        ax1.set_xlabel('Elapsed time since audio starts/sec')
        ax1.set_ylabel('Audio/ Volt')
        ax1.set_ylim(-1, 1)
        plt.show()
        return