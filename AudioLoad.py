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


class AudioLoad:
    def __init__(self, Airport = 'KJFK', 
                 root_dir = "/AudioDownload/Tower/", 
                 File_Type = 'Twr', 
                 Anal_Date = 'Apr-28-2017', 
                 Anal_Sample_Time = ['1800Z', '1830Z']):
        # File_Type: 'Twr', 'ROBER', 'Final', 'CAMRN'
        self.path = os.getcwd() + root_dir
        self.start_str = Airport + '-' + File_Type + '-' + Anal_Date
        print('Analyzed File Type and Date: %s'%self.start_str)
        # self.daily_file_list = [filename for filename in os.listdir(self.path) if filename.startswith(self.start_str)]
        # self.daily_file_list.sort()
        # self.sample_audio_file_list = []
        self.sample_audio = AudioSegment.empty()
        # self.combine_sample = combine_sample
        # if self.combine_sample:
        #     self.sample_audio = AudioSegment.empty()
        # else:
        #     self.sample_audio = []

        try:
            for sample_time in Anal_Sample_Time:
                # self.sample_audio_file_list.append(self.start_str + '-' + sample_time + '.mp3')
                # if self.combine_sample:
                self.sample_audio += AudioSegment.from_mp3(self.path + self.start_str + '-' + sample_time + '.mp3')
                # else:
                #     self.sample_audio.append(AudioSegment.from_mp3(self.path + self.start_str + '-' + sample_time + '.mp3'))

            # if combine_sample:
            print('Duration of the sample audio: %.2f'%self.sample_audio.duration_seconds)
            print('Sampling rate of the sample audio: %d'%self.sample_audio.frame_rate)
            self.sample_rate = self.sample_audio.frame_rate
            self.sound_track = np.array(self.sample_audio.get_array_of_samples(), dtype = np.int16)/32678
            # else:
            #     print('Sampling rate of the sample audio: %d'%self.sample_audio[0].frame_rate)
            #     self.sample_rate = self.sample_audio[0].frame_rate
        except:
            print('No sample audio loaded')
            self.sample_rate = 0
            self.sound_track = np.array([])
            pass

    def getAudio(self):
        # if self.combine_sample:
        return self.sample_audio, self.sound_track

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