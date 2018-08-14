# -*- coding: utf-8 -*-
# @Author: Yulin Liu
# @Date:   2018-08-13 14:23:44
# @Last Modified by:   Yulin Liu
# @Last Modified time: 2018-08-14 11:11:18

import numpy as np
import os
from pydub import AudioSegment
import pandas as pd
from .utils import baseline_time
"""
load audio data into memory and extract features
"""


def audio_data_loader(file_list, verbose = True):
    """
    load audio data
    file_list specifies the path to data file
    due to the large size of audio data, usually the size of the list is small (<10)
    """

    sample_audio = AudioSegment.empty()
    try:
        for file_name in file_list:
            # sample_audio_file_list.append(self.start_str + '-' + sample_time + '.mp3')
            if verbose:
                print('Analyzed File: %s'%file_name)
            sample_audio += AudioSegment.from_mp3(file_name)

        if verbose:
            print('Duration of the sample audio: %.2f'%sample_audio.duration_seconds)
            print('Sampling rate of the sample audio: %d'%sample_audio.frame_rate)
        sample_rate = sample_audio.frame_rate
        sound_track = (np.array(sample_audio.get_array_of_samples(), dtype = np.int16)/32678).astype(np.float32)
        sound_length = sample_audio.duration_seconds
    except:
        print('No sample audio loaded')
        sample_rate = 0
        sound_track = np.array([], dtype = np.float32)
        sound_length = 0
        pass
    return sound_track, sample_rate, sound_length

def TTF_data_loader(file_list, airport = 'JFK'):

    ## valid_cp and cp_to_channel_dict are currently hard-coded, which could be further improved in the future
    valid_cp = ['CAMRN', 'STW', 'ESJAY', 'KEAVR','TINNI', 'OTPIF', '(NorthWest)', 'FALTY','HESIN', 
            'ROBER','EPARE', '(East)']

    cp_to_channel_dict = {'CAMRN': 'CAMRN', 
                          'STW': 'CAMRN', 
                          'ESJAY': 'CAMRN', 
                          'KEAVR': 'CAMRN',
                          'TINNI': 'CAMRN', 
                          'OTPIF': 'CAMRN', 
                          '(NorthWest)': 'CAMRN', 
                          'FALTY': 'CAMRN',
                          'HESIN': 'CAMRN', 
                          'ROBER': 'ROBER',
                          'EPARE': 'ROBER', 
                          '(East)': 'ROBER'}

    # load TTF data
    if len(file_list) == 1:
        N90 = pd.read_csv(file_list[0], 
                          usecols=[0, 2, 3, 4, 5, 6, 7, 16, 17, 22, 31, 32, 33, 34, 35, 36])
    else:
        N90 = []
        for file_name in file_list:
            N90 += [pd.read_csv(file_name, 
                                usecols=[0, 2, 3, 4, 5, 6, 7, 16, 17, 22, 31, 32, 33, 34, 35, 36])]
        N90 = pd.concat(N90)

    # Preprocessing N90
    # Filter all flights into JFK
    N90_jfk = N90.loc[N90.Airport == 'JFK'].reset_index(drop = True)
    N90_jfk_topcp = N90_jfk.loc[N90_jfk.CornerPost.isin(valid_cp)].reset_index(drop = True)
    N90_jfk_topcp['channel'] = N90_jfk_topcp.CornerPost.replace(cp_to_channel_dict)

    N90_jfk_topcp['Date(UTC) '] = pd.to_datetime(N90_jfk_topcp['Date(UTC) '], 
                                             infer_datetime_format=True,
                                             errors = 'coerce')
    N90_jfk_topcp['Event Date/Time(UTC)'] = pd.to_datetime(N90_jfk_topcp['Event Date/Time(UTC)'], 
                                                           infer_datetime_format=True, 
                                                           errors = 'coerce')
    N90_jfk_topcp['CPPassTime(UTC)'] = pd.to_datetime(N90_jfk_topcp['CPPassTime(UTC)'], 
                                                      infer_datetime_format=True, 
                                                      errors = 'coerce')
    N90_jfk_topcp['Stop Date and Time(UTC)'] = pd.to_datetime(N90_jfk_topcp['Stop Date and Time(UTC)'], 
                                                              infer_datetime_format=True, 
                                                              errors = 'coerce')
    N90_jfk_topcp = N90_jfk_topcp.dropna().reset_index(drop = True)

    N90_jfk_topcp['evtime_elapsed'] = (N90_jfk_topcp['Event Date/Time(UTC)'] - baseline_time).apply(lambda x: x.total_seconds())
    N90_jfk_topcp['cptime_elapsed'] = (N90_jfk_topcp['CPPassTime(UTC)'] - baseline_time).apply(lambda x: x.total_seconds())
    N90_jfk_topcp['stptime_elapsed'] = (N90_jfk_topcp['Stop Date and Time(UTC)'] - baseline_time).apply(lambda x: x.total_seconds())

    return N90_jfk_topcp, N90