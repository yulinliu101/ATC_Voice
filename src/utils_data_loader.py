# -*- coding: utf-8 -*-
# @Author: Yulin Liu
# @Date:   2018-08-13 14:23:44
# @Last Modified by:   Lu Dai
# @Last Modified time: 2018-09-17 15:50:44

import numpy as np
import os
from pydub import AudioSegment
import pandas as pd
from utils import baseline_time
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

def TTF_data_loader(root_dir, file_list, airport = 'JFK', compression = True):

    ## valid_cp and cp_to_channel_dict are currently hard-coded, which could be improved in the future
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
    if compression:
      if len(file_list) == 1:
        N90 = pd.read_csv(root_dir + '/' + file_list[0] + '.gz', 
                 compression='gzip', header=0, sep=',', quotechar='"')
      else:
        N90 = []
        for file_name in file_list:
          N90 += [pd.read_csv(root_dir + '/' + file_name + '.gz', compression='gzip', header=0, sep=',', quotechar='"')]
        N90 = pd.concat(N90)
    else:
      if len(file_list) == 1:
          N90 = pd.read_csv(root_dir + '/' + file_list[0])
      else:
          N90 = []
          for file_name in file_list:
              N90 += [pd.read_csv(root_dir + '/' + file_name)]
          N90 = pd.concat(N90)

    # Preprocessing N90
    # Filter all flights into arg "airport"
    N90_jfk = N90.loc[N90.AIRPORT == airport].reset_index(drop = True)
    N90_jfk_topcp = N90_jfk.loc[N90_jfk.CORNER_POST.isin(valid_cp)].reset_index(drop = True)
    N90_jfk_topcp['channel'] = N90_jfk_topcp.CORNER_POST.replace(cp_to_channel_dict)

    N90_jfk_topcp['DATE_UTC'] = pd.to_datetime(N90_jfk_topcp['DATE_UTC'], 
                                             infer_datetime_format=True,
                                             errors = 'coerce')
    N90_jfk_topcp['EVENT_DATE_TIME_UTC'] = pd.to_datetime(N90_jfk_topcp['EVENT_DATE_TIME_UTC'], 
                                                           infer_datetime_format=True, 
                                                           errors = 'coerce')
    N90_jfk_topcp['CORNER_POST_PASS_TIME_UTC'] = pd.to_datetime(N90_jfk_topcp['CORNER_POST_PASS_TIME_UTC'], 
                                                      infer_datetime_format=True, 
                                                      errors = 'coerce')
    N90_jfk_topcp['STOP_DATE_AND_TIME_UTC'] = pd.to_datetime(N90_jfk_topcp['STOP_DATE_AND_TIME_UTC'], 
                                                              infer_datetime_format=True, 
                                                              errors = 'coerce')
    N90_jfk_topcp = N90_jfk_topcp.dropna().reset_index(drop = True)

    N90_jfk_topcp['evtime_elapsed'] = (N90_jfk_topcp['EVENT_DATE_TIME_UTC'] - baseline_time).apply(lambda x: x.total_seconds())
    N90_jfk_topcp['cptime_elapsed'] = (N90_jfk_topcp['CORNER_POST_PASS_TIME_UTC'] - baseline_time).apply(lambda x: x.total_seconds())
    N90_jfk_topcp['stptime_elapsed'] = (N90_jfk_topcp['STOP_DATE_AND_TIME_UTC'] - baseline_time).apply(lambda x: x.total_seconds())

    return N90_jfk_topcp, N90
