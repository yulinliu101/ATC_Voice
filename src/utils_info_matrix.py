# -*- coding: utf-8 -*-
# @Author: Yulin Liu
# @Date:   2018-08-13 16:09:35
# @Last Modified by:   Yulin Liu
# @Last Modified time: 2018-08-13 17:45:50

from utils import baseline_time
import numpy as np
from utils_data_loader import audio_data_loader
from utils_feature_extractor import AudioFeatures
from utils_VAD import voice_activity_detector

# import AudioLoad
# import AudioFeatures
# import AudioActDet
# import AudioSegmentation
# from energy_helper import _energy_helper
# import scipy
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import os
# import time
# import pandas as pd
# import datetime
# from sklearn.decomposition import PCA
# import statsmodels.api as sm 
# import math
# from itertools import groupby, chain, count
# from operator import itemgetter

def gather_info_matrix(file_list, 
                       dump_to_dir = 'Data/AudioMatrix/CAMRN/',
                       verbose = False):
    """
    from MatchAudioTTF import gather_info_matrix
    path = "F:/AudioData/CAMRN/"
    Call gather_info_matrix(root_dir = path,
                            file_list = [path + i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and 'Jul' in i],
                            dump_to_dir = 'Data/AudioMatrix/CAMRN/',
                            verbose = False)
    to gather info matrices from audio files located at root_dir + file_list. Also dump file to the directory specified by "dump_to_dir"
    """
    # root_dir = "F:/AudioData/CAMRN/"
    # file_list = [path + i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and 'Apr' in i]

    path = root_dir
    file_idx = 0
    for filename in file_list:
        file_idx += 1
        st = time.time()
        # filename = j.replace(path, '').replace('.mp3','')
        try:
            sound_track, sample_rate, sound_length = audio_data_loader([filename])
            FeatureClass = AudioFeatures.AudioFeatures(sound_track, 
                                                       sample_rate, 
                                                       sound_length,
                                                       nperseg = 512,
                                                       overlap_rate = 8, 
                                                       nfft = 1024, 
                                                       fbank_hfreq = None,
                                                       pre_emphasis = True)
            freqs, time_ins, Pxx = FeatureClass.stft(power_mode = 'PSD')
            energy  = FeatureClass.Energy(boundary = None)
            silence_seg, silence_seg_2d, idx_act = voice_activity_detector(sec_to_bin = FeatureClass.sec_to_bin, 
                                                                           time_ins = time_ins, 
                                                                           Pxx = Pxx,
                                                                           power_threshold = 0,
                                                                           silence_sec = 0.1, 
                                                                           mvg_point = 5)

            enersum = []
            enermax = []
            enermin = []
            ener25 = []
            enermed = []
            ener75 = []
            ener90 = []
            enermea = []
            enerstd = []
            
            sectobin = []
            actidxs = []
            actrate = []
            
            time_list = np.arange(0, len(energy), VADClass.sec_to_bin)
            if len(time_list) < 1800:
                print(filename, 'not enough time')
            else:
                for i in time_list:
                    sectobin.append(VADClass.sec_to_bin)
                    part = energy[int(i):int(i+VADClass.sec_to_bin)]
                    enersum.append(sum(part))
                    enermax.append(max(part))
                    enermin.append(min(part))
                    
                    ener25.append(np.percentile(part, 25))
                    enermed.append(np.median(part))
                    ener75.append(np.percentile(part, 75))
                    ener90.append(np.percentile(part, 90))
                    
                    enermea.append(np.mean(part))
                    enerstd.append(np.std(part))

                    actidx = idx_act[np.where((idx_act < i+VADClass.sec_to_bin) & (idx_act >= i))]#.tolist()
                    idx_range = []

                    groups = groupby(actidx, key = lambda q,x = count(): q-next(x))
                    tmp = [list(g) for k, g in groups]
                    for item in tmp:
                        idx_range.append(item[0])
                        idx_range.append(item[-1])
        #                 group = map(itemgetter(1), g)
        #                 idx_range.append(group[0])
        #                 idx_range.append(group[-1])
                    actidxs.append(idx_range)
                    actrate.append(len(actidx)/(int(i+VADClass.sec_to_bin) - int(i)))

                lens = np.array([len(item) for item in actidxs])
                mask = lens[:,None] > np.arange(lens.max())
                out = np.full(mask.shape,-1) # Fill with -1
                out[mask] = np.concatenate(actidxs)
                result = np.array([enersum, enermax, enermin, ener25, enermed, ener75, ener90, enermea, enerstd, sectobin, actrate])
                info = np.concatenate((result, out.T), axis = 0)
                info = info.astype(np.float32)
                if dump_to_dir is not None:
                    np.save(dump_to_dir + filename, info.T)
                if file_idx % 100 == 0:
                    print('processing file %d'%file_idx)
                if verbose:
                    print('Matrix Dimension: ', info.shape)
                    print('Computation Time: ', time.time()-st)
                    print('********************************************************************')
        except IndexError:
            print(filename, "is not available.")
    return 