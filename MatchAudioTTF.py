from __future__ import division
import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from dateutil import parser
import re

global baseline_time 
baseline_time = parser.parse('01/01/2017 0:0:0')

def gather_info_matrix(root_dir,
                       file_list, 
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
    import AudioLoad
    import AudioFeatures
    import AudioActDet
    import AudioSegmentation
    from energy_helper import _energy_helper
    import scipy
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    import os
    import time
    import pandas as pd
    import datetime
    from sklearn.decomposition import PCA
    import statsmodels.api as sm 
    import math
    from itertools import groupby, chain, count
    from operator import itemgetter

    path = root_dir
    file_idx = 0
    for j in file_list:
        file_idx += 1
        st = time.time()
        filename = j.replace(path, '').replace('.mp3','')
        try:
            AudioClass = AudioLoad.AudioLoad(file_list = [j])
            FeatureClass = AudioFeatures.AudioFeatures(AudioLoad = AudioClass,
                                                   nperseg = 512,
                                                   overlap_rate = 8, 
                                                   nfft = 1024, 
                                                   fbank_hfreq = None,
                                                   pre_emphasis = True)
            freqs, time_ins, features = FeatureClass.stft(power_mode = 'PSD')
            energy  = FeatureClass.Energy(boundary = None)
            VADClass = AudioActDet.AudioActDet(FeatureClass)
            sil_seg, silence_seg_2d, idx_act = VADClass.detect_silence(power_threshold = 0,
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


class InfoMatrix:
    def __init__(self, 
                 file_list):

        '''
        file_list should be a list of file names (dirs) that has all info matrices
        '''
        self.file_list = file_list

    def load_all_info_matrices(self):
        load_files = []
        tmp_info_matrices = []

        max_len = 0
        for fname in file_names:
            if 'CAMRN' in fname and 'Jul' in fname:
                info_matrix = np.load('Data/AudioMatrix/CAMRN/'+fname)[:1800, :]
                if info_matrix.shape[0] != 1800:
                    pass
                else:
                    if info_matrix.shape[1] > max_len:
                        max_len = info_matrix.shape[1]
                    elap_time = (parser.parse(re.findall('Jul-\d\d-\d\d\d\d-\d\d\d\d',fname)[0]) - baseline_time).total_seconds()
                    elap_time = np.arange(1800, dtype=np.float32) + elap_time
                    info_matrix = np.insert(info_matrix, 0, elap_time, axis = 1)
                    tmp_info_matrices.append(info_matrix)
                    load_files.append(fname)
        info_matrices = []
        for element in tmp_info_matrices:
            element = np.pad(element, ((0,0), (0, max_len + 1 - element.shape[1])), mode = 'constant', constant_values = -1)
            info_matrices.append(element)

        info_matrices = np.array(info_matrices) # has the shape of [n_audio, 1800 (sec), 17]
        # # bound active rate
        info_matrices[:, :, 8] = np.minimum(1, info_matrices[:, :, 8])
        info_matrices[:, :, 8] = np.maximum(0, info_matrices[:, :, 8])

        return info_matrices, 