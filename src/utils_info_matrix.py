# -*- coding: utf-8 -*-
# @Author: Yulin Liu
# @Date:   2018-08-13 16:09:35
# @Last Modified by:   Yulin Liu
# @Last Modified time: 2018-09-04 17:39:57

import numpy as np
from utils import baseline_time
from utils_data_loader import audio_data_loader
from utils_feature_extractor import AudioFeatures
from utils_VAD import voice_activity_detector
import zipfile
import os
from dateutil import parser
import re
import time
from itertools import groupby, chain, count
from operator import itemgetter

def gather_info_matrix(root_dir,
                       file_list,
                       channel,
                       dump_to_tmp = True,
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
    
    if dump_to_tmp:
        try:
            os.makedirs('tmp/%s'%channel)
        except:
            pass
    else:
        pass

    file_idx = 0
    for filename in file_list:
        file_idx += 1
        st = time.time()
        # filename = j.replace(path, '').replace('.mp3','')
        try:
            sound_track, sample_rate, sound_length = audio_data_loader([root_dir + filename])
            FeatureClass = AudioFeatures(sound_track, 
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
            
            time_list = np.arange(0, len(energy), FeatureClass.sec_to_bin)
            if len(time_list) < 1800:
                print(filename, 'not enough time')
            else:
                for i in time_list:
                    sectobin.append(FeatureClass.sec_to_bin)
                    part = energy[int(i):int(i+FeatureClass.sec_to_bin)]
                    enersum.append(sum(part))
                    enermax.append(max(part))
                    enermin.append(min(part))
                    
                    ener25.append(np.percentile(part, 25))
                    enermed.append(np.median(part))
                    ener75.append(np.percentile(part, 75))
                    ener90.append(np.percentile(part, 90))
                    
                    enermea.append(np.mean(part))
                    enerstd.append(np.std(part))

                    actidx = idx_act[np.where((idx_act < i+FeatureClass.sec_to_bin) & (idx_act >= i))]#.tolist()
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
                    actrate.append(len(actidx)/(int(i+FeatureClass.sec_to_bin) - int(i)))

                lens = np.array([len(item) for item in actidxs])
                mask = lens[:,None] > np.arange(lens.max())
                out = np.full(mask.shape,-1) # Fill with -1
                out[mask] = np.concatenate(actidxs)
                result = np.array([enersum, enermax, enermin, ener25, enermed, ener75, ener90, enermea, enerstd, sectobin, actrate])
                info = np.concatenate((result, out.T), axis = 0)
                info = info.astype(np.float32)
                ###################################################
                ## dump file to a temporary folder
                ###################################################
                if dump_to_tmp:
                    np.save('tmp/%s/%s.npy'%(channel, filename.split('/')[-1][:-4]), info.T)
                else:
                    pass

                if file_idx % 100 == 0:
                    print('processing file %d'%file_idx)
                if verbose:
                    print('Matrix Dimension: ', info.shape)
                    print('Computation Time: ', time.time()-st)
                    print('********************************************************************')
        except IndexError:
            print("%s is not available."%filename)
            return
    return info.T


def load_channel_features(file_pointer, channel = 'CAMRN'):
    # energy sum, max energy, min energy, median energy, mean energy, std energy, sec_to_bin, active rate
    # index of active st, index of active end (pairs)
    """
    original info matrix contains:
    [0|enersum, 
     1|enermax, 
     2|enermin, 
     3|ener25, 
     4|enermed, 
     5|ener75, 
     6|ener90, 
     7|enermea, 
     8|enerstd, 
     9|sectobin, 
     10|actrate, 
     11|(active rate index tuple)]
     
     returned info matrix adds the time stamps to the index 0
    """
    pointer_file_names = np.load(file_pointer)

    load_files = []
    tmp_info_matrices = []
    max_len = 0
    # for fname in file_names:
    for fname in pointer_file_names.keys():
        if 'KJFK-NY-App-%s'%channel in fname or 'KJFK-%s'%channel in fname:
            info_matrix = pointer_file_names[fname][:1800, :]
            if info_matrix.shape[0] != 1800:
                pass
            else:
                if info_matrix.shape[1] > max_len:
                    max_len = info_matrix.shape[1]
                elap_time = (parser.parse(re.findall('(?:Jan?|Feb?|Mar?|Apr?|May|Jun?|Jul?|Aug?|Sep?|Oct?|Nov?|Dec)-\d\d-\d\d\d\d-\d\d\d\d',fname)[0]) - baseline_time).total_seconds()
                elap_time = np.arange(1800, dtype=np.float32) + elap_time
                info_matrix = np.insert(info_matrix, 0, elap_time, axis = 1) # insert time stamps to index 0
                tmp_info_matrices.append(info_matrix)
                load_files.append(fname)
    info_matrices = []
    for element in tmp_info_matrices:
        element = np.pad(element, ((0,0), (0, max_len + 1 - element.shape[1])), mode = 'constant', constant_values = -1)
        info_matrices.append(element)

    info_matrices = np.array(info_matrices)
    # # bound active rate
    info_matrices[:, :, 11] = np.minimum(1, info_matrices[:, :, 11])
    info_matrices[:, :, 11] = np.maximum(0, info_matrices[:, :, 11])
    
    return info_matrices

    # lbda = np.array([N.shape[0] for N in new_active_index])/30/60 # N/sec
    # lbda = []
    # active_dur = []
    # j = 0
    # for N in new_active_index:
    #     j += 1
    #     lbda.append(N.shape[0]/30/60) # N/sec
    #     try:
    #         active_dur += ((N[:, 1] - N[:, 0])/info_matrices[0, 0, 7]).tolist() # sec
    #     except:
    #         pass
    # lbda = np.array(lbda)
    # active_dur = np.array(active_dur)
    # active_dur = active_dur[np.where(~np.isnan(active_dur))]
    # mu = np.mean(active_dur) # sec
    # print(mu)
    # print(np.std(active_dur))