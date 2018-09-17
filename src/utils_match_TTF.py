# -*- coding: utf-8 -*-
# @Author: Yulin Liu
# @Date:   2018-08-14 12:23:01
# @Last Modified by:   Yulin Liu
# @Last Modified time: 2018-09-04 23:30:56

import numpy as np
import pandas as pd

def get_TTF_array_from_df(TTF_df):
    TTF_df['Index'] = TTF_df.index.values
    data_val_array = TTF_df[['Index','evtime_elapsed', 'cptime_elapsed', 'stptime_elapsed', 'channel']].values
    return data_val_array

def voice_congestion_stat_helper(info_matrices):
    """
    Extract frequency congestion statistics from info matrices
    """
    # combine consective active indices into one for every 30-minute voice clip
    row, col, ver = np.where(info_matrices[:, :, 12:] != -1)
    active_index = info_matrices[:, :, 12:][row, col, ver].reshape(-1, 2).astype(np.int)
    active_index = np.split(active_index, np.cumsum(np.bincount(row))//2)[:-1]

    new_active_index = []
    for element in active_index:
        head = -99
        tail = -99
        tmp_active_index = []
        for tmp_row in element:
            if tmp_row[0] != tail + 1:
                tmp_active_index.append([head, tail])
                head = tmp_row[0]
                tail = tmp_row[1]
            else:
                tail = tmp_row[1]
        tmp_active_index.append([head, tail])
        tmp_active_index = np.array(tmp_active_index)
        tmp_active_index = tmp_active_index[1:]
        new_active_index.append(tmp_active_index) # list of np arrays, each array is the merged st idx for a 30-min period

    # Compute congestion statistics: lambda and mu
    active_index_to_time = []
    start_tape_time = info_matrices[:, 0, 0]
    i = 0
    for element in new_active_index:
        active_index_to_time.append(start_tape_time[i] + element/info_matrices[i, 0, 10])
        i += 1
    active_index_to_time = np.concatenate(active_index_to_time)

    return active_index_to_time


def augment_voice_feature(data_val_array, 
                          camrn_info, 
                          rober_info, 
                          twr_info):
    '''
    data_val_array should be a numpy array with 5 columns:
    0| index to match back to pandas df
    1| event time elapsed from the baseline time
    2| corner post passing time elapsed from the baseline time
    3| stop time (landing time) elapsed from the baseline time
    4| utilized channels (inferred)
    
    '''
    act_idx_camrn_info = voice_congestion_stat_helper(camrn_info)
    act_idx_rober_info = voice_congestion_stat_helper(rober_info)
    act_idx_twr_info = voice_congestion_stat_helper(twr_info)

    cp_energy_25q_of_mean = []
    cp_energy_25q_of_90q = []

    twr_energy_25q_of_mean = []
    twr_energy_25q_of_90q = []

    cp_lambdas = []
    cp_mus = []
    twr_lambdas = []
    twr_mus = []
    for val_ in data_val_array:
        if val_[-1] == 'CAMRN':
            query = camrn_info[np.where((camrn_info[:, :, 0]>=val_[2])&(camrn_info[:, :, 0]<val_[1]))][:, 1:10]
            query_cong = act_idx_camrn_info[np.where((act_idx_camrn_info[:, 0]>=val_[2])&(act_idx_camrn_info[:, 0]<val_[1]))]

        elif val_[-1] == 'ROBER':
            query = rober_info[np.where((rober_info[:, :, 0]>=val_[2])&(rober_info[:, :, 0]<val_[1]))][:, 1:10]
            query_cong = act_idx_rober_info[np.where((act_idx_rober_info[:, 0]>=val_[2])&(act_idx_rober_info[:, 0]<val_[1]))]
        else:
            raise ValueError('Channel not found: %s'%val_[-1])
        query2 = twr_info[np.where((twr_info[:, :, 0]>=val_[1])&(twr_info[:, :, 0]<=val_[3]))][:, 1:10]
        query_cong2 = act_idx_twr_info[np.where((act_idx_twr_info[:, 0]>=val_[1])&(act_idx_twr_info[:, 0]<val_[3]))]

        query_size = query.shape[0]
        query_size2 = query2.shape[0]
        query_cong_size = query_cong.shape[0]
        query_cong_size2 = query_cong2.shape[0]
        
        if query_size > 0:
            cp_energy_25q_of_mean.append(np.percentile(query[:, 7], 25))
            cp_energy_25q_of_90q.append(np.percentile(query[:, 6], 25))
        else:
            cp_energy_25q_of_mean.append(np.nan)
            cp_energy_25q_of_90q.append(np.nan)
            
        if query_size2 > 0:
            twr_energy_25q_of_mean.append(np.percentile(query2[:, 7], 25))
            twr_energy_25q_of_90q.append(np.percentile(query2[:, 6], 25))
        else:
            twr_energy_25q_of_mean.append(np.nan)
            twr_energy_25q_of_90q.append(np.nan)

        if query_cong_size > 0:
            # indices.append(idx)
            cp_lambdas.append(query_cong_size/(val_[1] - val_[2]))
            cp_service_time = (query_cong[:, 1] - query_cong[:, 0])
            # service_time = (query_cong[:, 1] - query_cong[:, 0]).tolist()
            # service_times.append(service_time)
            cp_mus.append(np.mean(cp_service_time))
            # cp_total_durs.append(np.sum(cp_service_time))
        else:
            cp_lambdas.append(np.nan)
            cp_mus.append(np.nan)

        if query_cong_size2 > 0:
            # indices.append(idx)
            twr_lambdas.append(query_cong_size2/(val_[3] - val_[1]))
            twr_service_time = (query_cong2[:, 1] - query_cong2[:, 0])
            # service_time = (query_cong[:, 1] - query_cong[:, 0]).tolist()
            # service_times.append(service_time)
            twr_mus.append(np.mean(twr_service_time))
            # cp_total_durs.append(np.sum(cp_service_time))
        else:
            twr_lambdas.append(np.nan)
            twr_mus.append(np.nan)

            
        if val_[0] % 600 == 0:
            print('================== finished flight %d =================='%val_[0])
        
        
    feature_space = np.array([data_val_array[:, 0], 
                              cp_energy_25q_of_mean, 
                              cp_energy_25q_of_90q, 
                              twr_energy_25q_of_mean, 
                              twr_energy_25q_of_90q,
                              cp_lambdas,
                              cp_mus,
                              twr_lambdas,
                              twr_mus], dtype = np.float32).T
    
    return feature_space

def merge_with_original_TTF(processed_TTF, original_TTF, feature_space):
    df_feature_space = pd.DataFrame(data = feature_space, columns=['Index', 
                                                                   'cp_energy_25q_of_mean', 
                                                                   'cp_energy_25q_of_90q', 
                                                                   'twr_energy_25q_of_mean', 
                                                                   'twr_energy_25q_of_90q',
                                                                   'cp_lambdas',
                                                                   'cp_mus',
                                                                   'twr_lambdas',
                                                                   'twr_mus'])
    tmp_df = processed_TTF[['Index', 'AIRCRAFT_ID', 'CKEY']].merge(df_feature_space, left_on='Index', right_on='Index', how = 'inner')
    output_df = original_TTF.merge(tmp_df[['AIRCRAFT_ID', 
                                           'CKEY', 
                                           'cp_energy_25q_of_mean', 
                                           'cp_energy_25q_of_90q', 
                                           'twr_energy_25q_of_mean', 
                                           'twr_energy_25q_of_90q',
                                           'cp_lambdas',
                                           'cp_mus',
                                           'twr_lambdas',
                                           'twr_mus']], on = ['AIRCRAFT_ID', 'CKEY'], how = 'left')
    return output_df

def dump_to_csv(path_to_csv, df_to_dump):
    df_to_dump.to_csv(path_to_csv, index = False)

