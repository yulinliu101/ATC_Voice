# -*- coding: utf-8 -*-
# @Author: Yulin Liu
# @Date:   2018-08-14 12:23:01
# @Last Modified by:   Yulin Liu
# @Last Modified time: 2018-08-23 15:28:19

import numpy as np
import pandas as pd

def get_TTF_array_from_df(TTF_df):
    TTF_df['Index'] = TTF_df.index.values
    data_val_array = TTF_df[['Index','evtime_elapsed', 'cptime_elapsed', 'stptime_elapsed', 'channel']].values
    return data_val_array

def augment_voice_feature(data_val_array, camrn_info, rober_info, twr_info):
    '''
    data_val_array should be a numpy array with 5 columns:
    0| index to match back to pandas df
    1| event time elapsed from the baseline time
    2| corner post passing time elapsed from the baseline time
    3| stop time (landing time) elapsed from the baseline time
    4| utilized channels (inferred)
    
    '''
    cp_energy_25q_of_mean = []
    cp_energy_25q_of_90q = []

    twr_energy_25q_of_mean = []
    twr_energy_25q_of_90q = []
    for val_ in data_val_array:
        if val_[-1] == 'CAMRN':
            query = camrn_info[np.where((camrn_info[:, :, 0]>=val_[2])&(camrn_info[:, :, 0]<val_[1]))][:, 1:10]
        elif val_[-1] == 'ROBER':
            query = rober_info[np.where((rober_info[:, :, 0]>=val_[2])&(rober_info[:, :, 0]<val_[1]))][:, 1:10]
        else:
            raise ValueError('Channel not found: %s'%val_[-1])
        query2 = twr_info[np.where((twr_info[:, :, 0]>=val_[1])&(twr_info[:, :, 0]<=val_[3]))][:, 1:10]
        query_size = query.shape[0]
        query_size2 = query2.shape[0]
        
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
            
        if val_[0] % 600 == 0:
            print('================== finished flight %d =================='%val_[0])
    
    feature_space = np.array([data_val_array[:, 0], 
                              cp_energy_25q_of_mean, 
                              cp_energy_25q_of_90q, 
                              twr_energy_25q_of_mean, 
                              twr_energy_25q_of_90q], dtype = np.float32).T
    
    return feature_space

def merge_with_original_TTF(processed_TTF, original_TTF, feature_space):
    df_feature_space = pd.DataFrame(data = feature_space, columns=['Index', 
                                                                   'cp_energy_25q_of_mean', 
                                                                   'cp_energy_25q_of_90q', 
                                                                   'twr_energy_25q_of_mean', 
                                                                   'twr_energy_25q_of_90q'])
    tmp_df = processed_TTF[['Index', 'AIRCRAFT_ID', 'CKEY']].merge(df_feature_space, left_on='Index', right_on='Index', how = 'inner')
    output_df = original_TTF.merge(tmp_df[['AIRCRAFT_ID', 
                                           'CKEY', 
                                           'cp_energy_25q_of_mean', 
                                           'cp_energy_25q_of_90q', 
                                           'twr_energy_25q_of_mean', 
                                           'twr_energy_25q_of_90q']], on = ['AIRCRAFT_ID', 'CKEY'], how = 'left')
    return output_df

def dump_to_csv(path_to_csv, df_to_dump):
    df_to_dump.to_csv(path_to_csv, index = False)

