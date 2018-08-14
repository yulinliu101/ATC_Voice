# -*- coding: utf-8 -*-
# @Author: Yulin Liu
# @Date:   2018-08-14 14:02:44
# @Last Modified by:   Yulin Liu
# @Last Modified time: 2018-08-14 16:12:07

from utils import *
from utils_data_loader import *
from utils_feature_extractor import *
from utils_info_matrix import *
from utils_match_TTF import *
from utils_VAD import *


# load audio data (48 * 3 files) per day
# load TTF data (daily)

camrn_file_name_list = audio_file_header_collector(year = 2018, 
                                                  month = 4, 
                                                  day = 1, 
                                                  start_hour = 0, 
                                                  end_hour = 23, 
                                                  channel = 'CAMRN', 
                                                  airport = 'KJFK')

_ = gather_info_matrix(root_dir = 'debugger/VoiceData/CAMRN/20180401/',
                       file_list = camrn_file_name_list,
                       channel = 'CAMRN')

rober_file_name_list = audio_file_header_collector(year = 2018, 
                                                  month = 4, 
                                                  day = 1, 
                                                  start_hour = 0, 
                                                  end_hour = 23, 
                                                  channel = 'ROBER', 
                                                  airport = 'KJFK')
_ = gather_info_matrix(root_dir = 'debugger/VoiceData/ROBER/20180401/',
                       file_list = rober_file_name_list,
                       channel = 'ROBER')

tower_file_name_list = audio_file_header_collector(year = 2018, 
                                                  month = 4, 
                                                  day = 1, 
                                                  start_hour = 0, 
                                                  end_hour = 23, 
                                                  channel = 'Twr', 
                                                  airport = 'KJFK')
_ = gather_info_matrix(root_dir = 'debugger/VoiceData/Tower/20180401/',
                       file_list = tower_file_name_list,
                       channel = 'Twr')


tmp_file_zipper(target_path = 'tmp/', 
                dump_to_zipfile = 'debugger/voice_feature_20180401.zip',
                clean_target_path = True)

pointer_file_names = 'debugger/voice_feature_20180401.zip'
camrn_info = load_channel_features(pointer_file_names, channel = 'CAMRN')
rober_info = load_channel_features(pointer_file_names, channel = 'ROBER')
twr_info = load_channel_features(pointer_file_names, channel = 'Twr')
print(camrn_info.shape)
print(rober_info.shape)
print(twr_info.shape)






# # to run in console
# if __name__ == '__main__':
#     import click
#     # Use click to parse command line arguments
#     @click.command()
#     @click.option('--train_or_predict', type=str, default='train', help='Train the model or predict model based on input')
#     @click.option('--config', default='configs/encoder_decoder_nn.ini', help='Configuration file name')
#     @click.option('--name', default=None, help='Path for retored model')
#     @click.option('--train_from_model', type=bool, default=False, help='train from restored model')

#     # for prediction
#     @click.option('--test_data', default='../data/test/test_data.csv', help='test data path')

#     # Train RNN model using a given configuration file
#     def main(config='configs/encoder_decoder_nn.ini',
#              name = None,
#              train_from_model = False,
#              train_or_predict = 'train',
#              test_data = '../data/test_data.csv'):
#         try:        
#             os.mkdir('log')     
#         except:     
#             pass
#         log_name = '{}_{}_{}'.format('log/log', train_or_predict, time.strftime("%Y%m%d-%H%M%S"))
#         logging.basicConfig(level=logging.DEBUG,
#                             format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
#                             filename=log_name + '.log',
#                             filemode='w')
#         global logger
#         logger = logging.getLogger(os.path.basename(__file__))
#         consoleHandler = logging.StreamHandler()
#         consoleHandler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
#         logger.addHandler(consoleHandler)

#         # create the Tf_train_ctc class
#         if train_or_predict == 'train':
#             tmpBinary = True
#         elif train_or_predict == 'predict':
#             tmpBinary = False
#         else:
#             raise ValueError('train_or_predict not valid')
#         tf_train = trainRNN(conf_path=config,
#                             model_name=name, 
#                             sample_traj = not tmpBinary)
#         if tmpBinary:
#             # run the training
#             tf_train.run_model(train_from_model = train_from_model)
#         else:
#             tf_train.run_model(train_from_model = False,
#                                test_data_start_track = test_data) 
#     main()