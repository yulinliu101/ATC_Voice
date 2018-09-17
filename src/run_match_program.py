# -*- coding: utf-8 -*-
# @Author: Yulin Liu
# @Date:   2018-08-14 14:02:44
# @Last Modified by:   Yulin Liu
# @Last Modified time: 2018-09-04 15:53:01

# to run in console

"""
How this works:
python run_match_program.py --year 2018 --month 8 --day 18 --start_hour 5 --end_hour 5 --next_day True --root_dir_audio debugger/VoiceData/ --root_dir_ttf debugger/CurrentData/ --dir_to_audio_feature debugger/VoiceFeature/ --dir_to_processed_ttf debugger/NewTTF/
"""


if __name__ == '__main__':
    import click
    from utils import *
    from utils_data_loader import *
    from utils_feature_extractor import *
    from utils_info_matrix import *
    from utils_match_TTF import *
    from utils_VAD import *
    # Use click to parse command line arguments
    @click.command()
    @click.option('--year', type=int, default=2018, help='year of interest')
    @click.option('--month', type=int, default=8, help='month of interest')
    @click.option('--day', type=int, default=18, help='day of interest')
    @click.option('--start_hour', type=int, default=23, help='start hour of voice data to process')
    @click.option('--end_hour', type=int, default=3, help='end hour of voice data to process')
    @click.option('--next_day', type=bool, default=True, help='is the end hour for the next dat? if yes specify true')

    # dirs
    @click.option('--root_dir_audio', type=str, default='debugger/VoiceData/', help='root directory to the audio data. e.g., VoiceData/')
    @click.option('--root_dir_ttf', type=str, default='debugger/CurrentData/', help='root directory to the TTF data. e.g., CurrentData/')
    @click.option('--dir_to_audio_feature', type=str, default='debugger/VoiceFeature/', help='directory to dump the processed audio features')
    @click.option('--dir_to_processed_ttf', type=str, default='debugger/NewTTF/', help='directory to dump the processed TTF with audio features')

    def main(year = 2018,
             month = 8,
             day = 1,
             start_hour = 5,
             end_hour = 5,
             next_day = True,
             root_dir_audio = 'debugger/VoiceData/',
             root_dir_ttf = 'debugger/CurrentData/',
             dir_to_audio_feature = 'debugger/VoiceFeature/',
             dir_to_processed_ttf = 'debugger/NewTTF/'):
        # # load audio data (48 * 3 files) per day
        # # load TTF data (daily)
        import os
        yyyymmdd = '%d%s%s'%(year, str(month).zfill(2), str(day).zfill(2))
        print('=============================================================================')
        print('Start Processing Audios for %s'%yyyymmdd)
        print('Start hour: %d'%start_hour)
        print('End hour: %d'%end_hour)
        if next_day:
            print('End hour is on the next day')
        print('=============================================================================\n')
        try:
            os.makedirs(dir_to_processed_ttf)
        except:
            pass
        try:
            os.makedirs(dir_to_audio_feature)
        except:
            pass
        print('Directories made for audio features and new TTF data (with audio features)')
        print('Voice features will be dumped to %s'%dir_to_audio_feature)
        print('New TTF with audio features will be dumped to %s'%dir_to_processed_ttf)

        print('=============================================================================')
        print('Start extracting voice features...')
        print('=============================================================================\n')

        camrn_file_name_list = audio_file_header_collector(year = year, 
                                                           month = month, 
                                                           day = day, 
                                                           start_hour = start_hour, 
                                                           end_hour = end_hour, 
                                                           channel = 'CAMRN', 
                                                           airport = 'KJFK',
                                                           nextday_end_hour = next_day)       

        rober_file_name_list = audio_file_header_collector(year = year, 
                                                           month = month, 
                                                           day = day, 
                                                           start_hour = start_hour, 
                                                           end_hour = end_hour, 
                                                           channel = 'ROBER', 
                                                           airport = 'KJFK',
                                                           nextday_end_hour = next_day)
        
        tower_file_name_list = audio_file_header_collector(year = year, 
                                                           month = month, 
                                                           day = day, 
                                                           start_hour = start_hour, 
                                                           end_hour = end_hour, 
                                                           channel = 'Twr', 
                                                           airport = 'KJFK',
                                                           nextday_end_hour = next_day)

        _ = gather_info_matrix(root_dir = root_dir_audio,
                               file_list = rober_file_name_list,
                               channel = 'ROBER')
        _ = gather_info_matrix(root_dir = root_dir_audio,
                               file_list = camrn_file_name_list,
                               channel = 'CAMRN')
        _ = gather_info_matrix(root_dir = root_dir_audio,
                               file_list = tower_file_name_list,
                               channel = 'Twr')

        tmp_file_zipper(target_path = 'tmp/', 
                        dump_to_zipfile = '%s/voice_feature_%s.zip'%(dir_to_audio_feature, yyyymmdd),
                        clean_target_path = True,
                        brutal = True)

        pointer_file_names = '%s/voice_feature_%s.zip'%(dir_to_audio_feature, yyyymmdd)
        camrn_info = load_channel_features(pointer_file_names, channel = 'CAMRN')
        rober_info = load_channel_features(pointer_file_names, channel = 'ROBER')
        twr_info = load_channel_features(pointer_file_names, channel = 'Twr')
        print(camrn_info.shape)
        print(rober_info.shape)
        print(twr_info.shape)


        # start matching
        file_list = TTF_file_header_collector(year = year, month = month, start_day=day, end_day=day)
        processed_TTF, original_TTF = TTF_data_loader(root_dir = root_dir_ttf, file_list = file_list, airport = 'JFK')
        data_val_arr = get_TTF_array_from_df(processed_TTF)
        test_feature_space = augment_voice_feature(data_val_arr, camrn_info, rober_info, twr_info)
        output_df = merge_with_original_TTF(processed_TTF, original_TTF, test_feature_space)

        dump_to_csv('%s/%s.csv'%(dir_to_processed_ttf, yyyymmdd), output_df)

    main()