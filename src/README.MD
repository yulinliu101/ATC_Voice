

```python
# @Author: Yulin Liu
# @Date:   2018-08-23 16:42:01
# @Last Modified by:   Yulin Liu
# @Last Modified time: 2018-08-23 18:14:23
```

# Overnight Augmenting Turn-To-Final (TTF) Dataset with ATC Audio Energy Features
This project provides various of APIs to augment the TTF data with ATC audio energy features on a regular basis. The overnight update should be done by specifying the time of interest and supporting datasets.

## Getting Started

These instructions will get you a copy of the project up and run on your local machine for development and testing purposes. Parallel computing is highly possible and recommended, however, it is not developed in this prototype.

From here on, all command will be assumed to be operated on a Linux machine. "$" is the default prompt.

### Platform
All APIs have been tested on Windows 10 with Anaconda (Python 3.6.4) and supporting packages installed; (by 08/22/2018)
All APIs have been tested on Linux (Ubuntu 16.04 LTS) with Anaconda (Python 3.5.5) and supporting packages installed; (by 08/23/2018)

### Prerequisites
Software:
	
- Anaconda with Python 3+ (https://www.anaconda.com/download/)
```
$ curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
$ sha256sum Anaconda3-5.0.1-Linux-x86_64.sh
$ bash Anaconda3-5.0.1-Linux-x86_64.sh
```

- ffmpeg (https://www.ffmpeg.org/)
```
$ sudo apt-get update
$ sudo apt-get install ffmpeg
```

Python packages:

Most python packages are installed along with Anaconda. However, you can always use ```pip``` to install them.
- click

    ```$ pip install click```
    
- dateutil

    ```$ pip install dateutil```
    
- pydub

    ```$ pip install pydub```

### Data Requirement

- ATC Voice Data

ATC voice data are downloaded from https://www.liveatc.net/ and must be stored in a specific format of directory tree. For JFK, there are 4 channels downloaded (CAMRN, ROBER, TOWER, FINAL), therefore, they should be stored as in parent_dir/CHANNEL/YYYYMMDD/*

Example:
```
$ ls parent_dir
> CAMRN ROBER Tower
$ ls CAMRN/
> 20180818 20180819
$ ls Tower/
> 20180818 20180819
$ ls Tower/20180818
> KJFK-Twr-Aug-18-2018-0000Z.mp3 KJFK-Twr-Aug-18-2018-0030Z.mp3
```

- N90 Turn-To-Final (TTF) Data

TTF Data should be stored as in text-delimited format (If they are compressed in a ZIP file, they must be extracted). General file name looks like 
```SVRSCSV_N90$TURNTOFINAL$_YYYYMMDD-YYYYMMDD_ALL```

## Running the tests

The following code will automatically match all flights in the TTF dataset from *08/18/2018 05:00:00 UTC* to *08/19/2018 05:00:00 UTC* with audio energy features in the same time period. The TTF dataset is stored in ```debugger/CurrentData/``` under the name of ```SVRSCSV_N90$TURNTOFINAL$_20180818-20180818_ALL```; the audio datasets are stored in the directory ```debugger/VoiceData/```; temporary voice energy features will be dumped to ```tmp/``` but will be automatically removed after the program terminates; All processed (extracted) voice features will then be compressed (as in zip file) and stored in ```debugger/VoiceFeature/```; The final TTF dataset with matched voice features will be dumped to ```debugger/NewTTF/```.

```$ python run_match_program.py --year 2018 --month 8 --day 18 --start_hour 5 --end_hour 5 --next_day True --root_dir_audio debugger/VoiceData/ --root_dir_ttf debugger/CurrentData/ --dir_to_audio_feature debugger/VoiceFeature/ --dir_to_processed_ttf debugger/NewTTF/```


### API highlights breakdown

#### src.utils


```python
# Collect file names for TTF datasets by specifying the datetime.
import numpy as np
from utils import TTF_file_header_collector
ttf_fname_list = TTF_file_header_collector(year = 2018, 
                                           month = 8, 
                                           start_day = 18, 
                                           end_day = 18)
assert ttf_fname_list == ['SVRSCSV_N90$TURNTOFINAL$_20180818-20180818_ALL']
```


```python
# Collect file names for audio datasets by specifying the datetime.
from utils import audio_file_header_collector
audio_file_list = audio_file_header_collector(year = 2018, 
                                          month = 8, 
                                          day = 18, 
                                          start_hour = 23, 
                                          end_hour = 1, 
                                          channel = 'Twr', 
                                          airport = 'KJFK',
                                          nextday_end_hour = True)
assert audio_file_list == ['Tower/20180818/KJFK-Twr-Aug-18-2018-2300Z.mp3',
                             'Tower/20180818/KJFK-Twr-Aug-18-2018-2330Z.mp3',
                             'Tower/20180819/KJFK-Twr-Aug-19-2018-0000Z.mp3',
                             'Tower/20180819/KJFK-Twr-Aug-19-2018-0030Z.mp3',
                             'Tower/20180819/KJFK-Twr-Aug-19-2018-0100Z.mp3',
                             'Tower/20180819/KJFK-Twr-Aug-19-2018-0130Z.mp3']
```


```python
# Compress all temporary voice feature files into a zipfile and dump them to dump_to_zipfile; Remove all files in tmp/ after the program terminate
from utils import tmp_file_zipper
tmp_file_zipper(target_path = 'tmp/', 
                dump_to_zipfile = 'debugger/voice_feature.zip',
                clean_target_path = True,
                brutal = True)
```

    brutal cleaned all tmp files!
    

#### src.utils_data_loader


```python
# Load audios file(s) from file list into memory and return numerical representation of the sound track(s)
from utils_data_loader import audio_data_loader
sound_track, sample_rate, sound_length = audio_data_loader(file_list=['debugger/VoiceData/%s'%i for i in audio_file_list[:2]], verbose = True)
```

    Analyzed File: debugger/VoiceData/Tower/20180818/KJFK-Twr-Aug-18-2018-2300Z.mp3
    Analyzed File: debugger/VoiceData/Tower/20180818/KJFK-Twr-Aug-18-2018-2330Z.mp3
    Duration of the sample audio: 3792.01
    Sampling rate of the sample audio: 22050
    


```python
# Load TTF file(s) from file list into memory and preprocess them
# processed_TTF should only contain flights into JFK and have three more columns indicating different time stamps
from utils_data_loader import TTF_data_loader
processed_TTF, original_TTF = TTF_data_loader(root_dir = 'debugger/CurrentData/', file_list = ttf_fname_list, airport = 'JFK')
assert np.array_equal(processed_TTF.AIRPORT.unique(), np.array(['JFK']))
assert np.array_equal(np.array(processed_TTF.columns[-3:]), np.array(['evtime_elapsed', 'cptime_elapsed', 'stptime_elapsed']))
```

#### src.utils_feature_extractor (computation bottleneck)


```python
# Build a class with all different voice features
from utils_feature_extractor import AudioFeatures
FeatureClass = AudioFeatures(sound_track, 
                           sample_rate, 
                           sound_length,
                           nperseg = 512,
                           overlap_rate = 8, 
                           nfft = 1024, 
                           fbank_hfreq = None,
                           pre_emphasis = True)
# conduct short time fourier transform
freqs, time_ins, Pxx = FeatureClass.stft(power_mode = 'PSD')
# compute energy
energy  = FeatureClass.Energy(boundary = None)
```

#### src.utils_VAD


```python
# Voice Activity Detection Algorithm
# Return indicies of silence period
from utils_VAD import voice_activity_detector
silence_seg, silence_seg_2d, idx_act = voice_activity_detector(sec_to_bin = FeatureClass.sec_to_bin, 
                                                               time_ins = time_ins, 
                                                               Pxx = Pxx,
                                                               power_threshold = 0,
                                                               silence_sec = 0.1, 
                                                               mvg_point = 5)

```

#### src.utils_info_matrix


```python
# call utils_VAD and utils_feature_extractor to generate voice features, and dump the features arrays to a temporary folder tmp/ as .npy files
from utils_info_matrix import gather_info_matrix
# gather_info_matrix(root_dir,
#                    file_list,
#                    channel,
#                    dump_to_tmp = True,
#                    verbose = False)
```


```python
# load compressed voice feature file into memory
from utils_info_matrix import load_channel_features
pointer_file_names = 'debugger/voice_feature_20180817.zip'
camrn_info = load_channel_features(pointer_file_names, channel = 'CAMRN')
rober_info = load_channel_features(pointer_file_names, channel = 'ROBER')
twr_info = load_channel_features(pointer_file_names, channel = 'Twr')
print(camrn_info.shape)
print(rober_info.shape)
print(twr_info.shape)
```

    (58, 1800, 20)
    (60, 1800, 20)
    (60, 1800, 20)
    

#### src.utils_match_TTF


```python
# extract useful info from processed TTF data and return a data array
from utils_match_TTF import get_TTF_array_from_df
data_val_arr = get_TTF_array_from_df(processed_TTF)
assert data_val_arr.shape[0] == processed_TTF.shape[0]
assert data_val_arr.shape[1] == 5
```


```python
# Augment TTF data with voice energy features
from utils_match_TTF import augment_voice_feature
test_feature_space = augment_voice_feature(data_val_arr, camrn_info, rober_info, twr_info)
```

    ================== finished flight 0 ==================
    ================== finished flight 600 ==================
    


```python
# Merge matched features with the original TTF data and return the outputs!
from utils_match_TTF import merge_with_original_TTF
output_df = merge_with_original_TTF(processed_TTF, original_TTF, test_feature_space)
assert output_df.shape[0] == original_TTF.shape[0]
assert output_df.shape[1] == original_TTF.shape[1] + 4
```

## Acknowledgments

* www.liveatc.net
