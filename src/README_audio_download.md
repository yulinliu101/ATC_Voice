```python
# @Author: Lu Dai
# @Date:   2018-08-25 08:19:32
# @Last Modified by:   Lu Dai
# @Last Modified time: 2018-08-25 20:43:26
```

# Air Traffic Control (ATC) Voice Data Downloading

This project is designed to download ATC voice data from https://www.liveatc.net/ and store in the local file system or upload to a FTP site. Data will be mounted under specific directory paths for later analysis.

With ATC voice data, you can:
- Pre-process audio data (.mp3) and extract energy features
- Voice Activity Detection (VAD)
- Generate voice information matrix according to the result of audio feature extraction and VAD
- Process N90 TTF data and match with voice information matrix

## Getting Started

This instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Platform

This code has been tested on OS X 10.11.6 with Anaconda (Python 3.6.6) and supporting packages installed (by 08/25/2018).

### Installing
What things you need to install and a series of packages that get a development environment running:

Software:
- Installable Python kits, and information about using Python, are available at https://www.python.org/.
- Anaconda with Python 3+ (https://www.anaconda.com/download/)

Python packages:
Most python packages are installed along with Anaconda. However, you can always install and update using  ```pip```.

| Package | Repository |
|---------|------------|
| click | https://github.com/pallets/click |
| dateutil | https://github.com/dateutil/dateutil |
| wget | https://pypi.org/project/wget/ |

## Data

### Data Source

The ATC voice data are the recordings of controller-pilot ATC communications. Live broadcasts are updated every 30 minutes by local time and archived for retrieval for up to 30 days. Users may not be able to use this code parsing the voice data one month ago. For JFK airport, there are four different channels being used when approaching: CAMRN, ROBER, Final, Tower. Each data file covers a 30-minute voice communication between the pilots and controllers and is stored as mp3 format. Authorization is needed from LiveATC to download all the voice data. 

### Directory Tree

For the purpose of overnight processing, this code stored the voice data in the following directory tree, either in the local file system or on the FTP site:
```
parent_dir/VoiceData/											              # directory to save voice data
├── CAMRN 														                  # Channel folder: CAMRN
│   ├── 20180801                                      	# Daily folder: 08/01/2018
│	  │		├── KJFK-NY-App-CAMRN-Aug-01-2018-0000Z.mp3 		# Voice data: 08/01/2018 00:00:00 - 00:30:00 (UTC)
│	  │		├── KJFK-NY-App-CAMRN-Aug-01-2018-0030Z.mp3 		# Voice data: 08/01/2018 00:30:00 - 01:00:00 (UTC)
│	  │		├── KJFK-NY-App-CAMRN-Aug-01-2018-0100Z.mp3 		# Voice data: 08/01/2018 01:00:00 - 01:30:00 (UTC)
│	  │		├── KJFK-NY-App-CAMRN-Aug-01-2018-0130Z.mp3 		# Voice data: 08/01/2018 01:30:00 - 02:00:00 (UTC)
│	  │		├── ......
│	  │		├── KJFK-NY-App-CAMRN-Aug-01-2018-2300Z.mp3 		# Voice data: 08/01/2018 23:00:00 - 23:30:00 (UTC)
│	  │		└── KJFK-NY-App-CAMRN-Aug-01-2018-2330Z.mp3 		# Voice data: 08/01/2018 23:30:00 - 00:00:00 (UTC)
│   ├── 20180802       											            # Daily folder: 08/02/2018
│   └── 20180803  												              # Daily folder: 08/03/2018
├── ROBER              		  									          # Channel folder: ROBER
├── Final             											            # Channel folder: Final
└── Tower  														                  # Channel folder: Tower

```

## How To

### Quick Start

Here is a quick code to get you started. After forking the repository and installing all the required software and packages, run the following line in the terminal. Make sure to set the `PYTHONPATH` to the repository top level directory.

```bash
python run_audio_download.py
```

This command uses the default setting:
```bash
python run_audio_download.py --year 2018 --month 8 --day 23 --start_hour 5 --end_hour 4 --nextday_end_hour True \
                            --channel CAMRN --airport KJFK --small_file_size 500\
                            --ftpurl ftp.atac.com --username sn2018 --Teardr2p --ftpfolder VoiceData/ \
                            --localfolder ATCAudio --audiourl http://archive.fmt2.liveatc.net/kjfk/ \
                            --mode local --verbose False
```

It will:
1. Automatically download the *CAMRN* voice recordings (.mp3) in *KJFK* airport from *08/23/2018 05:00:00 UTC* to *08/23/2018 05:00:00 UTC*.
2. Create a folder called "ATCAudio" in the parent directory and "CAMRN" folder under the "ATCAudio" directory. Under the "CAMRN" directory, create '20180823' daily folder for storing voice data from *08/23/2018 05:00:00 UTC* to *08/23/2018 23:30:00 UTC*, create '20180824' daily folder for storing voice data *08/24/2018 00:00:00 UTC* to *08/24/2018 04:30:00 UTC*.
3. Remove any invalid audio files that smaller than 500 bytes before downloading. If valid audio files have been downloaded, skip that.


### Configuration Parameters
The ATC voice data will be downloaded in a local file system and/or upload to a FTP site by specifying time of interest, frequency channel, airport ICAO code, and directories. "nextday_end_hour" parameter allows the code to download day-span voice data, in which time can be divided by users at random. Audio file may fail to be stored if a file already exists with the same name. A function was built to remove any invalid audio files that smaller than specified file size (in bytes). These options can be passed to the run_audio_download.py script; run
```bash
python run_audio_download.py --help
```
to find out more:
```
Usage: run_audio_download.py [OPTIONS]

Options:
  --year INTEGER                  4-digit year value as a decimal number
  --month INTEGER                 Month as a decimal number
  --day INTEGER                   Day of the month as a decimal number
  --start_hour INTEGER            Start hour (24-hour clock) of voice data as
                                  a decimal number, inclusive time
  --end_hour INTEGER              End hour (24-hour clock) of voice data as a
                                  decimal number, inclusive time
  --nextday_end_hour BOOLEAN      If start_hour and end_hour are on the same
                                  date, specified False, o/w True
  --channel [CAMRN|ROBER|Final|Tower]
                                  Frequency channels of interest
  --airport TEXT                  Airport ICAO code
  --small_file_size INTEGER       Remove invalid audio files that smaller than
                                  specified size (in bytes)
  --ftpurl TEXT                   FTP site that used to upload voice data
  --username TEXT                 Username for logging on FTP site
  --password TEXT                 Password for logging on FTP site
  --ftpfolder TEXT                FTP directory to upload the voice data
  --localfolder TEXT              Local directory to download the voice data
  --audiourl TEXT                 ATC audio source url
  --mode [both|FTP|local]
                                  Specifed 'both' to save voice data on both
                                  FTP and local folder
  --verbose BOOLEAN               True for printing download details for each
                                  audio file
  --help                          Show this message and exit.
```


### Work with FTP
There are three modes for downloading ATC voice data by passing value to *mode*:
- "local": ATC voice data will be stored under the specified *localfolder* directory only;
- "FTP": ATC voice data will be uploaded to the specified *ftpfolder* directory of *ftpurl* with access permit (*username*, *password*) only;
- "both": ATC voice data will be stored in the local file system and uploaded to the FTP site.

Note: If "Fail to access FTP site" error is given, please recheck the username and password, and make sure your IP is not blocked by the FTP server.

```bash
python run_audio_download.py --year 2018 --month 7 --day 31 --start_hour 17 --end_hour 16 --nextday_end_hour True \
                            --channel ROBER --airport KJFK --small_file_size 600 \
                            --ftpurl ftp.atac.com --username sn2018 --password Teardr2p --ftpfolder VoiceData/ \
                            --localfolder ATCAudio/ --audiourl http://archive.fmt2.liveatc.net/kjfk/ \
                            --mode both --verbose True
```
This code will:
1. Automatically download the *ROBER* voice recordings (.mp3) in *KJFK* airport from *07/31/2018 17:00:00 UTC* to *08/01/2018 17:00:00 UTC*.
2. In the local folder, create a folder called "ATCAudio" in the parent directory and "ROBER" folder under the "ATCAudio" directory. Under the "ROBER" directory, create '20180823' daily folder for storing voice data from *07/31/2018 17:00:00 UTC* to *07/31/2018 23:30:00 UTC*, create '20180801' daily folder for storing voice data *08/01/2018 00:00:00 UTC* to *08/01/2018 16:30:00 UTC*. Same data are uploaded to FTP site with the same directory tree.
3. Remove any invalid audio files that smaller than 600 bytes before downloading. If valid audio files have been downloaded, skip that.


### Multi-days Downloading

This script is designed for overnight processing on a daily basis. However, users can download multi-days voice data in one time by
1. Using loops to execute commands in one terminal:
```bash
for i in {1..5}; do \
	python run_audio_download.py --month 8 --day $i --start_hour 5 --end_hour 4 --channel CAMRN --mode local; \
done
```

2. Or opening five terminals (up to the number of cores your computer has):
```bash
python run_audio_download.py --month 8 --day 1 --start_hour 5 --end_hour 4 --channel CAMRN --mode local
```
```bash
python run_audio_download.py --month 8 --day 2 --start_hour 5 --end_hour 4 --channel CAMRN --mode local
```
```bash
python run_audio_download.py --month 8 --day 3 --start_hour 5 --end_hour 4 --channel CAMRN --mode local
```
```bash
python run_audio_download.py --month 8 --day 4 --start_hour 5 --end_hour 4 --channel CAMRN --mode local
```
```bash
python run_audio_download.py --month 8 --day 5 --start_hour 5 --end_hour 4 --channel CAMRN --mode local
```

Both methods will automatically download the *CAMRN* voice recordings (.mp3) in *KJFK* airport from *08/01/2018 5:00:00 UTC* to *08/06/2018 5:00:00 UTC* and stored in the local computer. The second way is recommended as long as the number of terminals do not exceed your computer cores.


## Contact

The code and issue tracker are hosted on Github https://github.com/yulinliu101/ATC_Voice/tree/master/src. You can file bugs in this github and/or email authors at dailu@berkeley.edu. 

## Acknowledgments

* https://www.liveatc.net/

