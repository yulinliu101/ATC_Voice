# -*- coding: utf-8 -*-
# @Author: Lu Dai
# @Date:   2018-08-14 14:02:44
# @Last Modified by:   Lu Dai
# @Last Modified time: 2018-08-25 15:02:45

"""
python run_audio_download.py --year 2018 --month 8 --day 23 --start_hour 5 --end_hour 4 --nextday_end_hour True \
                            --channel CAMRN --airport KJFK --small_file_size 500\
                            --ftpurl ftp.atac.com --username sn2018 --password Teardr2p --ftpfolder VoiceData/ \
                            --localfolder ATCAudio --audiourl http://archive.fmt2.liveatc.net/kjfk/ \
                            --mode local --verbose False
"""

if __name__ == '__main__':
	from utils import *
	from utils_audio_download import *
	import click
	import os
	import shutil
	from datetime import datetime, timedelta

	@click.command()
	@click.option('--year', type = int, default = 2018, help = '4-digit year value as a decimal number')
	@click.option('--month', type = int, default = 8, help = 'Month as a decimal number')
	@click.option('--day', type = int, default = 23, help = 'Day of the month as a decimal number')
	@click.option('--start_hour', type = int, default = 5, help = 'Start hour (24-hour clock) of voice data as a decimal number, inclusive time')
	@click.option('--end_hour', type = int, default = 4, help = 'End hour (24-hour clock) of voice data as a decimal number, inclusive time')
	@click.option('--nextday_end_hour', type = bool, default = True, help = 'If start_hour and end_hour are on the same date, specified False, o/w True')
	@click.option('--channel', default = "CAMRN", type = click.Choice(['CAMRN', 'ROBER', 'Final', 'Tower']), help = 'Frequency channels of interest')
	@click.option('--airport', type = str, default = "KJFK", help = 'Airport ICAO code')
	@click.option('--small_file_size', type = int, default = 500, help = 'Remove invalid audio files that smaller than specified size in bytes')

	@click.option('--ftpurl', type = str, default = 'ftp.atac.com', help = 'FTP site that used to upload voice data')
	@click.option('--username', type = str, default = 'sn2018', help = 'Username for logging on FTP site')
	@click.option('--password', type = str, default = 'Teardr2p', help = 'Password for logging on FTP site')
	@click.option('--ftpfolder', type = str, default = 'VoiceData/', help = 'FTP directory to upload the voice data')

	@click.option('--localfolder', type = str, default = 'ATCAudio/', help = 'Local directory to download the voice data')
	@click.option('--audiourl', type = str, default = 'http://archive.fmt2.liveatc.net/kjfk/', help = 'ATC audio source url')
	@click.option('--mode', default = 'local', type = click.Choice(['both', 'FTP','local']), 
		help = "Specifed 'both' to save voice data on both FTP and local folder")
	@click.option('--verbose', type = bool, default = False, help = 'True for printing download details for each audio file')

	def main(year = 2018,
			 month = 8,
			 day = 23,
			 start_hour = 5,
			 end_hour = 4,
			 channel = "CAMRN",
			 airport = 'KJFK',
			 small_file_size = 500,
			 ftpurl = 'ftp.atac.com',
			 username = 'sn2018',
			 password = 'Teardr2p',
			 ftpfolder = 'VoiceData/',
			 localfolder = 'ATCAudio/',
			 audiourl = 'http://archive.fmt2.liveatc.net/kjfk/',
			 nextday_end_hour = True,
			 mode = 'local',
			 verbose = False):


		yyyymmdd = '%d%s%s'%(year, str(month).zfill(2), str(day).zfill(2))
		start_time = datetime.strptime(yyyymmdd + str(start_hour), '%Y%m%d%H').strftime("%m/%d/%Y %H:%M:%S")
		if nextday_end_hour:
			end_time = datetime.strptime(yyyymmdd + str(end_hour+1), '%Y%m%d%H') + timedelta(days = 1)
		else:
			end_time = datetime.strptime(yyyymmdd + str(end_hour+1), '%Y%m%d%H')
		print('Downloading Audios from ', start_time, ' to ', end_time.strftime("%m/%d/%Y %H:%M:%S"))

		if mode == 'local':
			Audio_to_Local(year = year,
						   month = month,
						   day = day,
						   start_hour = start_hour,
						   end_hour = end_hour,
						   channel = channel,
						   airport = airport,
						   small_file_size = small_file_size,
						   audiourl = audiourl,
						   nextday_end_hour = nextday_end_hour,
						   verbose = verbose)
			print('\t',channel, ' voice tapes of ', airport, ' airport have been downloaded to local folder ', localfolder)
		else:
			Audio_to_FTP(year = year,
						 month = month,
						 day = day,
						 start_hour = start_hour,
						 end_hour = end_hour,
						 channel = channel,
						 airport = airport,
						 small_file_size = small_file_size,
						 ftpurl = ftpurl,
						 username = username,
						 password = password,
						 ftpfolder = ftpfolder,
						 localfolder = localfolder,
						 audiourl = audiourl,
						 nextday_end_hour = nextday_end_hour,
						 verbose = verbose)
			if mode == 'FTP':
				shutil.rmtree(os.getcwd() + '/' + localfolder)
				print('\t', channel, ' voice tapes of ', airport, ' airport have been uploaded to FTP folder ', ftpurl, '/', ftpfolder)
			elif mode == 'both':
				print('\t', channel, ' voice tapes of ', airport, ' airport have been downloaded to local folder ', localfolder, 
					' and uploaded to FTP folder ' , ftpurl, '/', ftpfolder)
	main()






