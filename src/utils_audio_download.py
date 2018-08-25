import wget
import os
import ftplib
import time
import random
import shutil
from utils import audio_file_header_collector
from utils_filestool import ftp_small_file, local_small_file, mkdir_chdir


def Audio_to_Local(year = 2018,
				 month = 8,
				 day = 24,
				 start_hour = 5,
				 end_hour = 4,
				 channel = "CAMRN",
				 airport = 'KJFK',
				 small_file_size = 500,
				 localfolder = 'ATCAudio/',
				 audiourl = 'http://archive.fmt2.liveatc.net/kjfk/',
				 nextday_end_hour = True,
				 verbose = False):
	parent_path = os.getcwd()

	mkdir_chdir(localfolder)
	mkdir_chdir(channel)
	filenames = audio_file_header_collector(year = year, month = month, day = day, 
                                        start_hour = start_hour, end_hour = end_hour,channel = channel,
                                        airport = airport, nextday_end_hour = nextday_end_hour)
	for i in filenames:
		date = i.split(channel + '/')[1][:8]
		FileName = i.split(date + '/')[1]
		url = audiourl + FileName
		mkdir_chdir(date)

		# Check small size files
		local_small_file(os.getcwd(), small_file_size)

		# Check file exists
		if FileName in os.listdir():
			pass
		else:
			try:
				filename = wget.download(url, out = os.getcwd())
			except:
				print('File Skipped: %s'%FileName)
		if verbose:
			print(FileName)
		os.chdir("..")
	os.chdir(parent_path)

def Audio_to_FTP(year = 2018,
				 month = 8,
				 day = 24,
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
				 verbose = False):

	global session
	parent_path = os.getcwd()

	mkdir_chdir(localfolder)
	mkdir_chdir(channel)

	try:
		session = ftplib.FTP(ftpurl, username, password)
	except:
		print('Fail to access FTP cite.')
	session.cwd(ftpfolder + channel)

	filenames = audio_file_header_collector(year = year, month = month, day = day, 
                                        start_hour = start_hour, end_hour = end_hour,channel = channel,
                                        airport = airport, nextday_end_hour = nextday_end_hour)
	for i in filenames:
		date = i.split(channel + '/')[1][:8]
		FileName = i.split(date + '/')[1]
		url = audiourl + FileName
		mkdir_chdir(date)

		# Check small size files
		local_small_file(os.getcwd(), small_file_size)

		# Check file exists
		if FileName in os.listdir():
			os.remove(FileName)

		filename = wget.download(url, out = os.getcwd())

		try:
			session.mkd(date)
		except:
			pass
		session.cwd(date)

		# Check small size files
		ftp_small_file(session, size = small_file_size)

		if FileName in session.nlst():
			pass
		else:
			one = open(filename, 'rb')
			session.storbinary('STOR ' + FileName, one)
			one.close()
			time.sleep(random.randint(1,3))
			# session.quit()

		if verbose:
			print(FileName)

		os.chdir("..")
		session.cwd("..")
	os.chdir(parent_path)