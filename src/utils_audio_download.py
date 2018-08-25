import wget
import calendar
import os
import argparse
import ftplib
import time
import random
import shutil
from utils import audio_file_header_collector

def rmd_small_file(ftpobj, size = 500):
    for i in ftpobj.nlst():
        if ftpobj.size(i) < size:
            try:
                ftpobj.delete(i)
            except Exception:
                ftpobj.rmd(i)
            print(i + ' has been removed due to small size')
        else:
            print(i + ' has been downloaded.')

def Audio_Download(year = 2018,
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
				 ftpfolder = 'VoiceData\\',
				 audiourl = 'http://archive.fmt2.liveatc.net/kjfk/',
				 extday_end_hour = True,
				 local = True,
				 FTP = True,
				 verbose = False):

	parent_path = os.getcwd()

	try:
		os.mkdir("ATCAudio")
	except:
		os.chdir("ATCAudio")

	if FTP:
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

		try:
			os.mkdir(channel)
		except:
			os.chdir(channel)
		try:
			os.mkdir(date)
		except:
			os.chdir(date)

		# Check small size files
		try:
			for i in os.listdir():
				if os.path.getsize(i) < small_file_size:
					os.remove(i)
		except:
			pass
		if FileName in os.listdir():
			os.remove(FileName)

		filename = wget.download(url, out = os.getcwd())

		if FTP:
			try:
				session.mkd(date)
			except:
				session.cwd(date)

			# Check small size files
			try:
				rmd_small_file(session, size = small_file_size)
			except:
				pass

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

		if not local:
			shutil.rmtree(parent_path + '/ATCAudio')



		


