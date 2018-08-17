# -*- coding: utf-8 -*-
# @Author: Lu Dai, Yulin Liu
# @Date:   2018-08-15 16:20:17
# @Last Modified by:   Lu Dai
# @Last Modified time: 2018-08-17 15:09:01

import wget
import calendar
import os
import argparse
import ftplib
import time
import random
import pandas as pd

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


def audio_to_FTP(year = 2018,
                 month = 1,
                 day = 1,
                 channel = 'CAMRN',
                 airport = 'KJFK',
                 ftpurl = 'ftp.atac.com',
                 username = 'sn2018',
                 password = 'Teardr2p',
                 ftpfolder = 'VoiceData\\',
                 audiourl = 'http://archive.fmt2.liveatc.net/kjfk/',
                 verbose = False):
    try:
        session = ftplib.FTP(ftpurl, username, password)
    except:
        print('Fail to access FTP')
    session.cwd(ftpfolder + channel)
    date = str(year) + str(month).zfill(2) + str(day).zfill(2)
    try:
        session.mkd(date)
    except:
        print(date + ' folder is already exsit.')
    session.cwd(date)
    
    rmd_small_file(session)
    for Period in range(24):
        for HalfHour in ['00','30']:
            if channel == 'Tower':
                FileName = 'KJFK-Twr-' + calendar.month_abbr[month] + '-' + '{:02d}'.format(day) + '-' + str(year) + '-' + '{:02d}'.format(Period) + HalfHour + 'Z.mp3'
            elif channel == 'Tower1191':
                FileName = 'KJFK-Twr-1191-' + calendar.month_abbr[month] + '-' + '{:02d}'.format(day) + '-' + str(year) + '-' + '{:02d}'.format(Period) + HalfHour + 'Z.mp3'
            elif channel == 'Tower1239':
                FileName = 'KJFK-Twr-1239-' + calendar.month_abbr[month] + '-' + '{:02d}'.format(day) + '-' + str(year) + '-' + '{:02d}'.format(Period) + HalfHour + 'Z.mp3'
            else:
                FileName = 'KJFK-NY-App-' + channel + '-' + calendar.month_abbr[month] + '-' + '{:02d}'.format(day) + '-' + str(year) + '-' + '{:02d}'.format(Period) + HalfHour + 'Z.mp3'
            url = audiourl + FileName
            if FileName in session.nlst():
                pass
            else:
                try:
                    # Check if the file exist, if so, skip, o/w. download
                    # If not os.path.exist(FileName):
                    filename = wget.download(url)
                    one = open(filename, 'rb')
                    session.storbinary('STOR ' + filename, one)
                    one.close()
                    time.sleep(random.randint(1,3))
                    if verbose:
                        print(filename)
                except:
                    print('File Skipped: %s'%FileName)
    rmd_small_file(session) #Re-check small files
    # session.quit()