from dateutil.parser import parse
from datetime import datetime
from dateutil import tz
import shutil


# METHOD 1: Hardcode zones:
to_zone = tz.gettz('UTC')
from_zone = tz.gettz('America/New_York')

def copy_file(local_times, channels = ['Tower'], root_dir = 'E:/AudioData/', new_dir = 'C:/Users/Yulin Liu/Desktop/New folder'):
    for channel in channels:
        for local_time in local_times:
            local_time = parse(local_time)
            local_time = local_time.replace(tzinfo=from_zone)
            utc_time = local_time.astimezone(to_zone)
            if channel == 'Tower':
                fname_0 = 'KJFK-Twr-'
            else:
                fname_0 = 'KJFK-NY-App-%s-'%channel
            fname_1 = datetime.strftime(utc_time, '%b-%d-%Y-%H')
            fname_2 = 30*(utc_time.minute//30)
            fname = fname_0 + fname_1 + str(fname_2).zfill(2) + 'Z.mp3'
            try:
                shutil.copy(root_dir + channel + '/' + fname, new_dir)
            except:
                print(fname)

copy_file(['3/4/2018 16:11', 
            '3/18/2018 16:21',
            '3/29/2018 5:32',
            '3/14/2018 8:46',
            '3/15/2018 23:39',
            '1/26/2018 15:20',
            '1/19/2018 18:18',
            '2/25/2018 7:37',
            '2/5/2018 18:06'], channels = ['Tower', 'ROBER', 'Final', 'CAMRN'])