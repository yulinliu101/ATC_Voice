import os
import ftplib

def ftp_small_file(ftpobj, size = 500):
    for i in ftpobj.nlst():
        if ftpobj.size(i) < size:
            try:
                ftpobj.delete(i)
            except Exception:
                ftpobj.rmd(i)
            print(i + ' has been removed due to small size')

def local_small_file(path, size = 500):
	for i in os.listdir(path):
		if os.path.getsize(i) < size:
			try:
				os.remove(i)
			except:
				pass

def mkdir_chdir(foldername):
	try:
		os.mkdir(foldername)
	except:
		pass
	os.chdir(foldername)