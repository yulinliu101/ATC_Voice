# -*- coding: utf-8 -*-
# @Author: Yulin Liu
# @Date:   2018-08-13 14:46:06
# @Last Modified by:   Yulin Liu
# @Last Modified time: 2018-08-14 12:05:31

import calendar
from dateutil import parser

global baseline_time
baseline_time = parser.parse('01/01/2017 0:0:0')
"""
useful functions
"""
def audio_file_header_collector(year = 2018, 
                          month = 1, 
                          day = 1, 
                          start_hour = 0, 
                          end_hour = 23, 
                          channel = 'Twr', 
                          airport = 'KJFK'):
    """
    start_hour and end hour is in [0, 23]
    all time in UTC
    channel selection: 'CAMRN', 'ROBER', 'Twr'
    """
    if channel == 'Twr' or channel == 'Tower':
        header = '%s-%s-%s-%s-%d-'%(airport, channel, calendar.month_abbr[month], str(day).zfill(2), year)
    elif channel == 'ROBER' or channel == 'CAMRN':
        header = '%s-NY-App-%s-%s-%s-%d-'%(airport, channel, calendar.month_abbr[month], str(day).zfill(2), year)
    else:
        raise ValueError('channel not found!')

    footer = 'Z.mp3'
    file_name_list = []
    for i in range(start_hour, end_hour + 1):
        file_name_list += [header + str(i).zfill(2) + '00' + footer,
                           header + str(i).zfill(2) + '30' + footer]

    return file_name_list

def tmp_file_zipper(target_path, 
                    dump_to_zipfile,
                    clean_target_path = True):

    # TODO: Add entire target_path to zipfile; Add remove dir function
    with zipfile.ZipFile(dump_to_zipfile, mode = 'w', compression = zipfile.ZIP_DEFLATED) as zfile:
        for file in os.listdir(target_path):
            zfile.write('%s/%s'%(target_path, file))

def Hz2Mel(freq):
    # convert Hz to Mel
    return 2595 * np.log10(1 + freq/700.)

def Mel2Hz(mel):
    # convert Mel to Hz
    return 700 * (10 ** (mel/2595.) - 1)

# help link
# https://dsp.stackexchange.com/questions/3377/calculating-the-total-energy-of-a-signal
# https://github.com/scipy/scipy/blob/master/scipy/signal/spectral.py
def _energy_helper(x, 
                   fs = 1.0, 
                   nperseg=None, 
                   noverlap=None,
                   nfft=None, 
                   axis=-1, 
                   boundary=None,
                   padded=False):

    if x.size == 0:
        return np.empty(x.shape), np.empty(x.shape), np.empty(x.shape)

    if x.ndim > 1:
        if axis != -1:
            x = np.rollaxis(x, axis, len(x.shape))


    if nperseg is not None:  # if specified by user
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap

    # Padding occurs after boundary extension, so that the extended signal ends
    # in zeros, instead of introducing an impulse at the end.
    # I.e. if x = [..., 3, 2]
    # extend then pad -> [..., 3, 2, 2, 3, 0, 0, 0]
    # pad then extend -> [..., 3, 2, 0, 0, 0, 2, 3]

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
        nadd = (-(x.shape[-1]-nperseg) % nstep) % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = np.concatenate((x, np.zeros(zeros_shape)), axis=-1)

    # Perform the windowed energy
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        step = nperseg - noverlap
        shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
        strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, 
                                                 shape=shape,
                                                 strides=strides)

    result = result.real
    result = np.fft.rfft(result, n=nfft)
    
    result = np.sum(np.abs(result)**2, axis = 1)
    result += 1e-15
    result = np.log10(result)

    time = np.arange(nperseg/2, 
                     x.shape[-1] - nperseg/2 + 1,
                     nperseg - noverlap)/float(fs)
    return time, result

from itertools import groupby, count
def combine_to_range(power, power_threshold, sec_to_bin, silence_sec = 0.5):
    # power is the matrix produced by specgram
    # power_threshold usually is 0
    # sec_to_bin is usally bin.shape[0] / audio length(ms)/1000
    
    idx_range = []

    groups = groupby(np.where(power <= power_threshold)[0], key = lambda i,x = count():i-next(x))
    tmp = [list(g) for k, g in groups]

    for item in tmp:
        if len(item) == 1:
            idx_range.append([item[0], item[0]])
        else:
            idx_range.append([item[0], item[-1]])

    idx_range = np.array(idx_range)
    silence_time_idx = idx_range[np.where((idx_range[:,1] 
                                              - idx_range[:,0]) > silence_sec * sec_to_bin), :][0]
    silence_time_range = idx_range[np.where((idx_range[:,1] 
                                              - idx_range[:,0]) > silence_sec * sec_to_bin), :][0]/sec_to_bin
    idx_nums = ()
    for i in silence_time_idx:
        idx_nums = np.append(idx_nums,range(int(i[0]), int(i[1]+1)))
    idx_act = np.sort(np.array(list(set(range(power.shape[0])).difference(set(idx_nums)))))
    
    silence_time_duration = silence_time_range[:,1] - silence_time_range[:,0]
    silence_time = sum(silence_time_duration)
    return silence_time_idx, silence_time, idx_act

def Movingavg(x, n = 5):
    padarray = np.lib.pad(x, n//2, 'edge')
    csum = np.cumsum(np.insert(padarray, 0, 0))
    y = (csum[n:] - csum[:-n])/n
    return y