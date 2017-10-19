from __future__ import division
import numpy as np
import scipy
import matplotlib.pyplot as plt

def Hz2Mel(freq):
    # convert Hz to Mel
    return 2595 * np.log10(1 + freq/700.)

def Mel2Hz(mel):
    # convert Mel to Hz
    return 700 * (10 ** (mel/2595.) - 1)


class AudioFeatures:
    def __init__(self, 
                 sound_track, 
                 sample_rate, 
                 nperseg, 
                 overlap_rate, 
                 nfft, 
                 window_fun = 'hann', 
                 pre_emphasis = True, 
                 premp_alpha = 0.97,
                 nfbank = 40, 
                 fbank_lowfreq = 0,
                 fbank_hfreq = None):

        self.sound_track = sound_track
        self.sample_rate = sample_rate
        self.nperseg = nperseg
        self.overlap_rate = overlap_rate
        self.nfft = nfft
        self.window_fun = window_fun
        self.nfbank = nfbank
        self.fbank_lowfreq = fbank_lowfreq
        self.fbank_hfreq = fbank_hfreq

        if pre_emphasis:
            self.sound_track = self.preEmphasis(premp_alpha)

    def preEmphasis(self, alpha = 0.97):
        emphasized_signal = np.append(self.sound_track[0], self.sound_track[1:] - alpha * self.sound_track[:-1])
        return emphasized_signal

    def stft(self, power_mode = 'PS'):
        freqs, time_ins, Pxx = scipy.signal.stft(self.sound_track, 
                                                 fs = self.sample_rate, 
                                                 nperseg = self.nperseg, 
                                                 noverlap = self.nperseg/self.overlap_rate, 
                                                 nfft = self.nfft,
                                                 window = self.window_fun, 
                                                 return_onesided = True, 
                                                 padded = False, 
                                                 boundary = None, 
                                                 detrend = 'constant')
        if power_mode == 'PS':
            Pxx = (np.abs(Pxx))**2 # Power Spectrum V**2
        elif power_mode == 'PSD':
            Pxx = (np.abs(Pxx))**2/self.nfft # Power Spectrum Density V**2/Hz
        elif power_mode == 'magnitude':
            Pxx = np.abs(Pxx)
        else:
            raise ValueError('power_mode can only be "PS", "PSD", or "magnitude"')
        return freqs, time_ins, Pxx

    def melFilterBank(self, nfilt = 40, lowfreq = 0, highfreq = None):
        """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A np array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        if highfreq is None:
            highfreq = self.sample_rate / 2.
        if highfreq > self.sample_rate / 2.:
            highfreq = self.sample_rate / 2.
            print("highfreq is greater than sample_rate/2, auto corrected to sample_rate/2")

        # compute points evenly spaced in mels
        lowmel = Hz2Mel(lowfreq)
        highmel = Hz2Mel(highfreq)
        melpoints = np.linspace(lowmel, highmel, nfilt + 2)
        # convert Hz to fft bin number
        bin_num = np.floor((self.nfft + 1) * Mel2Hz(melpoints) / self.sample_rate)

        fbank = np.zeros([nfilt, self.nfft//2 + 1])
        for j in range(0, nfilt):
            for i in range(int(bin_num[j]), int(bin_num[j + 1])):
                fbank[j, i] = (i - bin_num[j]) / (bin_num[j + 1] - bin_num[j])
            for i in range(int(bin_num[j + 1]), int(bin_num[j + 2])):
                fbank[j, i] = (bin_num[j + 2] - i) / (bin_num[j + 2] - bin_num[j + 1])
        return fbank

    def MFCC(self, 
             num_cep = 12, 
             lifting = True, 
             lifting_coef = 22, 
             mean_normalization = True):
        self.num_cep = num_cep
        self.fbank = self.melFilterBank(nfilt = self.nfbank, lowfreq = self.fbank_lowfreq, highfreq = self.fbank_hfreq)
        self.freqs, self.time_ins, self.Pxx = self.stft(power_mode = 'PSD')
        
        fbank_energy = self.fbank.dot(self.Pxx)
        self.fbank_energy = np.where(fbank_energy == 0, 1e-10, fbank_energy)
        self.fbank_energy_db = 10. * np.log10(self.fbank_energy) # db/Hz

        self.mfcc = scipy.fftpack.dct(self.fbank_energy_db, type = 2, axis = 0, norm = 'ortho')[1:(num_cep + 1), :]

        if lifting:
            return self.lifter(lifting_coef, mean_normalization)
        else:
            return self.mfcc

    def Delta(self, 
              nshift = 2, 
              order = 1):
        if nshift < 1:
            raise ValueError("shiting parameter has to be greater than 1")

        if order == 1:
            try:
                target = self.mfcc.copy()
            except:
                target = self.MFCC()
        else:
            target = self.Delta(nshift, order = order - 1)

        n_frame = target.shape[1]
        denominator = nshift * (nshift + 1) * (2 * nshift + 1)/3.
        delta = np.zeros(shape = target.shape)
        delta_pad = np.pad(target, ((0,0), (nshift, nshift)), mode = 'edge')

        for i in range(nshift):
            delta += (i + 1) * (delta_pad[:, (nshift + i + 1): (delta_pad.shape[1] + i - nshift + 1)] - \
                                 delta_pad[:, (nshift - 1 - i): (delta_pad.shape[1] - i - nshift - 1)])

        return delta/denominator

    def FeatureExtraction(self, highest_order, **kwarg):
        self.all_features = self.MFCC(kwarg['num_cep'], kwarg['lifting'], kwarg['lifting_coef'], kwarg['mean_normalization'])
        if highest_order == 0:
            return self.all_features
        for order in range(1, highest_order + 1):
            exec("delta_%d = self.Delta(nshift = kwarg['nshift'], order = %d)"%(order, order))
            exec("self.all_features = np.concatenate((self.all_features, delta_%d), axis = 0)"%order)
        return self.all_features


    def lifter(self, lifting_coef = 22, mean_normalization = True):
        num_cep, n_frame = self.mfcc.shape
        lift = 1 + (lifting_coef/2.) * np.sin(np.pi * np.arange(num_cep)/lifting_coef)
        new_mfcc = (lift * self.mfcc.T).T
        if mean_normalization:
            return (new_mfcc.T - np.mean(new_mfcc, axis = 1)).T
        else:
            return new_mfcc

    def Visualizer(self, item = 'mfcc', tmin = None, tmax = None):
        plt.figure(figsize=(18,6))
        if tmin is None:
            tmin = self.time_ins.min()
        if tmax is None:
            tmax = self.time_ins.max()
        # tmin, tmax = self.time_ins.min(), self.time_ins.max()
        if item == 'signal':
            print('only plot the first 200 seconds')
            plt.title('Signal')
            end_sec = 200
            start_sec = 0
            t = np.linspace(start_sec, end_sec, (end_sec - start_sec) * self.sample_rate)
            plt.plot(t, self.sound_track[int(start_sec * self.sample_rate): int(end_sec * self.sample_rate)])
            plt.ylim(-1,1)

        elif item == 'mfcc':
            plt.title('MFCC Diagram')
            im = plt.imshow(np.flipud(self.mfcc[:, np.where((self.time_ins >= tmin) & (self.time_ins <= tmax))[0]]), 
                            aspect = 'auto', 
                            extent = [tmin, tmax, 0, self.num_cep], 
                            interpolation = 'nearest')
            plt.xlabel('Time/ sec')
            plt.ylabel('MFCC Coefficients')
            cb = plt.colorbar(im, orientation='horizontal')
            cb.set_label('MFCC')
        elif item == 'spectrogram':
            plt.title('Spectrogram')
            im = plt.imshow(np.flipud(10 * np.log10(self.Pxx[:, np.where((self.time_ins >= tmin) & (self.time_ins <= tmax))[0]])), 
                            aspect = 'auto', 
                            extent = [tmin, tmax, self.freqs.min(), self.freqs.max()], 
                            interpolation = 'nearest')
            plt.xlabel('Time/ Sec')
            plt.ylabel('Frequency / Hz')
            cb = plt.colorbar(im, orientation='horizontal')
            cb.set_label('Cross Spectral Density (dB/Hz)')
        elif item == 'filterbank':
            plt.title('Filterbank Diagram')
            im = plt.imshow(np.flipud(self.fbank_energy_db[:, np.where((self.time_ins >= tmin) & (self.time_ins <= tmax))[0]]), 
                            aspect = 'auto', 
                            extent = [tmin, tmax, self.freqs.min(), self.freqs.max()], 
                            interpolation = 'nearest')
            plt.xlabel('Time/ Sec')
            plt.ylabel('Frequency / Hz')
            cb = plt.colorbar(im, orientation='horizontal')
            cb.set_label('Filter Banks Energy')
        elif item == 'features':
            plt.figure(figsize=(18,12))
            plt.title('Feature Map')
            im = plt.imshow(np.flipud(self.all_features[:, np.where((self.time_ins >= tmin) & (self.time_ins <= tmax))[0]]), 
                            aspect = 'auto', 
                            extent = [tmin, tmax, 0, self.all_features.shape[0]], 
                            interpolation = 'nearest')
            plt.xlabel('Time/ Sec')
            plt.ylabel('Feature Order')
            cb = plt.colorbar(im, orientation='horizontal')
            cb.set_label('Feature Map Coefficients')
        else:
            raise ValueError('item not implemented! item could only be "signal","mfcc", "spectrogram", "features", or "filterbank"')
        plt.show()
