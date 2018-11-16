import numpy as np
import matplotlib.pyplot as plt

def Hz2Mel(freq):
    # convert Hz to Mel
    return 2595 * np.log10(1 + freq/700.)

def Mel2Hz(mel):
    # convert Mel to Hz
    return 700 * (10 ** (mel/2595.) - 1)

def melFilterBank(nfilt = 40, lowfreq = 0, highfreq = None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A np array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    sample_rate = 22050
    nfft = 1024
    if highfreq is None:
        highfreq = sample_rate / 2.
    if highfreq > sample_rate / 2.:
        highfreq = sample_rate / 2.
        print("highfreq is greater than sample_rate/2, auto corrected to sample_rate/2")

    # compute points evenly spaced in mels
    lowmel = Hz2Mel(lowfreq)
    highmel = Hz2Mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    # convert Hz to fft bin number
    bin_num = np.floor((nfft + 1) * Mel2Hz(melpoints) / sample_rate)

    fbank = np.zeros([nfilt, nfft//2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin_num[j]), int(bin_num[j + 1])):
            fbank[j, i] = (i - bin_num[j]) / (bin_num[j + 1] - bin_num[j])
        for i in range(int(bin_num[j + 1]), int(bin_num[j + 2])):
            fbank[j, i] = (bin_num[j + 2] - i) / (bin_num[j + 2] - bin_num[j + 1])
    return fbank

fbank = melFilterBank()
freq_band = np.linspace(0, 11025, 1024//2 + 1)

plt.figure(figsize = (8, 4))
for row in fbank:
    plt.plot(freq_band, row)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('The full 40 filter banks')
plt.show()