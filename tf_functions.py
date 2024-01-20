import numpy as np
import scipy.signal as signal
from numpy.lib.stride_tricks import sliding_window_view


def get_stft(sub_signal,nperseg):
    nperseg = 1024
    noverlap = int(nperseg//2)
    sigO = sliding_window_view(sub_signal, nperseg)[::noverlap,:]
    smoothing_window = signal.windows.hann(nperseg)
    sigO = np.multiply(sigO,smoothing_window)
    #print("done multiplying")
    sig_stft = np.fft.fft(sigO,axis=1)       # applies FFT to each row
    sig_stft = np.abs(sig_stft)              # take magnitude
    sig_stft = sig_stft[:,:nperseg//2+1]     # keep only positive frequencies
    return sig_stft