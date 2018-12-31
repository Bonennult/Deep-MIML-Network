# -*- coding: utf-8 -*-

import wave
import numpy as np
from util import nmf_calculate
import scipy.signal

def nmf(filename):
    stft_result = []
    nmf_result =[]
    # Load audio file & compute stft
    for file in filename:
        (stft,params,Length)=Stft(file)
        stft_result.append(abs(stft))
    # Calculate nmf for every audio
    for stft in stft_result:
        W_init = np.random.rand(stft.shape[0],16)
        H_init = np.random.rand(16,stft.shape[1])
        (W,H) = nmf_calculate.nmf_cal(stft,W_init,H_init,1e-4,100,200,True)
        nmf_result.append(W)
    return nmf_result

def Stft(Filename):
    # Calculate STFT for audio
    f = wave.open(Filename,'rb')
    params = f.getparams()
    nframes = params[3]
    strData = f.readframes(nframes)
    waveData = np.fromstring(strData,dtype=np.int16)
    f.close()
    Length = len(waveData)
    win_length = 4800
    hop_length = int(win_length / 2)
    stft2 = scipy.signal.stft(x=waveData,nperseg=win_length,noverlap=hop_length)[2]
    return stft2,params,Length


