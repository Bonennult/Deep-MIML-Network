# -*- coding: utf-8 -*-

#import stft_utils
import os
import wave
import numpy as np
from dataset.NMF import stft_utils
from dataset.NMF import constants
from dataset.NMF import transformer_nmf
import scipy
import sklearn
'''
filepath = "D:/shiting/audio"
filename= os.listdir(filepath)
stft_result = []
nmf_result =[]
'''
def hello_world():
    print("hello world")

def nmf(filename):
    stft_result = []
    nmf_result =[]
    # Load audio file & compute stft
    for file in filename:
        stft=Stft(file)
        stft_result.append(abs(stft))
    # Calculate nmf for every audio
    for stft in stft_result:
        #NMF =transformer_nmf.TransformerNMF(input_matrix=stft, num_components=25,
        #                                             seed=2018,should_update_activation =True,
         #                                            should_update_template=True,max_num_iterations=50,
          #                                           should_do_epsilon=False,
            #                                         distance_measure='kl_divergence')
        #NMF.transform()
        model = sklearn.decomposition.NMF(n_components=16,init='random',random_state=0)
        W = model.fit_transform(stft)
        H = model.components_
        nmf_result.append(W)
    return nmf_result

def Stft(Filename):
    f = wave.open(Filename,'rb')
    params = f.getparams()
    #print(params)
    nframes = params[3]
    strData = f.readframes(nframes)
    waveData = np.fromstring(strData,dtype=np.int16)
    #print(waveData.shape)
    #print("RawData:",waveData)
    #waveData = waveData[:44000*30]
    win_type = constants.WINDOW_HANN
    win_length = 4800
    hop_length = int(win_length / 2)
    #stft = stft_utils.e_stft(signal=waveData, window_length=win_length,
    #                                         hop_length=hop_length, window_type=win_type,n_fft_bins=4801)
    fs = 48000
    #window = scipy.signal.get_window('hamming',)
    stft2 = scipy.signal.stft(x=waveData,fs=fs,nperseg=win_length,noverlap=hop_length)[2]
    #print (stft)
    #print (stft-stft2)
    return stft2


