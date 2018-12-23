# -*- coding: utf-8 -*-

#import stft_utils
import os
import wave
import numpy as np
from dataset.NMF import stft_utils
from dataset.NMF import constants
from dataset.NMF import transformer_nmf

'''
filepath = "D:/shiting/audio"
filename= os.listdir(filepath)
stft_result = []
nmf_result =[]
'''

def nmf(filename):
    stft_result = []
    nmf_result =[]
    # Load audio file & compute stft
    for file in filename:
        f = wave.open(file,'rb')
        params = f.getparams()
        nframes = params[3]
        strData = f.readframes(nframes)
        waveData = np.fromstring(strData,dtype=np.int16)
        waveData = waveData[:44000*30]
        win_type = constants.WINDOW_HANN
        win_length = 4400
        hop_length = int(win_length / 2)
        stft = stft_utils.e_stft(signal=waveData, window_length=win_length,
                                             hop_length=hop_length, window_type=win_type,n_fft_bins=4801)
        stft_result.append(abs(stft))

    # Calculate nmf for every audio
    for stft in stft_result:
        nmf_result.append( transformer_nmf.TransformerNMF(input_matrix=stft, num_components=16,
                                                     seed=0, should_do_epsilon=False,
                                                     max_num_iterations=50,
                                                     distance_measure='kl_divergence').template_dictionary)
    return nmf_result
