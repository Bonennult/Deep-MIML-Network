import os
import wave
import numpy as np
import scipy

def Istft(V_matrix,Dir,Name,SegNum,Params,Audio_length):
    win_length = 4800
    hop_length = int(win_length / 2)
    istft = scipy.signal.istft(Zxx=V_matrix,nperseg=win_length,noverlap=hop_length)[1]
    istft = istft.astype(np.int16)
    Pth = Name+'_seg'+str(SegNum)+'.wav'
    print(os.path.join(Dir, Pth))
    # Establish Dictionary
    if not os.path.exists(Dir):
        os.makedirs(Dir)
    f = wave.open(os.path.join(Dir, Pth),"wb")
    istft = istft[:Audio_length]
    print("Save audio to ",Dir+Pth)
    f.setparams(Params)
    f.writeframes(istft.tostring())
    f.close()
    return Pth
