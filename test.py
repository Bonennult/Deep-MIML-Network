#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import os
import h5py
import json
import time
import numpy as np
from util import audio_import,audio_save,nmf_calculate
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.preprocess import preprocess_for_test, preprocess_for_test_by_seg
from util.feat_extractor import names, idxs


if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.batchSize = 1  # set batchSize = 1 for testing
    
    # 对数据进行预处理
    print('Pre-processing datasets for test ...')
    start = time.time()
    preprocess_for_test_by_seg(opt.dataset_root)
    end = time.time()
    print('Pre-process for test completed !')
    print('Pre-process time : %d min' % int((end-start)/60))

    start = time.time()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#testing images = %d' % dataset_size)

    Bases = json.load(open("bases.json","r"))

    model = torch.load(os.path.join('.', opt.checkpoints_dir, opt.name, str(opt.which_epoch) + '.pth'))
    model.BasesNet.eval()

    accuracies = []
    losses = []


    #--------------------------------------------------------------------------------

    # Get Path for audio
    h5f_path = os.path.join(opt.dataset_root, opt.mode + ".h5")
    h5f = h5py.File(h5f_path, 'r')
    img_path = []
    audio_path = []
    for Path in h5f['bases']:
        audio_path.append( Path.decode()[:-10])

    def Extract(W,num=20):
        GroupSize = int(W.shape[1]/num)
        Extract_W = np.zeros([W.shape[0],1])
        for i in range(num):
            W_part = W[:,GroupSize*i:GroupSize*(i+1)]
            Extract_W = np.hstack((Extract_W,W_part.mean(axis=1).reshape(W.shape[0],1)))
        return Extract_W[:,1:]

    opt.how_many = dataset_size
    Instruments = {0:'accordion',1:'acoustic_guitar',2:'cello',
                   3:'trumpet',4:'flute',5:'xylophone',6:'saxophone',7:'violin'}
    FileDict = {}
    opt.how_many = dataset_size
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        print(i)

        # STFT of test audio 
        [V_matrix,params,Audio_length] = audio_import.Stft(audio_path[i]+'.wav')
        V_matrix_re = abs(V_matrix)

        # Feed data into MIML
        accuracy, loss = model.test_multi_label(data)
        accuracies.append(accuracy)
        losses.append(loss)
        Result = model.output.detach().numpy()
        Labels = [int(x) for x in data['label'][0][:2]]

        # Load key bases according to label
        W_matrix_1 = np.array(Bases[str(Labels[0])]).T
        W_matrix_2 = np.array(Bases[str(Labels[1])]).T

        # Balance number of key bases
        Shape = min(W_matrix_1.shape[1],W_matrix_2.shape[1],100)
        W_matrix_1 = Extract(W_matrix_1,Shape)
        W_matrix_2 = Extract(W_matrix_2,Shape)

        # W = (W_1,W_2)
        W_matrix = np.hstack((W_matrix_1 , W_matrix_2))
        print("Shape for W",W_matrix.shape)

        # Operate Fixed NMF on H
        H_matrix_init = np.random.rand(W_matrix.shape[1],V_matrix_re.shape[1])
        H_matrix = nmf_calculate.nmf_cal(V_matrix_re,W_matrix,H_matrix_init,1e-4,50,20,False)[1]
        #H_matrix = H_matrix_init

        # Seperate H matrix
        H_matrix_1 = H_matrix[:W_matrix_1.shape[1],:]
        H_matrix_2 = H_matrix[W_matrix_1.shape[1]:,:]

        V_recover = []
        V_re = []
        # V_re_j = W_j*H_j
        V_re.append(np.dot(W_matrix_1 ,H_matrix_1))
        V_re.append(np.dot(W_matrix_2 ,H_matrix_2))
        V_sum = V_re[0]+V_re[1]
        V_sum[V_sum==0]=1

        # V_recover_j = V_re_j/sum(V_re_j) * V_matrix
        V_recover.append(np.multiply(V_re[0]/(V_sum),V_matrix))
        V_recover.append(np.multiply(V_re[1]/(V_sum),V_matrix))
        Orig_pth = audio_path[i].split('/')
        FileDict[Orig_pth[-1]+'.mp4']=[]   # 有改动
        for k,V in enumerate(V_recover):
            Save_dir = os.path.join(opt.dataset_root, 'result_audio/')
            Dict = {}
            Dict["audio"]=audio_save.Istft(V, Save_dir, Orig_pth[-1], k+1, params, Audio_length)
            Dict["label"]=Instruments[Labels[k]]
            FileDict[Orig_pth[-1]+'.mp4'].append(Dict)  # 有改动

    if not os.path.exists(os.path.join(opt.dataset_root, 'result_json')):
        os.mkdir(os.path.normpath(os.path.join(opt.dataset_root, 'result_json')))
    with open(os.path.join(opt.dataset_root, 'result_json', 'result.json'),'w') as outfile:
        json.dump(FileDict,outfile,ensure_ascii=False)
        outfile.write('\n')   
    accuracy = sum(accuracies)/len(accuracies)
    loss = sum(losses)/len(losses)
    print('Sound source seperation completed !')
    print()
    
    # 音源定位
    import json
    with open(os.path.join(opt.dataset_root, 'locations.json'),'r') as f:
        locations = json.load(f)
    with open(os.path.join(opt.dataset_root, 'result_json', 'result.json'),'r') as f:
        seperations = json.load(f)
    for file in seperations.keys():
        locate = locations[file]
        locate0 = locate[names.index(seperations[file][0]['label'])]  # 根据乐器名字查找对应位置
        locate1 = locate[names.index(seperations[file][1]['label'])]
        seperations[file][0]['position'] = int(locate0 > locate1)
        seperations[file][1]['position'] = 1 - seperations[file][0]['position']
    if not os.path.exists(os.path.join(opt.dataset_root, 'result_json')):
        os.path.mkdir(os.path.normpath(os.path.join(opt.dataset_root, 'result_json')))
    with open(os.path.join(opt.dataset_root, 'result_json', 'result.json'),'w') as f:
        json.dump(seperations, f)
    
    end = time.time()
    print('Sound source seperation and location complete !')
    print('Process time : %d min' % int((end-start)/60))
    print('Result json path  : ', os.path.normpath(os.path.join(opt.dataset_root, 'result_json', 'result.json')))
    print('Result audio path : ', os.path.normpath(os.path.join(opt.dataset_root, 'result_audio/')))
    print('All work done !')
