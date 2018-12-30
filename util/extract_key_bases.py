from util import audio_import,audio_save,nmf_calculate
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
import torch
import os
import json
import h5py

#opt = TestOptions().parse()

def extract_key_bases(opt):
    opt.batchSize = 1  # set batchSize = 1 for testing

    h5f_path = os.path.join(opt.dataset_root, opt.mode + ".h5")
    h5f = h5py.File(h5f_path, 'r')
    img_path = []
    audio_path = []
    for Path in h5f['bases']:
        audio_path.append( Path.decode())


    def Find_key_bases(Map,label):
        Bases_idx = []
        for i in range(Map.shape[0]):
            if Map[i,label]==max(Map[i,:]):
                Bases_idx.append(i)
        Bases_idx = list(set(Bases_idx))
        return Bases_idx

    opt.mode = "train"  # 或许可以删掉

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#testing images = %d' % dataset_size)

    model = torch.load(os.path.join('.', opt.checkpoints_dir, opt.name, 'latest0' + '.pth'))
    model.BasesNet.eval()

    key_bases = {}
    for x in range(8):
        key_bases[x]=[]

    opt.how_many = dataset_size
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        print("Dataset %d to extract:"%(i))
        print("Label for data is",data['label'])
        Label =[]
        model.test(data)
        Label.append(int(data['label'][0][0]))
        if int(data['label'][0][1]) != -1:
            Label.append(int(data['label'][0][1]))
        Relation_Map = model.softmax_normalization_output.detach().numpy()
        Relation_Map = Relation_Map.reshape([Relation_Map.shape[2],Relation_Map.shape[3]]) 
        print("Size of Relation Map:",Relation_Map.shape)
        for label in Label:
            key_base_idx = Find_key_bases(Relation_Map,label)
            Bases = data['bases'].detach().numpy()
            key_base = Bases.reshape([Bases.shape[1],Bases.shape[2]])[:,key_base_idx]
            key_bases[label] = key_bases[label] + key_base.T.tolist()
    for i in range(8):
        print("key_bases shape for %d: %d" %(i,len(key_bases[i])))

    with open('bases.json','w') as outfile:
        #print(outfile)
        json.dump(key_bases,outfile,ensure_ascii=False)
        outfile.write('\n')
