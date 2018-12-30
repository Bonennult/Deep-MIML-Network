import os
import numpy as np
import h5py
import json
import random
from util import audio_import
from util.feat_extractor import load_model, get_CAM, feat_pred

# idxs，names 是在 feat_extractor.py 中定义的 global 变量
# idxs=[401, 402, 486, 513, 558, 642, 776, 889]
# names=['accordion','acoustic_guitar','cello','trumpet','flute','xylophone','saxophone','violin']

def preprocess_for_train(train_path):
    # train_path = 'h:/Study/bpfile/dataset/'
    # train_path = 'h:/Study/bpfile/dataset/audios/'
    train_path = os.path.normcase(os.path.join(train_path, 'audios/')).replace('\\','/')
    wav_files = []
    bases_filepath = []
    labels_filepath = []

    # 遍历数据集 `path` 路径下所有的 wav 文件，记录他们的路径
    # wav_files[0] = 'h:/study/bpfile/dataset/audios/duet/acoustic_guitarviolin/1.wav'
    for root,dirs,files in os.walk(train_path, topdown=True):
        for name in files:
            if '.wav' in name:
                wav_files.append(os.path.normcase(root+os.path.sep+name).replace('\\','/'))

    # 计算 bases 和 labels
    for i, fname in enumerate(wav_files):
        label = np.zeros(1000)   # 1000个乐器类别
        
        # 每个 wav 对应的 NMF 结果保存为 *.npy 文件，对应路径存入 bases
        # bases_filepath[0] = 'h:/study/bpfile/dataset/audios/duet/acoustic_guitarviolin/1_bases.npy'
        # np.array(nmf_base[0]).shape = (2401, 16)
        nmf_base = audio_import.nmf([fname])
        np.save(fname[:-4]+'_bases'+'.npy', nmf_base[0])
        bases_filepath.append(fname[:-4]+'_bases'+'.npy')

        # 根据文件路径名确定当前音频中乐器的 label，作为 ground truth 训练
        # labels_filepath[0] = 'h:/study/bpfile/dataset/audios/duet/acoustic_guitarviolin/1_labels.npy'
        for instrument, index in zip(names, idxs):
            if instrument in fname:
                label[index] = 1
        np.save(fname[:-4]+'_labels'+'.npy', label)
        labels_filepath.append(fname[:-4]+'_labels'+'.npy')

        print(bases_filepath[-1])
        print(labels_filepath[-1])

    # 保存为 train.h5 和 val.h5
    dataset_num = len(wav_files)
    train_num = int(0.7*dataset_num)

    # 随机选取 70% 的数据作为验证集
    train_bases = random.sample(range(dataset_num), train_num)

    # 路径 utf-8 编码，否则报错
    train_bases_encode = []
    train_labels_encode = []
    val_bases_encode = []
    val_labels_encode = []
    for i in range(dataset_num):
        if i in train_bases:
            train_bases_encode.append(bases_filepath[i].encode(encoding='utf-8', errors='strict'))
            train_labels_encode.append(labels_filepath[i].encode(encoding='utf-8', errors='strict'))
        else:
            val_bases_encode.append(bases_filepath[i].encode(encoding='utf-8', errors='strict'))
            val_labels_encode.append(labels_filepath[i].encode(encoding='utf-8', errors='strict'))

    h5f = h5py.File(os.path.join(train_path, 'train.h5'), 'w')
    h5f.create_dataset('bases', data=train_bases_encode)
    h5f.create_dataset('labels', data=train_labels_encode)
    h5f.close()

    h5f = h5py.File(os.path.join(train_path, 'val.h5'), 'w')
    h5f.create_dataset('bases', data=val_bases_encode)
    h5f.create_dataset('labels', data=val_labels_encode)
    h5f.close()
    
    print('All work done !')
    
    
def preprocess_for_test(test_path):
    # test_path = 'h:/Study/bpfile/testset25/'
    wav_files = []
    img_files = []
    bases_filepath = []
    labels_filepath = []
    locations = {}

    # 寻找音频路径与图片路径
    # wav_dir = 'h:/Study/bpfile/testset25/testaudio/'
    # img_dir = 'h:/Study/bpfile/testset25/testimage/'
    wav_dir = os.path.normcase(os.path.join(test_path, 'gt_audio/')).replace('\\','/')
    img_dir = os.path.normcase(os.path.join(test_path, 'testimage/')).replace('\\','/')
    '''
    for cur_dir in os.listdir(test_path):
        if 'audio' in cur_dir:
            wav_dir = os.path.normcase(os.path.join(test_path, cur_dir, '/')).replace('\\','/')
        if 'image' in cur_dir:
            img_dir = os.path.normcase(os.path.join(test_path, cur_dir, '/')).replace('\\','/')
            '''

    # 寻找所有 wav 的文件名、图片文件所在文件夹
    # wav_files[0] = 'accordion_1_saxophone_1.wav'
    # img_files[0] = 'accordion_1_saxophone_1'
    for f in os.listdir(wav_dir):   # 剔除非 wav 文件，以及 ground truth
        if '_gt1.wav' not in f and '_gt2.wav' not in f and '.wav' in f:
            wav_files.append(f)
    for f in os.listdir(img_dir):   # 剔除目录下所有文件，只保留文件夹
        if os.path.isdir(img_dir + f):
            img_files.append(f)
            locations[f+'.mp4'] = []

    # 排序确保 wav 文件与图片文件夹相对应
    wav_files.sort()
    img_files.sort()
    load_model()

    # 计算 bases 和 labels
    for wav_fname, img_folder in zip(wav_files, img_files):
        assert wav_fname[:-4] == img_folder    # 确保 wav 文件与图片文件夹相对应

        # 计算 NMF 并将结果保存在 npy 文件中
        # bases_filepath[0] = 'h:/Study/bpfile/testset25/testaudio/accordion_1_saxophone_1_bases.npy'
        # np.array(nmf_base[0]).shape = (2401, 16)
        nmf_base = audio_import.nmf([wav_dir+wav_fname])
        np.save(wav_dir+wav_fname[:-4]+'_bases'+'.npy', nmf_base[0])
        bases_filepath.append(wav_dir+wav_fname[:-4]+'_bases'+'.npy')

        imgs = []
        # 获得当前文件夹下所有图片（绝对）路径，并随机抽样
        # img_folder = 'accordion_1_saxophone_1'
        # imgs[0] = '000001.jpg'
        for img in os.listdir(img_dir+img_folder):
            if '.jpg' in img or '.png' in img:
                imgs.append(img)
        imgs = random.sample(imgs, 30)  # 由于图片数量过多，每个视频中只随机抽取 50 张图片进行预测

        probs=np.zeros([8])
        location = np.zeros([8])
        # 计算图像的 label，并获得每种乐器的定位（列号）
        for img in imgs:
            probs1 = feat_pred(img_dir+img_folder, img)
            probs = probs + np.array(probs1)
            #(locate, CAMs, heatmap) = get_CAM(img_dir+img_folder, 'results', img)  # heatmap 保存为文件
            locate = get_CAM(img_dir+img_folder, 'results', img)
            location = location + locate
            '''
            print(np.argmax(CAMs))
            print(CAMs)
            plt.figure()
            plt.imshow(0.3*CAMs)
            plt.figure()
            plt.imshow(heatmap)
            '''

        locations[img_folder+'.mp4'] = location
        print(names)
        print(probs)
        print(location)
        softmax = np.zeros(1000)
        for i in range(len(probs)):
            softmax[idxs[i]] = probs[i]
        np.save(img_dir+img_folder+'/labels'+'.npy', softmax)
        labels_filepath.append(img_dir+img_folder+'/labels'+'.npy')

        print(bases_filepath[-1])
        print(labels_filepath[-1])
        print()

        # 保存为 test.h5 文件
        test_bases_encode = []
        test_labels_encode = []

        for i in range(len(bases_filepath)):
            test_bases_encode.append(bases_filepath[i].encode(encoding='utf-8', errors='strict'))
            test_labels_encode.append(labels_filepath[i].encode(encoding='utf-8', errors='strict'))

        h5f = h5py.File(os.path.join(test_path, 'test.h5'), 'w')
        h5f.create_dataset('bases', data=test_bases_encode)
        h5f.create_dataset('labels', data=test_labels_encode)
        h5f.close()
        
        # 保存乐器位置 locations
        import json
        for k in locations.keys():
            locations[k] = list(locations[k])
        with open(os.path.join(test_path, 'locations.json'),'w') as f:
            json.dump(locations, f)
        
        print('All work done !')