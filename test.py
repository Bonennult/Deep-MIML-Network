#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import torch
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.preprocess import preprocess_for_test

opt = TestOptions().parse()

# 对数据进行预处理
print('Pre-processing datasets for test ...')
start = time.time()
#preprocess_for_test(opt.dataset_root)
end = time.time()
print('Pre-process for test completed !')
print('Pre-process time : %d min' % int((end-start)/60))

opt.batchSize = 1  # set batchSize = 1 for testing

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#testing images = %d' % dataset_size)

model = torch.load(os.path.join('.', opt.checkpoints_dir, opt.name, str(opt.which_epoch) + '.pth'))
model.BasesNet.eval()

accuracies = []
losses = []

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    print(i)
    accuracy, loss = model.test(data)
    accuracies.append(accuracy)
    losses.append(loss)

accuracy = sum(accuracies)/len(accuracies)
loss = sum(losses)/len(losses)

print(opt.mode + ' accuracy is: ' + str(accuracy))
print(opt.mode + ' loss is: ' + str(loss))

# 音频分离
# ...
# 音频分离结束，dataset_root/sep.json 中保存分离文件的路径
# 定位，将结果保存到 result.json 文件中
import json
with open(os.path.join(opt.dataset_root, 'locations.json'),'r') as f:
    locations = json.load(f)
with open(os.path.join(opt.dataset_root, 'result.json'),'r') as f:
    seperations = json.load(f)
for file in seperations.keys():
    locate = locations[file]
    locate0 = locate[names.index(seperations[file][0]['label'])]  # 根据乐器名字查找对应位置
    locate1 = locate[names.index(seperations[file][1]['label'])]
    seperations[k][0]['position'] = int(locate0 > locate1)
    seperations[k][1]['position'] = 1 - seperations[k][0]['position']
with open(os.path.join(opt.dataset_root, 'result.json'),'w') as f:
    json.dump(seperations, f)