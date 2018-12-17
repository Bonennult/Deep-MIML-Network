#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.utils.data
from data.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = None
    if opt.model == 'MIML':
        from data.MIML_dataset import MIMLDataset
        dataset = MIMLDataset()
    elif opt.model == 'KL_divergence':
        from data.KL_dataset import KLDataset
        dataset = KLDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.model)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)  #目的是传入参数opt
        self.dataset = CreateDataset(opt)     #根据opt.model创建一个数据集类“MIMLDataset/KLdataset”，并初始化（包括读取h5文件，放入self.bases,self.labels）
        self.dataloader = torch.utils.data.DataLoader(  #该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor，后续只需要再包装成Variable即可作为模型的输入
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
