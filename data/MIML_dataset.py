#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from data.base_dataset import BaseDataset
import h5py
import numpy as np
import sklearn.preprocessing
import torchvision.transforms as transforms
import torch

def normalizeBases(bases, norm):
    if norm == "l1":
        return sklearn.preprocessing.normalize(bases, norm='l1', axis=0)
    elif norm == "l2":
        return sklearn.preprocessing.normalize(bases, norm='l2', axis=0)
    elif norm == "max":
        return sklearn.preprocessing.normalize(bases, norm='max', axis=0)
    else:
        return bases

def subsetOfClasses(label, mode):
    #8 music instruments:['accordion','acoustic_guitar','cello','trumpet','flute','xylophone','saxophone','violin']
    indexes=[[401],[402],[486],[513],[558],[642],[776],[889]]
    selected_label = np.zeros(8)
    if mode == 'train':
        for i,class_indexes in enumerate(indexes):
            for index in class_indexes:
                #print("index & label:",index,label[index])
                if label[index]<0.5:
                    selected_label[i]=0
                else:
                    selected_label[i]=1
    elif mode == 'test':
        for i,class_indexes in enumerate(indexes):
            for index in class_indexes:
                selected_label[i] = selected_label[i] + label[index]
            selected_label[i] = selected_label[i] / len(class_indexes)
    else:
        raise ValueError
        
    return selected_label

def subsetOfClassesAnimals(label):
    pass

def subsetOfClassesVehicles(label):
    pass

def subsetOfClassesAll(label):
    pass

def softmax(x, mode):
    assert mode == 'train' or mode == 'test'
    #return x
    return np.exp(x) / np.sum(np.exp(x), axis=0) if mode == 'test' else x

class MIMLDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.bases = []

        #load hdf5 file here
        h5f_path = os.path.join(opt.dataset_root, opt.mode + ".h5")
        h5f = h5py.File(h5f_path, 'r')
        self.bases = h5f['bases'][:]
        self.labels = h5f['labels'][:]
        
    def __getitem__(self, index):
        bases = np.load(self.bases[index].decode("utf-8"))
        if self.opt.selected_classes:
            if self.opt.dataset == 'musicInstruments':
                loaded_label = softmax(subsetOfClasses(np.load(self.labels[index].decode("utf-8")), self.opt.mode), self.opt.mode)
            elif self.opt.dataset == 'animals':
                loaded_label = softmax(subsetOfClassesAnimals(np.load(self.labels[index].decode("utf-8"))))
            elif self.opt.dataset == 'vehicles':
                loaded_label = softmax(subsetOfClassesVehicles(np.load(self.labels[index].decode("utf-8"))))
            elif self.opt.dataset == 'all':
                loaded_label = softmax(subsetOfClassesAll(np.load(self.labels[index].decode("utf-8"))))
        else:
            loaded_label = softmax(np.load(self.labels[index].decode("utf-8")))

        if self.opt.using_multi_labels:
            label = np.zeros(self.opt.L) - 1 #-1 means incorrect labels
            label_index = [np.argmax(loaded_label)]
            if self.opt.mode == 'train':
                label_index = list(set(label_index) | set(np.where(loaded_label >= 0.3)[0]))
            elif self.opt.mode == 'test':
                label_index = list(np.argsort(loaded_label)[-2:])
            else:
                raise ValueError
            for i in range(len(label_index)):
                label[i] = label_index[i]
        else:
            label = np.argmax(loaded_label)

        #perform basis normalization
        bases = normalizeBases(bases, self.opt.norm)
        if self.opt.zeroCenterInput:
            bases = bases * 2 - 1

        if self.opt.isTrain:
            return {'bases': bases, 'label': label}
        else:
            return {'bases': bases, 'label': label}

    def __len__(self):
        return len(self.bases)

    def name(self):
        return 'MIMLDataset'
