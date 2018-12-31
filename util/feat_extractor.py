import os
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json

global net
global normalize
global pre_process
global features_blobs
global classes
global weight_softmax
labels_path='labels.json'
global idxs
global names
idxs=[401,402,486,513,558,642,776,889]
names=['accordion','acoustic_guitar','cello','trumpet','flute','xylophone','saxophone','violin']

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def hook_feature(module, input, output):
    global features_blobs
    features_blobs=output.data.cpu().numpy()

def load_model():
    global net
    global normalize
    global pre_process
    global features_blobs
    global classes
    global weight_softmax
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'
    net.eval()
    net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
    pre_process = transforms.Compose([
       transforms.Resize((224,224)),
       transforms.ToTensor(),
       normalize
    ])
    classes = {int(key):value for (key, value)
              in json.load(open(labels_path,'r')).items()}
    if torch.cuda.is_available():
        net=net.cuda()

def get_CAM(imdir,savedir,imname):
    location = np.zeros(8)   # 每种乐器类别的水平位置（列号）
    img = cv2.imread(os.path.join(imdir,imname))
    height, width, _ = img.shape
    for i in range(0, 8):
        CAMs = returnCAM(features_blobs, weight_softmax, [idxs[i]])
        location[i] = np.argmax(CAMs[0]) % width    # 计算列号
        
    return location

def feat_pred(imdir,imname):   # 计算8个乐器类别的概率并返回
    img_pil = Image.open(os.path.join(imdir,imname))
    img_tensor = pre_process(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    if torch.cuda.is_available():
        img_variable=img_variable.cuda()
    logit = net(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    if torch.cuda.is_available():
        h_x=h_x.cpu()
    probs1 = h_x.numpy()
    probs=[]
    for i in range(0, 8):
        probs.append(probs1[idxs[i]])
    return probs

def feat_pred_by_seg(imdir,imname):   # 将图片从中间分割成左右两个，分别计算8个乐器类别的概率并返回
    img_pil = Image.open(os.path.join(imdir,imname))
    width, height = img_pil.size
    img_pil_left = img_pil.crop((0,0,width//2,height))
    img_pil_right = img_pil.crop((width//2,0,width,height))
    # 处理左边图片
    img_tensor = pre_process(img_pil_left)
    img_variable = Variable(img_tensor.unsqueeze(0))
    if torch.cuda.is_available():
        img_variable=img_variable.cuda()
    logit = net(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    if torch.cuda.is_available():
        h_x=h_x.cpu()
    probs1 = h_x.numpy()
    probs_left=[]
    for i in range(0, 8):
        probs_left.append(probs1[idxs[i]])
    # 处理右边图片
    img_tensor = pre_process(img_pil_right)
    img_variable = Variable(img_tensor.unsqueeze(0))
    if torch.cuda.is_available():
        img_variable=img_variable.cuda()
    logit = net(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    if torch.cuda.is_available():
        h_x=h_x.cpu()
    probs1 = h_x.numpy()
    probs_right=[]
    for i in range(0, 8):
        probs_right.append(probs1[idxs[i]])
    return probs_left, probs_right
