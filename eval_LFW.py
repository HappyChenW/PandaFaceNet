# from __future__ import print_function

import argparse
import collections
from operator import mod
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Function, Variable
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from nets.facenet import Facenet
from nets.facenet_Deep import Facenet as Facenet_deep
# from testface import FaceNet
from utils.eval_metrics import evaluate
from utils.LFWdataset import LFWDataset
from math import sqrt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

      

    

def mytest(test_loader, model):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        with torch.no_grad():
            data_a, data_p = data_a.type(
                torch.FloatTensor), data_p.type(torch.FloatTensor)
            if cuda:
                data_a, data_p = data_a.cuda(), data_p.cuda()
            data_a, data_p, label = Variable(data_a), \
                Variable(data_p), Variable(label)
            out_a, out_p = model(data_a), model(data_p)
         
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
           
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())
        if batch_idx % log_interval == 0:
            pbar.set_description('Test Epoch: [{}/{} ({:.0f}%)]'.format(
                batch_idx * batch_size, len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(
        distances, labels)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Accuracy:%2.5f' % (np.max(accuracy)))
    # print('All_Accuracy:%2.5f'%(accuracy))
    # print("hdhdhdhdh:%2.5f"%(accuracy))
    print('Best_thresholds: %2.5f' % best_thresholds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    plot_roc(fpr, tpr, figure_name="roc_test.png")


def plot_roc(fpr, tpr, figure_name="roc.png"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(figure_name, dpi=fig.dpi)


if __name__ == "__main__":
    #--------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet
    #   inception_resnetv1
    #--------------------------------------#
    
    backbone = "inception_resnetv1"
    
   
    input_shape = [350, 350, 3]
    #--------------------------------------#
    #   训练好的权值文件
    #--------------------------------------#
    

  
   


    model_path="model_path\Epoch100-Total_Loss0.0725.pth-Val_Loss0.0075.pth"
   
    #--------------------------------------#
    #   Cuda的使用
    #--------------------------------------#
    cuda = True

    batch_size = 32
    # batch_size=1
    log_interval = 1

    
    test_loader = torch.utils.data.DataLoader(
        LFWDataset(dir="dataset/train/", pairs_path="new_find_all.txt", image_size=input_shape), batch_size=batch_size, shuffle=False)
    

    print('Loading weights into state dict...')
    print(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
   
    model = Facenet_deep(backbone=backbone, mode="predict")
    # if cuda:
    #     model = torch.nn.DataParallel(model)
    #     cudnn.benchmark = True
    #     model = model.cuda()
    # model = nn.DataParallel(model)
    # model.to(device)
    model.load_state_dict(torch.load(
        model_path, map_location=device), strict=False)
    model = model.eval()

    if cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()

    mytest(test_loader, model)
