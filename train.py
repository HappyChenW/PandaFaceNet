from cmath import log
from turtle import back
# from nets.facenet4 import Facenet as Facenet4  # mobileFacenet
# from nets.facenet3 import Facenet as Facenet3  # facenet+注意力机制
# from nets.facenet2 import Facenet as Facenet2  # facenet+深度卷积
# from nets.facenet_Deep2 import Facenet as Facenet_Deep2
from nets.facenet_Deep import Facenet as Facenet_Deep
# from nets.facenet5 import Facenet as Facenet5
from utils.LFWdataset import LFWDataset
from utils.eval_metrics import evaluate
from utils.dataloader import FacenetDataset, dataset_collate
from nets.facenet_training import triplet_loss, LossHistory, weights_init
from nets.facenet import Facenet
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import os
import time
from math import sqrt

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
# from ArcFace.ArcFace_loss import *


# torch.cuda.set_device(0)
# os.environ["CUDA_VISIBLE_DEVICES"]="3"




def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_ont_epoch(model, loss, epoch, epoch_size, gen, val_epoch_size, gen_val, Epoch, test_loader, cuda):
    # def fit_ont_epoch(model, loss, epoch, epoch_size, gen, val_epoch_size, gen_val, Epoch,cuda):
    total_triple_loss = 0
    total_CE_loss = 0
    total_accuracy = 0

    val_total_triple_loss = 0
    val_total_CE_loss = 0
    val_total_accuracy = 0

    net.train()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, labels = batch
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(
                        images).type(torch.FloatTensor)).cuda()
                    labels = Variable(torch.from_numpy(labels).long()).cuda()
                else:
                    images = Variable(torch.from_numpy(
                        images).type(torch.FloatTensor))
                    labels = Variable(torch.from_numpy(labels).long())

            optimizer.zero_grad()
            before_normalize, outputs1 = model.forward_feature(images)
            outputs2 = model.forward_classifier(before_normalize)
           
            _triplet_loss = loss(outputs1, Batch_size)
            _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
            _loss = _triplet_loss + _CE_loss
            # _loss = _triplet_loss

            _loss.backward()
            optimizer.step()

            with torch.no_grad():
                accuracy = torch.mean((torch.argmax(
                    F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

            total_accuracy += accuracy.item()
            total_triple_loss += _triplet_loss.item()
            total_CE_loss += _CE_loss.item()

            pbar.set_postfix(**{'total_triple_loss': total_triple_loss / (iteration + 1),
                                'total_CE_loss': total_CE_loss / (iteration + 1),
                                'accuracy': total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    net.eval()
    print('Start Validation')
    with tqdm(total=val_epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= val_epoch_size:
                break
            images, labels = batch
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(
                        images).type(torch.FloatTensor)).cuda()
                    labels = Variable(torch.from_numpy(labels).long()).cuda()
                else:
                    images = Variable(torch.from_numpy(
                        images).type(torch.FloatTensor))
                    labels = Variable(torch.from_numpy(labels).long())

                optimizer.zero_grad()
               
                before_normalize, outputs1 = model.forward_feature(images)
                outputs2 = model.forward_classifier(before_normalize)

               
                _triplet_loss = loss(outputs1, Batch_size)
                _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
                # _CE_loss=nn.
                _loss = _triplet_loss + _CE_loss
                # _loss = _triplet_loss

                accuracy = torch.mean((torch.argmax(
                    F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

                val_total_accuracy += accuracy.item()
                val_total_triple_loss += _triplet_loss.item()
                val_total_CE_loss += _CE_loss.item()

            pbar.set_postfix(**{'val_total_triple_loss': val_total_triple_loss / (iteration + 1),
                                'val_total_CE_loss': val_total_CE_loss / (iteration + 1),
                                'val_accuracy': val_total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Validation')
    loss_history.append_loss(accuracy, (total_triple_loss+total_CE_loss)/(
        epoch_size+1), (val_total_triple_loss+val_total_CE_loss)/(val_epoch_size+1))
#
    print('Epoch:' + str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f' %
          ((total_triple_loss+total_CE_loss)/(epoch_size+1)))
    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'model_path/Epoch%d-Total_Loss%.4f.pth-Val_Loss%.4f.pth' % ((epoch+1),
                                                                                                      (total_triple_loss+total_CE_loss)/(
        epoch_size+1),
        (val_total_triple_loss+val_total_CE_loss)/(val_epoch_size+1)))
    print("开始进行LFW数据集的测试")
    labels, distances = [], []
    for _, (data_a, data_p, label) in enumerate(test_loader):
        with torch.no_grad():
            data_a, data_p = data_a.type(
                torch.FloatTensor), data_p.type(torch.FloatTensor)
            if cuda:
                data_a, data_p = data_a.cuda(), data_p.cuda()
            data_a, data_p, label = Variable(
                data_a), Variable(data_p), Variable(label)
            out_a, out_p = model(data_a), model(data_p)
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    _, _, accuracy, _, _, _, _ = evaluate(distances, labels)
    print('LFW_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))

    # loss_history.append_loss(np.mean(accuracy), (total_triple_loss + total_CE_loss) / (epoch_size + 1),(val_total_triple_loss + val_total_CE_loss) / (val_epoch_size + 1))

    return (val_total_triple_loss + val_total_CE_loss)/(val_epoch_size+1)


if __name__ == "__main__":

   
   

    # 使用学习率衰减      
# 
# 
    
    log_dir="model_path"
    annotation_path = 'train.txt'
   
    num_classes = get_num_classes(annotation_path)
    #--------------------------------------#
    #   输入图片大小
    #--------------------------------------#
    
    input_shape = [350, 350, 3]
   
   
    #--------------------------------------#
    #   主干特征提取网络的选择
    
    #   inception_resnetv1
  
    
    backbone = "inception_resnetv1"
   
    #--------------------------------------#
    #   Cuda的使用
    #--------------------------------------#
    Cuda = True

    
    model = Facenet_Deep(num_classes=num_classes, backbone=backbone)
    
    print(model, "----------------------------------------------------------------------")
    weights_init(model)
    #-------------------------------------------#
    #   权值文件的下载请看README
    #   权值和主干特征提取网络一定要对应
    #-------------------------------------------#
    
    model_path="model_path\googlenet-1378be20.pth"

    # 加快模型训练的效率
    print('Loading weights into state dict...')
    print(model_path)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
   
    pretrained_dict = {k: v for k, v in pretrained_dict.items()}
    model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    model.load_state_dict(model_dict, strict=False)

    net = model.train()
    net = net.to(device)
   
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    loss = triplet_loss()
    # loss=loss.to(device)
    loss_history = LossHistory(log_dir)
#
    LFW_loader = torch.utils.data.DataLoader(LFWDataset(dir="txt/find2/", pairs_path="lfw.txt", image_size=input_shape),
                                             batch_size=1, shuffle=True)

    #-------------------------------------------------------#
    #   0.05用于验证，0.95用于训练
    #-------------------------------------------------------#
    val_split = 0.05
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

   
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#2
    if True:
        lr = 0.001
      
        Batch_size = 4
        Init_Epoch = 0
        Interval_Epoch = 100

        optimizer = optim.Adam(net.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, verbose=True, min_lr=1e-6)

        train_dataset = FacenetDataset(
            input_shape, lines[:num_train], num_train, num_classes)
        val_dataset = FacenetDataset(
            input_shape, lines[num_train:], num_val, num_classes)

        gen = DataLoader(train_dataset, batch_size=Batch_size, pin_memory=True, num_workers=4,
                         drop_last=True, collate_fn=dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, pin_memory=True, num_workers=4,
                             drop_last=True, collate_fn=dataset_collate)

        epoch_size = max(1, num_train//Batch_size)
        val_epoch_size = max(1, num_val//Batch_size)
        print(val_epoch_size)

        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch, Interval_Epoch):
            _loss = fit_ont_epoch(model, loss, epoch, epoch_size, gen,
                                  val_epoch_size, gen_val, Interval_Epoch, LFW_loader, Cuda)
            # _loss = fit_ont_epoch(model, loss, epoch, epoch_size, gen, val_epoch_size, gen_val, Interval_Epoch,Cuda)

            lr_scheduler.step(_loss)

    