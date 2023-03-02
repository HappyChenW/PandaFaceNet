from audioop import bias
import os
# from os import sched_yield
from re import S
# from nets.mobilenetv3 import MobileNetV3_Small
import torch.nn as nn
from torch.nn import functional as F
import torch

from nets.inception_resnetv1 import InceptionResnetV1
from nets.inception_resnetv1_SRM import InceptionResnetV1_SRM
# from nets.mobilenet import MobileNetV1
from nets.inception_resnetv2 import InceptionResnetV2
# from nets.inception_resnetv2 import InceptionResnetV2
from nets.inceptionv3 import InceptionV3
from nets.inceptionv4 import InceptionV4


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.ReLU6(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)

class inceptionv4(nn.Module):
    def __init__(self):
        super(inceptionv4, self).__init__()
        self.model = InceptionV4()

    def forward(self, x):
        # x = self.features(x)
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.mixed_3a(x)
        x = self.model.mixed_4a(x)
        x = self.model.mixed_5a(x)
        x = self.model.features(x)
        
        return x



        


class inceptionv3(nn.Module):
    def __init__(self):
        super(inceptionv3, self).__init__()
        self.model = InceptionV3()

    def forward_preaux(self, x):
        # N x 3 x 299 x 299
        x = self.model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.model.Pool1(x)
        # N x 64 x 73 x 73
        x = self.model.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.model.Pool2(x)
        # N x 192 x 35 x 35
        x = self.model.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.model.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.model.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.model.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6e(x)
        # N x 768 x 17 x 17
        return x

    def forward_postaux(self, x):
        x = self.model.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.model.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.model.Mixed_7c(x)
        # N x 2048 x 8 x 8
        return x

    def forward_features(self, x):
        x = self.model.forward_preaux(x)
        x = self.model.forward_postaux(x)
        return x

    
    def forward(self, x):
        x = self.model.forward_features(x)
     
        return x




class inception_resnetv1(nn.Module):
    def __init__(self):
        super(inception_resnetv1, self).__init__()
        self.model = InceptionResnetV1()
        # self.model=

    def forward(self, x):
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.conv2d_4b(x)
        x = self.model.repeat_1(x)
        # print(x.shape)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_2(x)
        x = self.model.mixed_7a(x)
        x = self.model.repeat_3(x)
        x = self.model.block8(x)
        return x




class inception_resnetv1_SRM(nn.Module):
    def __init__(self):
        super(inception_resnetv1_SRM, self).__init__()
        self.model = InceptionResnetV1_SRM()
        # self.model=

    def forward(self, x):
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.conv2d_4b(x)
        x = self.model.repeat_1(x)
        # print(x.shape)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_2(x)
        x = self.model.mixed_7a(x)
        x = self.model.repeat_3(x)
        x = self.model.block8(x)
        return x





class Facenet(nn.Module):
    def __init__(self, backbone="inception_resnetv1", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train"):
        super(Facenet, self).__init__()
        if backbone == "mobilenet":
            self.backbone = mobilenet()
            flat_shape = 1024
        elif backbone == "inception_resnetv1":
            # self.backbone = inception_resnetv1()
            self.backbone=inception_resnetv1_SRM()                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
            flat_shape = 1792
            # flat_shape=1024
        
        elif backbone=="inceptionv4":
            self.backbone=inceptionv4()
            flat_shape=1536
        elif backbone=="inceptionv3":
            self.backbone = inceptionv3()
            flat_shape = 2048
        
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))
        

        # 深度分离卷积
        # self.conv2 = ConvBlock(flat_shape, 512, 1, 1, 0)
        # self.linear7 = ConvBlock(512, 512, (22, 22), 1, 0, dw=True, linear=True)
        # self.linear7 = ConvBlock(960, 960, (11, 11), 1, 0, dw=True, linear=True)  #shufflenet
        # self.linear7 = ConvBlock(512, 512, (3, 3), 1, 0, dw=True, linear=True)
        # self.linear7 = ConvBlock(512, 512, (7, 7), 1, 0, dw=True, linear=True)
        # self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)


        # self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.linear7=nn.AvgPool2d(7,1)            # shufflenet
            
        # self.Bottleneck1 = nn.Linear(1280, 1024,bias=False)
        # self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        # self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        # self.linear=nn.Linear(30245,flat_shape,bias=False)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size,bias=False)
        self.Dropout1=nn.Dropout(1-dropout_keep_prob)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if mode == "train":
            # self.classifier2=nn.Linear(embedding_size,1024) 
            self.classifier2=nn.Linear(embedding_size,190)      
            # self.Dropout_c=nn.Dropout(1-dropout_keep_prob)   
            # self.classifier = nn.Linear(1024, 10575)
            # self.Dropout2=nn.Dropout(1-dropout_keep_prob)
            # self.classifier = nn.Linear(embedding_size, 31)


    def forward(self, x):
        x = self.backbone(x)
        # print(x.shape,"/////")
        x=self.linear7(x)
        # print(x.shape,".....")
        # x = self.avg(x)
        x = x.view(x.size(0), -1)
        # x = self.Bottleneck1(x)
        # x = self.Dropout(x)
        # x=self.linear(x)
        # print(x.shape,"1111")
        x = self.Bottleneck(x)
        x=self.Dropout1(x)
        x = self.last_bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward_feature(self, x):
        x = self.backbone(x)
        # print(x.shape,"/////1")
        # print(x.shape,".....")
        # x = self.avg(x)
        x=self.linear7(x)
        x = x.view(x.size(0), -1)
        # print(x.shape,".....")
        # x=self.linear(x)
        # x = self.Bottleneck1(x)
        # x = self.Dropout(x)
        # x = self.Dropout(x)
        # print(x.shape,"/////")
        x = self.Bottleneck(x)
        x=self.Dropout1(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        return before_normalize, x

    def forward_classifier(self, x):
        x = self.classifier2(x)
        # x=self.Dropout_c(x)
        # x=self.classifier(x)
        # x=self.Dropout2(x)

        return x
