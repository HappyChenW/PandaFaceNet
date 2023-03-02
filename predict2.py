from ast import With
from itertools import count
from symbol import with_item
import numpy as np
from PIL import Image
import os
import torch
import math

from facenet import Facenet
same = 0
diff = 0
same_correct = 0
diff_correct = 0
predict = 2
correct = 0
wrong = []

#
if __name__ == "__main__":
    #     # imgPath = "D:/chen/facenet-pytorch-main/datasets/test/2/"
    #     # imgPath2="D:/chen/facenet-pytorch-main/datasets/test/1/"
    #     # file=open("D:/chen/facenet-pytorch-main/1.txt")
    #     # imgPath="D:/chen/InsulatorDataSet/Defective_Insulators/images/"
    #     # imgPath="D:\\chen\\YoloV4_Insulators-main\\predict_imgs\\2\\"
    #     # file_list=os.listdir(imgPath)
    #     file_list2=os.listdir(imgPath2)
    #     file_list.sort()
    #     file_list2.sort()
    #     file_path_list2=[os.path.join(imgPath2,img) for img in file_list2]
    #     file_path_list=[os.path.join(imgPath,img) for img in file_list]
    #     l=len(file_path_list)
    #     l2=len(file_path_list2)
    #     print("有多少张图片？",l,l2)
    #
    #     # if l!=l2:
    model = Facenet()
    p=[]
    count=[NotADirectoryError]
    # with open("lfw6.txt") as f:
    # with open("my_test2.txt") as f:
    # with open("50_1.txt") as f:
    with open("50_2.txt") as f:
    # with open("wild2.txt") as f:
    # with open("newwild.txt") as f:
    # with open("find_all.txt") as f:
    # with open("1.txt") as f:
    # with open("11.txt") as f:
    # with open("new.txt") as f:
    # with open("test_same.txt") as f:
        l = f.readlines()
        len = len(l)
        for i in range(len):
            a = l[i].split(" ")
            print("第几行",i)
            # print(a,"aaaaaaaaaaaaaa")
            image_1 = Image.open(a[0])
            image_2 = Image.open(a[1])
            label = int(a[2])
            if label == 1:
                same += 1
            else:
                diff += 1
            probability = model.detect_image(image_1, image_2)
            # print("number:",i)
            # print(probability)
            # probability=probability*10000000
            # probability=probability*1000
            print(probability)
            p.append(probability)
            if probability <0.94:  # inv1:0.76 inv1_SRM:0.83 densenet:0.02 inv1_SRM_deep:0.84 inv3:0.94 resnet50:0.93
                predict = 1
                # print("是同一只")
                if predict == label:
                    print("是同一只")
                    correct += 1
                    same_correct += 1
                    print("same correct:", same_correct)
                    # print("第几行：",i)
                else:
                    print("出错了！")
                    print(i)
                    count.append(i)
                    wrong.append(i)
                    wrong.append(probability)

            else:
                predict = 0
                if predict == label:
                    correct += 1
                    print("不是同一只")
                    diff_correct += 1
                    print("different correct:", diff_correct)
                else:
                    print("出错了！")
                    print(i)

                    wrong.append(i)
                    # wrong.append(probability)
                # print("不是同一只")
    print("总共:", len)
    print("是同一只总共:", same, "预测正确:", same_correct, "正确率:", same_correct/same)
    print("不是同一只总共:", diff, "预测正确:", diff_correct, "正确率:", diff_correct/diff)
    print("正确率:", (same_correct+diff_correct)/len)
    # print(wrong)
    # print(len)
    # print(p)
    # while True:
    #     image_1 = input('Input image_1 filename:')
    #     try:
    #         image_1 = Image.open(image_1)
    #     except:
    #         print('Image_1 Open Error! Try again!')
    #         continue

    #     image_2 = input('Input image_2 filename:')
    #     try:
    #         image_2 = Image.open(image_2)
    #     except:
    #         print('Image_2 Open Error! Try again!')
    #         continue

    #     probability = model.detect_image(image_1,image_2)
    #     print(probability)
    #     if probability<1.22:
    #         print("是同一只")
    #     else:
    #         print("不是同一只")
