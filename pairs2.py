import sys
import os
import random
import time
import itertools
import pdb
import argparse
#src = '../../datasets/lfw2'
#dst = open('../../datasets/lfw/train.txt', 'a')
parser = argparse.ArgumentParser(description='generate image pairs')
# general
parser.add_argument('--data-dir', default='cut/', help='')
parser.add_argument('--data2-dir', default='cut/', help='')
parser.add_argument('--outputtxt', default='new_find.txt', help='path to save.')
parser.add_argument('--num-samepairs',default=10000000)
args = parser.parse_args()
cnt = 0
same_list = []
diff_list = []
list1 = []
list2 = []
folders_1 = os.listdir(args.data_dir)
folders_2=os.listdir(args.data2_dir)
# print(folders_1)
dst = open(args.outputtxt, 'a')
count = 0
dst.writelines('\n')
# 产生相同的图像对
for folder in folders_1:
    sublist1=[]
    imgs=os.listdir(os.path.join(args.data_dir, folder))
    for img in imgs:
        img_root_path1=os.path.join(args.data_dir, folder, imgs[0])
    

    sublist1.append(img_root_path1)
    list1.append(img_root_path1)
# print(list1,"dhdhdhdhd")
# print(len(list1),"\\\\\\\\")

for folder2 in folders_2:
    sublist2=[]
    imgs2=os.listdir(os.path.join(args.data2_dir, folder2))
    for img in imgs2:
        img_root_path2=os.path.join(args.data2_dir, folder2, imgs2[0])

    sublist2.append(img_root_path2)
    list2.append(img_root_path2)
# print(list2[0],"dhdhdhdhd")
# print(len(list2),"\\\\\\\\")
    #     sublist1= []
    # # same_list = []
    #     sublist2=[]
    #     imgs = os.listdir(os.path.join(args.data_dir, folder))
    #     imgs2=os.listdir(os.path.join(args.data_dir, folders))
    # # print(imgs)
    #     for img in imgs:
    #         img_root_path1= os.path.join(args.data_dir, folder, img)
    #     # print(img_root_path)
    #     sublist.append(img_root_path)
    #     list1.append(img_root_path)
    #     # print(sublist)
    #     # print(list1,'list1')
for i in range(len(list2)):
    for j in range(len(list1)):
        count=[]
        count.append(list2[i])
        # count.append(" ")
        count.append(list1[j])
        if list1[j]==list2[i]:
            continue
        # count.append(" ")
        # count.append("0")
        # diff_list.append(list2[i],li)
        # print("count!!!!",count)
        else:
            diff_list.append(count)
print("diff_list",len(diff_list))
for m in range(len(diff_list)):
    dst.writelines(diff_list[m][0]+' '+diff_list[m][1]+' '+'0'+'\n')
    # for item in itertools.combinations(sublist, 2):
    #     for name in item:
    #         same_list.append(name)
    #         # print(len(same_list),'sssss')
    # if len(same_list) > 0 and len(same_list) < 91:
    #     print('ll')
    #     for j in range(0, len(same_list), 2):
    #             print("进来了！！")
    #             if count < int(args.num_samepairs):#数量可以修改
    #                 dst.writelines(same_list[j] + ' ' + same_list[j+1]+ ' ' + '1' + '\n')
    #                 count += 1
#     if count >= int(args.num_samepairs):
#         break
# list2 = list1.copy()
# # 产生不同的图像对
# diff = 0
# print(count)
# # 如果不同的图像对远远小于相同的图像对，则继续重复产生，直到两者相差很小
# while True:
#     random.seed(time.time() * 100000 % 10000)
#     random.shuffle(list2)
#     for p in range(0, len(list2) - 1, 2):
#         if list2[p] != list2[p + 1]:
#             dst.writelines(list2[p] + ' ' + list2[p + 1] + ' ' + '0'+ '\n')
#             diff += 1
#             if diff >= 5000:
#                 break
#             print(diff,'diffff')
#     if diff < 5000:
#         #print('--')
#         continue
#     else:
#         break