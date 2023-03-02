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
parser.add_argument('--data-dir', default='dataset/train/', help='')
# parser.add_argument('--data2-dir', default='test1/', help='')
parser.add_argument('--outputtxt', default='new_find_all.txt', help='path to save.')
parser.add_argument('--num-samepairs',default=30000000)
args = parser.parse_args()
cnt = 0
same_list = []
diff_list = []
list1 = []
list2 = []
folders_1 = os.listdir(args.data_dir)
# folders_2=os.listdir(args.data2_dir)
# print(folders_1)
dst = open(args.outputtxt, 'a')
count = 0
# dst.writelines('\n')
# 产生相同的图像对
for folder in folders_1:
    sublist = []
    same_list = []
    imgs = os.listdir(os.path.join(args.data_dir, folder))
    # print(imgs)
    for img in imgs:
        img_root_path = os.path.join(args.data_dir, folder, img)
        # print(img_root_path)
        sublist.append(img_root_path)
        list1.append(img_root_path)
        # print(sublist)
        # print(list1,'list1')
    for item in itertools.combinations(sublist, 2):
        for name in item:
            same_list.append(name)
            # print(len(same_list),'sssss')
    if len(same_list) > 0 and len(same_list) < 30000000:
        # print('ll')
        for j in range(0, len(same_list), 2):
                # print("进来了！！")
                if count < int(args.num_samepairs):#数量可以修改
                    # if same_list[j]==same_list[j+1]:
                    dst.writelines(same_list[j] + ' ' + same_list[j+1]+ ' ' + '1' + '\n')
                    count += 1
                    # else:
                        # continue
    if count >= int(args.num_samepairs):
        break
list2 = list1.copy()
# 产生不同的图像对
diff = 0
print(count,"/////////////////////////////")
# 如果不同的图像对远远小于相同的图像对，则继续重复产生，直到两者相差很小
while True:
    random.seed(time.time() * 100000 % 10000)
    random.shuffle(list2)
    for p in range(0, len(list2) - 1, 2):
        if list2[p] != list2[p + 1]:
            dst.writelines(list2[p] + ' ' + list2[p + 1] + ' ' + '0'+ '\n')
            diff += 1
            if diff >= 24630936:
                break
            # print(diff,'diffff')
    if diff < 24630936:
        #print('--')
        continue
    else:
        break