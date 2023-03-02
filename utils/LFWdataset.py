import torchvision.datasets as datasets
import os
import numpy as np
from PIL import Image

def letterbox_image(image, size):
    image = image.convert("RGB")
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

class LFWDataset(datasets.ImageFolder):
    def __init__(self, dir, pairs_path, image_size, transform=None):
        super(LFWDataset, self).__init__(dir,transform)
        self.image_size = image_size
        self.pairs_path = pairs_path
        self.validation_images = self.get_lfw_paths()

    def read_lfw_pairs(self,pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs)

    def get_lfw_paths(self):

        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []

        # print(len(pairs))
        # for i in range(len(pairs)):
        for pair in pairs:
            # pair = pairs[i]
            # print(pairs)
            # print(pair)
            # print(pair[2])
            # print(type(pair[2]),'hdhdhhdhd')
            if pair[2]=='1':
                path0 = os.path.join(pair[0])
                path1 = os.path.join(pair[1])
                issame = True
            #     print(path0)
            #     print(path1)
            #     print("进来了！")
            elif pair[2]=='0':
                path0 = os.path.join(pair[0])
                path1 = os.path.join(pair[1])
            #     print("在这里！")
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
                path_list.append((path0,path1,issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs>0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)
        print(path_list,"hdhdhdh")
        return path_list

    def __getitem__(self, index):
        (path_1,path_2,issame) = self.validation_images[index]
        img1, img2 = Image.open(path_1), Image.open(path_2)
        img1 = letterbox_image(img1,[self.image_size[1],self.image_size[0]])
        img2 = letterbox_image(img2,[self.image_size[1],self.image_size[0]])
        
        img1, img2 = np.array(img1)/255, np.array(img2)/255
        img1 = np.transpose(img1,[2,0,1])
        img2 = np.transpose(img2,[2,0,1])

        return img1, img2, issame

    def __len__(self):
        # print("shshshsh")
        return len(self.validation_images)

# dataset =LFWDataset(dir="D:/chen/facenet-pytorch-main/datasets/pandas/",pairs_path="D:/chen/facenet-pytorch-main/model_data/test_pandas.txt",image_size=[350,350,3])
# print(dataset)
# test_loader = LFWDataset(dir="D:/exin/best-pytorch-main/best/dataset/face/test2",pairs_path="D:/exin/best-pytorch-main/best/dataset/lfw.txt",image_size=[350,350,3])
# print(test_loader)
