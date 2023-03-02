import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image


from nets.facenet import Facenet as facenet
from nets.facenet_Deep import Facenet as facenet_deep

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和backbone需要修改！
#--------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Facenet(object):
    _defaults = {
        
        #此电脑
        # "model_path":"model_path\Epoch100-Total_Loss0.0725.pth-Val_Loss0.0075.pth",
        "model_path":"",
        
        "input_shape"   : (350, 350, 3),
       
        # "backbone"      : "inception_resnetv1",
        "backbone":"shufflenet"
      
        "cuda"          : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Facenet
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()
        
    def generate(self):
        # 载入模型
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = facenet(backbone=self.backbone, mode="predict")
        # model=facenet_deep(backbone=self.backbone,mode="predict")
        model.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        self.net = model.eval()

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()
            
        print('{} model loaded.'.format(self.model_path))
    
    def letterbox_image(self, image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        if self.input_shape[-1]==1:
            new_image = new_image.convert("L")
        return new_image
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image_1, image_2):
        #---------------------------------------------------#
        #   图片预处理，归一化
        #---------------------------------------------------#
        with torch.no_grad():
            image_1 = self.letterbox_image(image_1, [self.input_shape[1], self.input_shape[0]])
            image_2 = self.letterbox_image(image_2, [self.input_shape[1], self.input_shape[0]])
            
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(np.asarray(image_1).astype(np.float64)/255,(2,0,1)),0)).type(torch.FloatTensor)
            photo_2 = torch.from_numpy(np.expand_dims(np.transpose(np.asarray(image_2).astype(np.float64)/255,(2,0,1)),0)).type(torch.FloatTensor)
            
            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            output1 = self.net(photo_1).cpu().numpy()
            output2 = self.net(photo_2).cpu().numpy()
           
            
            #---------------------------------------------------#
            #   计算二者之间的距离
            #---------------------------------------------------#
            l1 = np.linalg.norm(output1-output2, axis=1)
        
        
        return l1
