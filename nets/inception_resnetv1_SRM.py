import torch
from torch import nn
from torch.nn.parameter import Parameter

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d,self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SRMLayer(nn.Module):
    def __init__(self, channel):
        super(SRMLayer, self).__init__()

        self.cfc = Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)

        self.bn = nn.BatchNorm2d(channel)
        self.activation = nn.Sigmoid()

        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.size()

        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()

        t = torch.cat((channel_mean, channel_std), dim=2)
        return t

    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1

        z_hat = self.bn(z)
        g = self.activation(z_hat)

        return g

    def forward(self, x):
        # B x C x 2
        t = self._style_pooling(x)

        # B x C x 1 x 1
        g = self._style_integration(t)

        return x * g

class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super(Block35,self).__init__()

        self.scale = scale
        self.SRM=SRMLayer(256)
        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        # self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.conv3d = nn.Conv2d(352, 96, kernel_size=1, stride=1)
        # self.linear=nn.Linear(192,256)
        self.conv2d=nn.Conv2d(96,256,kernel_size=1,stride=1)
        
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.SRM(x)
        out = torch.cat((x0, x1, x2,x3), 1)
        # out=torch.cat((x0,x1,x2),1)
        # print(out.shape,"/////")
        out=self.conv3d(out)
        # out=self.linear(out)
        out = self.conv2d(out)        
        out = out * self.scale + x
        # print("hereherehere")
        out = self.relu(out)
        return out


class Block17(nn.Module):
    def __init__(self, scale=1.0):
        super(Block17,self).__init__()

        self.scale = scale
        
        self.SRM=SRMLayer(896)

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
        )
        self.conv3d=nn.Conv2d(1152,256,kernel_size=1, stride=1)
        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2=self.SRM(x)
        # out = torch.cat((x0, x1), 1)
        # print(out.shape,"....")
        out = torch.cat((x0, x1,x2), 1)
        # print(out.shape,"heheheh")
        out=self.conv3d(out)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        super(Block8,self).__init__()

        self.scale = scale
        self.noReLU = noReLU
        
        self.SRM=SRMLayer(1792)
        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
        )
        self.conv3d=nn.Conv2d(2176, 384, kernel_size=1, stride=1)
        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2=self.SRM(x)
        out = torch.cat((x0, x1,x2), 1)
        # print(out.shape,"少时诵诗书")
        out=self.conv3d(out)
        out = self.conv2d(out)
        out = out * self.scale + x
        # print("herehere")
        if not self.noReLU:
            out = self.relu(out)
        return out

class Mixed_6a(nn.Module):
    def __init__(self,scale=1.0, noReLU=False):
        super(Mixed_6a,self).__init__()

        self.scale=scale
        self.noReLU=False

        # self.SRM=SRMLayer(256)
        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

        # self.branch3=nn.Conv2d(256,192,kernel_size=3,stride=1)
        # self.conv2d = nn.Conv2d(896, 192, kernel_size=1, stride=1)
    def forward(self, x):
        x0 = self.branch0(x)
        # print("x0",x0.shape)
        x1 = self.branch1(x)
        # print("x1",x1.shape)
        x2 = self.branch2(x)
        # x3=self.SRM(x)
        # print("x",x.shape)
        # print("x2",x2.shape)
        # x3=self.branch3(x)
        # print("x3",x3.shape)
        out = torch.cat((x0, x1, x2), 1)
        # print("....",out.shape)
        # out = torch.cat((x0, x1, x2,x3), 1)

        # print("out",out.shape)

        # out=out*self.scale+x3
        # out = self.conv2d(out)
        # out = out * self.scale + x
        # if not self.noReLU:
        #     out = self.relu(out)
        return out


class Mixed_7a(nn.Module):
    def __init__(self,scale=1.0, noReLU=False):
        super(Mixed_7a,self).__init__()

        self.scale=scale
        self.noReLU=noReLU

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )
        # self.conv2d=nn.Conv2d(512,896,kernel_size=1,stride=1)

        self.branch3 = nn.MaxPool2d(3, stride=2)
        # self.branch4=nn.Conv2d(896,256,kernel_size=1,stride=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        # print(x3.shape)
        # x4=self.branch4(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        # out = self.conv2d(out)
        # out = out *self.scale+ x
        # if not self.noReLU:
        #     out = self.relu(out)
        # return out
        return out


#通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = x
        x2 = x
        x2 = self.max_pool(x2)
        x1 = self.avg_pool(x1)
        # x1=self.max_pool(x)
        # print(x1.shape, "x2")
        x1 = self.fc1(x1)
        x2 = self.fc1(x2)
        x1 = self.relu1(x1)
        x2 = self.relu1(x2)
        # print(x1.shape, "x3")
        avg_out = self.fc2(x1)
        max_out = self.fc2(x2)
        out = avg_out + max_out
        return self.sigmoid(out)


#空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class InceptionResnetV1_SRM(nn.Module):
    def __init__(self):
        super(InceptionResnetV1_SRM,self).__init__()
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)

        # self.inplanes=350
        # # 网络的第一层加入注意力机制
        # self.ca = ChannelAttention(256)
        # self.sa = SpatialAttention()
        # self.SRM=SRMLayer()

        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        #注意力机制
        # x=self.ca(x)*x
        # x=self.sa(x)*x

        x = self.repeat_1(x)
        # print(x.shape)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        return x

# net=InceptionResnetV1()
# print(net)