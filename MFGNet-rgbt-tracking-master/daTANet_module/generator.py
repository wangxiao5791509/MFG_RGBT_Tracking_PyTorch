import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
# import torchvision.ops as torchops

import math
from torch.autograd import Variable
from ops import * 
import pdb 

from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

from resnet import resnet18 
import numpy as np
import cv2 
import pdb 


def make_conv_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [maxpool2d()]
        else:
            conv = conv2d(in_channels, v)
            layers += [conv, relu(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_deconv_layers(cfg):
    layers = []
    in_channels = 4115   
    for v in cfg:
        if v == 'U':
            layers += [nn.Upsample(scale_factor=2)]
        else:
            deconv = deconv2d(in_channels, v)
            layers += [deconv]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'D': [512, 512, 512, 'U', 512, 512, 512, 'U', 256, 256, 256, 'U', 128, 128, 'U', 64, 64]
}

class _ConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class AdaptiveConv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(AdaptiveConv2d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias)

    def forward(self, input, dynamic_weight):
        # Get batch num
        batch_num = input.size(0)

        # Reshape input tensor from size (N, C, H, W) to (1, N*C, H, W)
        input = input.view(1, -1, input.size(2), input.size(3))

        # Reshape dynamic_weight tensor from size (N, C, H, W) to (1, N*C, H, W)
        dynamic_weight = dynamic_weight.view(-1, 1, dynamic_weight.size(2), dynamic_weight.size(3))

        # Do convolution
        conv_rlt = F.conv2d(input, dynamic_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Reshape conv_rlt tensor from (1, N*C, H, W) to (N, C, H, W)
        conv_rlt = conv_rlt.view(batch_num, -1, conv_rlt.size(2), conv_rlt.size(3))

        return conv_rlt


def encoder():
    return make_conv_layers(cfg['E'])

def decoder():
    return make_deconv_layers(cfg['D'])







#############################################################################
####                        naive RGBT generator  
#############################################################################

class naive_Generator(nn.Module):
    def __init__(self):
        super(naive_Generator, self).__init__()
        self.encoder = resnet18()
        self.decoder = decoder()
        self.mymodules = nn.ModuleList([deconv2d(64, 1, kernel_size=1, padding = 0), nn.Sigmoid()]) 


    def forward(self, tarObject, gray_tarObject, batch_imgClip, batch_grayClip): 

        _, x_2, x_3 = self.encoder(tarObject)  
        _, gray_x_2, gray_x_3 = self.encoder(gray_tarObject)  

        _, frame1_feat2_v, frame1_feat3_v = self.encoder(batch_imgClip[0])  
        _, frame2_feat2_v, frame2_feat3_v = self.encoder(batch_imgClip[1])  
        _, frame3_feat2_v, frame3_feat3_v = self.encoder(batch_imgClip[2])  

        _, frame1_feat2_i, frame1_feat3_i = self.encoder(batch_grayClip[0])  
        _, frame2_feat2_i, frame2_feat3_i = self.encoder(batch_grayClip[1])  
        _, frame3_feat2_i, frame3_feat3_i = self.encoder(batch_grayClip[2])  


        x_3 = nn.functional.interpolate(x_3, size=[x_2.shape[2], x_2.shape[3]])
        target_feats_v = torch.cat((x_2, x_3), dim=1)
        gray_x_3 = nn.functional.interpolate(gray_x_3, size=[gray_x_2.shape[2], gray_x_2.shape[3]])
        target_feats_i = torch.cat((gray_x_2, gray_x_3), dim=1)
        target_feats   = target_feats_v + target_feats_i 


        frame1_feat3_v = nn.functional.interpolate(frame1_feat3_v, size=[frame1_feat2_v.shape[2], frame1_feat2_v.shape[3]])
        frame1_feats_v = torch.cat((frame1_feat2_v, frame1_feat3_v), dim=1)
        frame1_feat3_i = nn.functional.interpolate(frame1_feat3_i, size=[frame1_feat2_i.shape[2], frame1_feat2_i.shape[3]])
        frame1_feats_i = torch.cat((frame1_feat2_i, frame1_feat3_i), dim=1) 
        frame1_feats   = frame1_feats_v + frame1_feats_i 

        frame2_feat3_v = nn.functional.interpolate(frame2_feat3_v, size=[frame2_feat2_v.shape[2], frame2_feat2_v.shape[3]])
        frame2_feats_v = torch.cat((frame2_feat2_v, frame2_feat3_v), dim=1)
        frame2_feat3_i = nn.functional.interpolate(frame2_feat3_i, size=[frame2_feat2_i.shape[2], frame2_feat2_i.shape[3]])
        frame2_feats_i = torch.cat((frame2_feat2_i, frame2_feat3_i), dim=1)
        frame2_feats   = frame2_feats_v + frame2_feats_i 


        frame3_feat3_v = nn.functional.interpolate(frame3_feat3_v, size=[frame3_feat2_v.shape[2], frame3_feat2_v.shape[3]])
        frame3_feats_v = torch.cat((frame3_feat2_v, frame3_feat3_v), dim=1)
        frame3_feat3_i = nn.functional.interpolate(frame3_feat3_i, size=[frame3_feat2_i.shape[2], frame3_feat2_i.shape[3]])
        frame3_feats_i = torch.cat((frame3_feat2_i, frame3_feat3_i), dim=1)
        frame3_feats   = frame3_feats_v + frame3_feats_i 

        ##### 
        feat_temp1 = torch.cat((target_feats, frame1_feats), dim=1) 
        feat_temp2 = torch.cat((frame2_feats, frame3_feats), dim=1) 
        feat_final = torch.cat((feat_temp1, feat_temp2), dim=1) 
        #### feat_final: torch.Size([3, 3072, 19, 19]) 

        # pdb.set_trace() 
        output = self.decoder(feat_final)
        output = self.mymodules[0](output)
        output = self.mymodules[1](output)
        
        return output 


class Recurrent_net(nn.Module):
    def __init__(self, size, in_channel, out_channel):
        super(Recurrent_net, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.vertical = nn.LSTM(input_size=in_channel, hidden_size=256, batch_first=True, bidirectional=True)  # each row
        self.horizontal = nn.LSTM(input_size=512, hidden_size=256, batch_first=True, bidirectional=True)  # each column
        self.conv = nn.Conv2d(512, out_channel, 1)

    def forward(self, *input):
        x = input[0]
        temp = []
        x = torch.transpose(x, 1, 3)  # batch, width, height, in_channel
        for i in range(self.size):
            h, _ = self.vertical(x[:, :, i, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=2)  # batch, width, height, 512
        temp = []
        for i in range(self.size):
            h, _ = self.horizontal(x[:, i, :, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=3)  # batch, height, 512, width
        x = torch.transpose(x, 1, 2)  # batch, 512, height, width
        x = self.conv(x)
        return x


#############################################################################
####              Direction-aware RGBT Target-aware Attention Module 
#############################################################################

class daGenerator(nn.Module):
    def __init__(self):
        super(daGenerator, self).__init__()
        self.encoder = resnet18()
        self.decoder = decoder()
        self.mymodules = nn.ModuleList([deconv2d(64, 1, kernel_size=1, padding = 0), nn.Sigmoid()]) 

        self.conv1x1_1 = nn.Conv2d(3072, 1024, kernel_size = 1, stride =1, padding=0, bias=False)
        self.conv1x1_2 = nn.Conv2d(3072, 19,   kernel_size = 1, stride =1, padding=0, bias=False)

        self.spatial_renet  = Recurrent_net(19, 1024, 1024)
        self.temporal_renet = Recurrent_net(19, 19,   19)

    def forward(self, tarObject, gray_tarObject, batch_imgClip, batch_grayClip): 

        _, x_2, x_3 = self.encoder(tarObject)  
        _, gray_x_2, gray_x_3 = self.encoder(gray_tarObject)  

        ## batch_imgClip: torch.Size([10, 3, 3, 300, 300]) 
        _, frame1_feat2_v, frame1_feat3_v = self.encoder(batch_imgClip[:, 0])  ## torch.Size([10, 256, 19, 19]) 
        _, frame2_feat2_v, frame2_feat3_v = self.encoder(batch_imgClip[:, 1])  
        _, frame3_feat2_v, frame3_feat3_v = self.encoder(batch_imgClip[:, 2])  

        _, frame1_feat2_i, frame1_feat3_i = self.encoder(batch_grayClip[:, 0])  
        _, frame2_feat2_i, frame2_feat3_i = self.encoder(batch_grayClip[:, 1])  
        _, frame3_feat2_i, frame3_feat3_i = self.encoder(batch_grayClip[:, 2])  

        x_3 = nn.functional.interpolate(x_3, size=[x_2.shape[2], x_2.shape[3]])
        target_feats_v = torch.cat((x_2, x_3), dim=1)
        gray_x_3 = nn.functional.interpolate(gray_x_3, size=[gray_x_2.shape[2], gray_x_2.shape[3]])
        target_feats_i = torch.cat((gray_x_2, gray_x_3), dim=1)
        target_feats   = target_feats_v + target_feats_i 


        frame1_feat3_v = nn.functional.interpolate(frame1_feat3_v, size=[frame1_feat2_v.shape[2], frame1_feat2_v.shape[3]])
        frame1_feats_v = torch.cat((frame1_feat2_v, frame1_feat3_v), dim=1)
        frame1_feat3_i = nn.functional.interpolate(frame1_feat3_i, size=[frame1_feat2_i.shape[2], frame1_feat2_i.shape[3]])
        frame1_feats_i = torch.cat((frame1_feat2_i, frame1_feat3_i), dim=1) 
        frame1_feats   = frame1_feats_v + frame1_feats_i 

        frame2_feat3_v = nn.functional.interpolate(frame2_feat3_v, size=[frame2_feat2_v.shape[2], frame2_feat2_v.shape[3]])
        frame2_feats_v = torch.cat((frame2_feat2_v, frame2_feat3_v), dim=1)
        frame2_feat3_i = nn.functional.interpolate(frame2_feat3_i, size=[frame2_feat2_i.shape[2], frame2_feat2_i.shape[3]])
        frame2_feats_i = torch.cat((frame2_feat2_i, frame2_feat3_i), dim=1)
        frame2_feats   = frame2_feats_v + frame2_feats_i 


        frame3_feat3_v = nn.functional.interpolate(frame3_feat3_v, size=[frame3_feat2_v.shape[2], frame3_feat2_v.shape[3]])
        frame3_feats_v = torch.cat((frame3_feat2_v, frame3_feat3_v), dim=1)
        frame3_feat3_i = nn.functional.interpolate(frame3_feat3_i, size=[frame3_feat2_i.shape[2], frame3_feat2_i.shape[3]])
        frame3_feats_i = torch.cat((frame3_feat2_i, frame3_feat3_i), dim=1)
        frame3_feats   = frame3_feats_v + frame3_feats_i 

        ##### 
        feat_temp1 = torch.cat((target_feats, frame1_feats), dim=1) 
        feat_temp2 = torch.cat((frame2_feats, frame3_feats), dim=1) 
        feat_final = torch.cat((feat_temp1, feat_temp2), dim=1) 
        #### feat_final: torch.Size([3, 3072, 19, 19]) 

        feat_temp = self.conv1x1_1(feat_final) ## torch.Size([3, 1024, 19, 19]) 
        feat_encoded1 = self.spatial_renet(feat_temp)


        feat_encoded2 = self.conv1x1_2(feat_final) ## torch.Size([3, 19, 19, 19]) 
        feat_encoded2 = torch.transpose(feat_encoded2, 1, 2) 
        feat_encoded2 = self.temporal_renet(feat_encoded2) 
        feat_encoded2 = torch.transpose(feat_encoded2, 1, 2) 

        feat_final1 = torch.cat((feat_encoded1, feat_encoded2), dim=1) 
        feat_final1 = torch.cat((feat_final1, feat_final), dim=1)  

        # pdb.set_trace() 
        output = self.decoder(feat_final1)
        output = self.mymodules[0](output)
        output = self.mymodules[1](output)
        
        return output 
