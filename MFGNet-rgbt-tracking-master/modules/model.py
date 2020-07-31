import os
import scipy.io
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import time
import sys
sys.path.insert(0,'./roi_align')
from roi_align import RoIAlignAvg,RoIAlignMax
import pdb 
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple



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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
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


def append_params(params, module, prefix):
    for child in module.children():
        for k,p in child._parameters.items():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=0.0001, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(2.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(2.0).pow(self.beta)
        x = x.div(div)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)




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




class MDNet(nn.Module):
    def __init__(self, model_path=None,K=1):
        super(MDNet, self).__init__()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2)
                                        )),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2,dilation=1),
                                        nn.ReLU(),
                                        LRN(),
                                        )),

                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1,dilation=3),
                                        nn.ReLU(),
                                        )),
                ('fc4',   nn.Sequential(nn.Linear(512 * 3 * 3 * 2, 512),
                                        nn.ReLU())),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU()))]))

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 2)) for _ in range(K)]) 

        self.sigmoid = nn.Sigmoid()

        self.roi_align_model = RoIAlignMax(3, 3, 1. / 8) 

        self.conv1x1_Tk = nn.Conv2d(1024, 512, 1, 1) 
        self.conv1x1_Tq = nn.Conv2d(1024, 9,   1, 1)
        self.conv1x1_Vk = nn.Conv2d(1024, 512, 1, 1)
        self.conv1x1_Vq = nn.Conv2d(1024, 9,   1, 1)

        self.conv1x1_Tk = self.conv1x1_Tk.cuda() 
        self.conv1x1_Tq = self.conv1x1_Tq.cuda() 
        self.conv1x1_Vk = self.conv1x1_Vk.cuda() 
        self.conv1x1_Vq = self.conv1x1_Vq.cuda() 

        self.channel_attention = ChannelAttention(1024)
        self.spatial_attention = SpatialAttention()


        # self.BatchNorm2D = nn.BatchNorm2d(100)

        self.receptive_field = 75.  # it is receptive fieald that a element of feat_map covers. feat_map is bottom layer of ROI_align_layer

        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d'%(k))
        for name, module in self.conv1x1_Tk.named_children():
            append_params(self.params, module, name)
        for name, module in self.conv1x1_Tq.named_children():
            append_params(self.params, module, name)
        for name, module in self.conv1x1_Vk.named_children():
            append_params(self.params, module, name)
        for name, module in self.conv1x1_Vq.named_children():
            append_params(self.params, module, name)
        for name, module in self.channel_attention.named_children():
            append_params(self.params, module, name)
        for name, module in self.spatial_attention.named_children():
            append_params(self.params, module, name)



    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False


    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params




    ###########################################################################################
    ####                        the forward function 
    ###########################################################################################

    def forward(self, x_v, x_i, k=0, in_layer='conv1', out_layer='fc6'):

        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x_v = module(x_v) 
                x_i = module(x_i) 
                

                if name == "conv3": 

                    rgbt_feats = torch.cat((x_v, x_i), dim=1)   ## torch.Size([1, 192, 62, 91]) 

                    # pdb.set_trace() 

                    rgbt_feats = self.channel_attention(rgbt_feats) * rgbt_feats
                    rgbt_feats = self.spatial_attention(rgbt_feats) * rgbt_feats
                    
                    Tk_feats = self.conv1x1_Tk(rgbt_feats)  ## torch.Size([1, 96, 117, 71])
                    Tq_feats = self.conv1x1_Tq(rgbt_feats)  ## torch.Size([1, 9, 117, 71]) 
                    Vk_feats = self.conv1x1_Vk(rgbt_feats) 
                    Vq_feats = self.conv1x1_Vq(rgbt_feats) 
                    
                    # pdb.set_trace() 

                    Tk_feats = torch.squeeze(Tk_feats, dim=0)
                    Tk_feats = Tk_feats.view(-1, Tk_feats.shape[1]*Tk_feats.shape[2])   ## torch.Size([96, 4150]) 

                    Tq_feats = torch.squeeze(Tq_feats, dim=0)
                    Tq_feats = Tq_feats.view(-1, Tq_feats.shape[1]*Tq_feats.shape[2])

                    Vk_feats = torch.squeeze(Vk_feats, dim=0)
                    Vk_feats = Vk_feats.view(-1, Vk_feats.shape[1]*Vk_feats.shape[2])

                    Vq_feats = torch.squeeze(Vq_feats, dim=0)
                    Vq_feats = Vq_feats.view(-1, Vq_feats.shape[1]*Vq_feats.shape[2])

                    #### T_output.shape: torch.Size([96, 9]) 
                    T_output = torch.matmul(Tk_feats, torch.transpose(Tq_feats, 1, 0)) 
                    V_output = torch.matmul(Vk_feats, torch.transpose(Vq_feats, 1, 0))  

                    # pdb.set_trace() 
                    T_filters = torch.reshape(T_output, (1, T_output.shape[0], 3, 3))  ## (96, 3, 3) 
                    V_filters = torch.reshape(V_output, (1, V_output.shape[0], 3, 3))  ## (96, 3, 3) 


                    # pdb.set_trace() 

                    adaptive_conv_T = AdaptiveConv2d(x_i.size(1), x_i.size(1), 3, padding=1, groups=x_i.size(1), bias=False)
                    adaptive_conv_V = AdaptiveConv2d(x_v.size(1), x_v.size(1), 3, padding=1, groups=x_v.size(1), bias=False)

                    dynamic_T_feats = adaptive_conv_T(x_v, T_filters)
                    dynamic_V_feats = adaptive_conv_V(x_i, V_filters)

                    dynamic_T_feats = self.sigmoid(dynamic_T_feats) 
                    dynamic_V_feats = self.sigmoid(dynamic_V_feats) 

                    x_v = x_v + dynamic_V_feats  
                    x_i = x_i + dynamic_T_feats 

                    fuse_x_v_i = torch.cat((x_v, x_i), dim=1)
                    
                    # pdb.set_trace()

                    # augmented_feats, p1 = self.attn1(fuse_x_v_i)

                if name == out_layer:
                    return x_v, x_i, fuse_x_v_i  


        # pdb.set_trace() 

        x_v = self.branches[k](x_v) 


        if out_layer=='fc6':
            return x_v
        elif out_layer=='fc6_softmax':
            return F.softmax(x_v)



    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i*4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:,0])

    def trainSpatialTransform(self, image, bb):

        return


class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score)[:,1]
        neg_loss = -F.log_softmax(neg_score)[:,0]

        loss = (pos_loss.sum() + neg_loss.sum())/(pos_loss.size(0) + neg_loss.size(0))
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):

        pos_correct = (pos_score[:,1] > pos_score[:,0]).sum().float()
        neg_correct = (neg_score[:,1] < neg_score[:,0]).sum().float()

        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.item(), neg_acc.item()


class Precision():
    def __call__(self, pos_score, neg_score):

        scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0)+1e-8)

        return prec.item()



