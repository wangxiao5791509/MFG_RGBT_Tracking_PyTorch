import cv2
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import pdb 

from PIL import Image

def to_variable(x,requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad)

def show(img):
    #print(img.shape)
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(img)
    s = np.array(pilImg)
    plt.figure()
    plt.imshow(s)
    
def show_gray(img):
    print(img.shape)
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(img)
    s = np.array(pilImg)
    plt.figure()
    plt.imshow(s)
    
def save_gray(img, path):
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(img)
    print('Image saved to ', path)
    pilImg.save(path)




def predict(model, img, validation_targetObject):
    to_tensor = transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0.
    im = to_tensor(img)
    val_targetObject = to_tensor(validation_targetObject)
    #show(im)
    inp = to_variable(im.unsqueeze(0), False)
    inp = nn.functional.interpolate(inp, size=[300, 300])

    val_targetObject_ = to_variable(val_targetObject.unsqueeze(0), False) 
    val_targetObject_ = nn.functional.interpolate(val_targetObject_, size=[100, 100]) 

    #print(inp.size())

    out = model(inp, val_targetObject_)
    out = nn.functional.interpolate(out, size=[im.shape[1], im.shape[2]]) 

    map_out = out.cpu().data.squeeze(0)
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(map_out)
    dynamic_atttentonMAP = np.asarray(pilImg)

    return dynamic_atttentonMAP 

