from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from tqdm import tqdm
from torch.autograd import Variable
import torchvision.transforms as transforms
import random
from generator import daGenerator
from utils import *
import pdb 
import os.path

generator_path = './20200525_directionAware_TANet_rgbt_model.pkl'
Generator = daGenerator()
Generator.load_state_dict(torch.load(generator_path))
Generator.cuda()

def to_variable(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad)

counter = 0
start_time = time.time()

attentionSave_path = "/home/wangxiao/experiments/directionAware_rgbt_TANet_module/rgbt_210_Attention/"
					   
dataset_path = "/wangxiao/dataset/RGB-T210/"


video_files = os.listdir(dataset_path) 
video_files.sort()
count = 0 

for videoidx in range(len(video_files)): 
    videoName = video_files[videoidx]     
    already_Done = os.listdir(attentionSave_path)
    
    if videoName in already_Done: 
        print("==>> Skip this video .... ")
    else:
	    dataset_img_path_v = dataset_path + videoName + "/visible/" 
	    dataset_img_files_v = os.listdir(dataset_img_path_v)
	    dataset_img_path_i = dataset_path + videoName + "/infrared/" 
	    dataset_img_files_i = os.listdir(dataset_img_path_i) 

	    dataset_img_files_v.sort()
	    dataset_img_files_i.sort()

	    cursor = 0
	    batch_size = 1 
	    clip_len = 3 
	    size = len(dataset_img_files_v) 
	    to_tensor = transforms.ToTensor()  
	    targetObject_v = torch.zeros(batch_size, 3, 300, 300)
	    targetObject_i = torch.zeros(batch_size, 3, 300, 300) 

	    gt_path = dataset_path + videoName + "/init.txt"
	    gt_files = np.loadtxt(gt_path, delimiter=',')
	    initBBox = gt_files[0] 

	    initImg_path_v = dataset_img_files_v[0] 
	    initImg_path_i = dataset_img_files_i[0] 

	    initImage_v = cv2.imread(dataset_img_path_v + initImg_path_v) 
	    initImage_i = cv2.imread(dataset_img_path_i + initImg_path_i) 

	    tarObject_v = initImage_v[int(initBBox[1]):int(initBBox[1]+initBBox[3]), int(initBBox[0]):int(initBBox[0]+initBBox[2]), :] 
	    tarObject_i = initImage_i[int(initBBox[1]):int(initBBox[1]+initBBox[3]), int(initBBox[0]):int(initBBox[0]+initBBox[2]), :] 

	    tarObject_v = cv2.resize(tarObject_v, (300, 300), interpolation=cv2.INTER_LINEAR)
	    tarObject_i = cv2.resize(tarObject_i, (300, 300), interpolation=cv2.INTER_LINEAR)

	    targetObject_v[0] = to_tensor(tarObject_v) 
	    targetObject_i[0] = to_tensor(tarObject_i) 
	    # cv2.imwrite('./tarObject_v.png', tarObject_v)

	    # pdb.set_trace() 
	    for idx in range(1, len(dataset_img_files_v)):

	        batch_imgClip_v = torch.zeros(batch_size, clip_len, 3, 300, 300) 
	        batch_imgClip_i = torch.zeros(batch_size, clip_len, 3, 300, 300) 
	        
	        #### initialize continuous 3 images 
	        if cursor < 1: 
	            v_prev_file = dataset_img_files_v[cursor]
	            i_prev_file = dataset_img_files_i[cursor]
	        else: 
	            v_prev_file = dataset_img_files_v[cursor-1]
	            i_prev_file = dataset_img_files_i[cursor-1]

	        v_curr_file = dataset_img_files_v[cursor]
	        i_curr_file = dataset_img_files_i[cursor]

	        if cursor == size: 
	            v_late_file = dataset_img_files_v[size-1]
	            i_late_file = dataset_img_files_i[size-1]
	        else: 
	            v_late_file = dataset_img_files_v[cursor]
	            i_late_file = dataset_img_files_i[cursor]
	        	

	        v_prev_img_path 	= os.path.join(dataset_img_path_v, v_prev_file)
	        i_prev_img_path 	= os.path.join(dataset_img_path_i, i_prev_file)
	        v_current_img_path  = os.path.join(dataset_img_path_v, v_curr_file)
	        i_current_img_path  = os.path.join(dataset_img_path_i, i_curr_file)  
	        v_late_img_path 	= os.path.join(dataset_img_path_v, v_late_file)
	        i_late_img_path 	= os.path.join(dataset_img_path_i, i_late_file)   

	        v_inputimage_prev 	 = cv2.imread(v_prev_img_path) 
	        i_inputimage_prev 	 = cv2.imread(i_prev_img_path) 
	        v_inputimage_current = cv2.imread(v_current_img_path) 
	        i_inputimage_current = cv2.imread(i_current_img_path) 
	        v_inputimage_late 	 = cv2.imread(v_late_img_path) 
	        i_inputimage_late 	 = cv2.imread(i_late_img_path) 

	        v_inputimage_prev = cv2.resize(v_inputimage_prev, (300, 300), interpolation=cv2.INTER_LINEAR)
	        i_inputimage_prev = cv2.resize(i_inputimage_prev, (300, 300), interpolation=cv2.INTER_LINEAR)
	        v_inputimage_current = cv2.resize(v_inputimage_current, (300, 300), interpolation=cv2.INTER_LINEAR)
	        i_inputimage_current = cv2.resize(i_inputimage_current, (300, 300), interpolation=cv2.INTER_LINEAR)
	        v_inputimage_late = cv2.resize(v_inputimage_late, (300, 300), interpolation=cv2.INTER_LINEAR)
	        i_inputimage_late = cv2.resize(i_inputimage_late, (300, 300), interpolation=cv2.INTER_LINEAR)


	        batch_imgClip_v[0, 0] = to_tensor(v_inputimage_prev) 
	        batch_imgClip_v[0, 1] = to_tensor(v_inputimage_current) 
	        batch_imgClip_v[0, 2] = to_tensor(v_inputimage_late) 

	        batch_imgClip_i[0, 0] = to_tensor(i_inputimage_prev) 
	        batch_imgClip_i[0, 1] = to_tensor(i_inputimage_current) 
	        batch_imgClip_i[0, 2] = to_tensor(i_inputimage_late) 

	        # pdb.set_trace() 

	        cursor += 1
	        attention_map = Generator(targetObject_v.cuda(), targetObject_i.cuda(), batch_imgClip_v.cuda(), batch_imgClip_i.cuda()) 
	        attention_map = nn.functional.interpolate(attention_map, size=[v_inputimage_prev.shape[0], v_inputimage_prev.shape[1]]) 

	        # pdb.set_trace() 
	        new_Savepath = attentionSave_path + videoName 

	        if os.path.exists(new_Savepath):  
	            print(" ")
	        else: 
	            os.mkdir(new_Savepath) 

	        pilTrans = transforms.ToPILImage()
	        pilImg = pilTrans(attention_map[0].detach().cpu()) 

	        new_path = new_Savepath + "/" + str(cursor+1) + "_attentionMap.jpg"
	        print('==>> Image saved to ', new_path)
	        pilImg.save(new_path)






