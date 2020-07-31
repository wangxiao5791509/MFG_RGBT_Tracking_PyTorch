from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

from torch.autograd import Variable
from generator import daGenerator
from utils import *
import pdb
import warnings
warnings.filterwarnings("ignore")
import torchvision.transforms as transforms
import random

batch_size = 5   
lr = 1e-5 

generator = daGenerator()

# generatorPath = "./directionAware_TANet_rgbt_model.pkl"
# generator_weights = torch.load(generatorPath)
# generator.load_state_dict(generator_weights)

if torch.cuda.is_available():
    generator.cuda()

criterion = nn.BCELoss()
g_optim = torch.optim.Adagrad(generator.parameters(), lr=lr)
num_epoch = 30   

def to_variable(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad)

start_time = time.time()
DIR_TO_SAVE = "./generator_output/"
if not os.path.exists(DIR_TO_SAVE):
    os.makedirs(DIR_TO_SAVE)

generator.train()


attention_path = "/home/wangxiao/targetAttention_train_dataset/"

video_files = os.listdir(attention_path) 
random.shuffle(video_files)
video_files = video_files[:300]



count = 0 



for current_epoch in range(num_epoch):
    g_cost_avg = 0
    
    for videoidx in range(len(video_files)): 
        videoName = video_files[videoidx] 

        dataset_img_path = attention_path + videoName + "/image/" 
        dataset_img_files = os.listdir(dataset_img_path)

        dataset_mask_path = attention_path + videoName + "/mask/" 
        dataset_tarObject_path = attention_path + videoName + "/tarObject/" 

        numBatches = len(dataset_img_files) / batch_size 
        cursor = 0

        # pdb.set_trace() 
        for idx in range(int(numBatches)):

            size = len(dataset_img_files) 

            if cursor + batch_size > size:
                cursor = 0
                # np.random.shuffle(dataset_img_files)
                np.sort(dataset_img_files)
            
            batch_img = torch.zeros(batch_size, 3, 300, 300)
            batch_map = torch.zeros(batch_size, 1, 300, 300)
            targetObject_img = torch.zeros(batch_size, 3, 300, 300)
            targetObject_gray = torch.zeros(batch_size, 3, 300, 300)

            clip_len = 3 
            batch_imgClip = torch.zeros(batch_size, clip_len, 3, 300, 300) 
            batch_grayClip = torch.zeros(batch_size, clip_len, 3, 300, 300) 
                     
            to_tensor = transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0. 

            for batchidx in range(batch_size):

                #### initialize continuous 3 images 
                if cursor < 1: 
                    prev_file = dataset_img_files[cursor]
                else: 
                    prev_file = dataset_img_files[cursor-1]

                curr_file = dataset_img_files[cursor]

                if cursor == size: 
                    late_file = dataset_img_files[size-1]
                else: 
                    late_file = dataset_img_files[cursor]
                
                imgIndex = curr_file[-12:]

                prev_imgIndex = prev_file[-12:]
                late_imgIndex = late_file[-12:]
                # print(videoName, " ", imgIndex) 

                targetObject_img_path = os.path.join(dataset_tarObject_path, videoName + '_target-00000001.jpg')
                full_img_path = os.path.join(dataset_img_path, videoName + "_image-" + imgIndex)

                prev_full_img_path = os.path.join(dataset_img_path, videoName + "_image-" + prev_imgIndex) 
                late_full_img_path = os.path.join(dataset_img_path, videoName + "_image-" + late_imgIndex) 

                full_map_path = os.path.join(dataset_mask_path, videoName + "_mask-" + imgIndex)
                cursor += 1

                inputimage = cv2.imread(full_img_path) 
                prev_inputimage = cv2.imread(prev_full_img_path) 
                late_inputimage = cv2.imread(late_full_img_path) 


                #### for the gray image: 
                gray_prev_inputimage = cv2.cvtColor(prev_inputimage, cv2.COLOR_BGR2GRAY) 
                gray_prev_inputimage = to_tensor(gray_prev_inputimage)
                gray_prev_inputimage = torch.stack([gray_prev_inputimage, gray_prev_inputimage, gray_prev_inputimage], 1)[0] 
                batch_grayClip[batchidx, 0] = gray_prev_inputimage 

                gray_inputimage = cv2.cvtColor(inputimage, cv2.COLOR_BGR2GRAY) 
                gray_inputimage = to_tensor(gray_inputimage)
                gray_inputimage = torch.stack([gray_inputimage, gray_inputimage, gray_inputimage], 1)[0] 
                batch_grayClip[batchidx, 1] = gray_inputimage 

                gray_late_inputimage = cv2.cvtColor(prev_inputimage, cv2.COLOR_BGR2GRAY) 
                gray_late_inputimage = to_tensor(gray_late_inputimage)
                gray_late_inputimage = torch.stack([gray_late_inputimage, gray_late_inputimage, gray_late_inputimage], 1)[0] 
                batch_grayClip[batchidx, 2] = gray_late_inputimage 


                # pdb.set_trace() 
                batch_img[batchidx] = to_tensor(inputimage)
                batch_imgClip[batchidx, 0] = to_tensor(prev_inputimage) 
                batch_imgClip[batchidx, 1] = to_tensor(inputimage) 
                batch_imgClip[batchidx, 2] = to_tensor(late_inputimage) 
                
                targetObjectimage = cv2.imread(targetObject_img_path)
                targetObject_img[batchidx] = to_tensor(targetObjectimage)
                
                gray_targetObjectimage = cv2.cvtColor(targetObjectimage, cv2.COLOR_BGR2GRAY) 
                gray_targetObjectimage = to_tensor(gray_targetObjectimage)
                gray_targetObjectimage = torch.stack([gray_targetObjectimage, gray_targetObjectimage, gray_targetObjectimage], 1)[0] 
                targetObject_gray[batchidx] = gray_targetObjectimage 


                saliencyimage = cv2.imread(full_map_path, 0)
                saliencyimage = np.expand_dims(saliencyimage, axis=2)
                batch_map[batchidx] = to_tensor(saliencyimage)



            batch_img = to_variable(batch_img, requires_grad=True)
            batch_map = to_variable(batch_map, requires_grad=False)
            targetObject_img = to_variable(targetObject_img, requires_grad=True)
            targetObject_gray = to_variable(targetObject_gray, requires_grad=True) 
            batch_imgClip = to_variable(batch_imgClip, requires_grad=True) 
            batch_grayClip = to_variable(batch_grayClip, requires_grad=True) 

            val_batchImg = batch_img
            val_targetObjectImg = targetObject_img 
            val_gray_targetObjectimage = targetObject_gray 
            val_imgClip = batch_imgClip  
            val_batch_grayClip = batch_grayClip 

            count = count + 1

            g_optim.zero_grad()
            attention_map = generator(targetObject_img, targetObject_gray, batch_imgClip, batch_grayClip)

            batch_map = nn.functional.interpolate(batch_map, size=[attention_map.shape[2], attention_map.shape[3]]) 


            # pdb.set_trace()
            g_gen_loss = criterion(attention_map, batch_map)
            g_loss = torch.sum(g_gen_loss)
            g_cost_avg += g_loss.item()
            g_loss.backward()
            g_optim.step()


            print("==>> Epoch [%d/%d], g_gen_loss: %.4f, vidIndex [%d/%d], LR: %.6f, time: %4.4f" % \
                (current_epoch, num_epoch, g_loss.item(), videoidx, len(video_files), lr, time.time()-start_time))


        # validation 
        out = generator(val_targetObjectImg, val_gray_targetObjectimage, val_imgClip, val_batch_grayClip)
        map_out = out.cpu().data.squeeze(0)
        for iiidex in range(batch_size): 
           new_path = DIR_TO_SAVE + str(current_epoch) + str(iiidex) + ".jpg"
           pilTrans = transforms.ToPILImage()
           pilImg = pilTrans(map_out[iiidex]) 
           # print('==>> Image saved to ', new_path)
           pilImg.save(new_path)


        g_cost_avg /= numBatches

    # pdb.set_trace()
    # Save weights 
    if current_epoch % 1 == 0:
        print("==>> save checkpoints ... ", ' ==>> Train_loss->', (g_cost_avg))
        torch.save(generator.state_dict(), '20200525_directionAware_TANet_rgbt_model.pkl')





