import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
from os.path import join, isdir
from tracker import *
import numpy as np
import argparse
import pickle
import math
import pdb 
import torchvision.transforms as transforms
import random
import warnings
warnings.filterwarnings("ignore") 



def genConfig(seq_path, set_type):

    path, seqname = os.path.split(seq_path)

    if set_type == 'OTB100':
        img_list = sorted([seq_path + '/img/' + p for p in os.listdir(seq_path + '/img') if os.path.splitext(p)[1] == '.png'])
        gt = np.loadtxt(seq_path + '/groundtruth_rect.txt', delimiter=',')

    #####################################################################
    #####               For the RGBT dataset 
    ##################################################################### 
    elif set_type == 'dataset234': 
        img_list_v = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        img_list_i = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] == '.jpg'])
        gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')

    elif set_type == 'dataset210': 
        img_list_v = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        img_list_i = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] == '.jpg'])
        gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')


    return img_list_v, img_list_i, gt 




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default = 'dataset234')
    parser.add_argument("-model_path", default = './models/test_CBAM_dfg_rtmdnet_trained_on_50.pth')
    parser.add_argument("-result_path", default = './result.npy')
    parser.add_argument("-visual_log",default=False, action= 'store_true')
    parser.add_argument("-visualize",default=False, action='store_true')
    parser.add_argument("-adaptive_align",default=True, action='store_false')
    parser.add_argument("-padding",default=1.2, type = float)
    parser.add_argument("-jitter",default=True, action='store_false')
    
    args = parser.parse_args()

    ##################################################################################
    #########################Just modify opts in this script.#########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    ## option setting
    opts['model_path']=args.model_path
    opts['result_path']=args.result_path
    opts['visual_log']=args.visual_log
    opts['set_type']=args.set_type
    opts['visualize'] = args.visualize
    opts['adaptive_align'] = args.adaptive_align
    opts['padding'] = args.padding
    opts['jitter'] = args.jitter
    ##################################################################################
    ############################Do not modify opts anymore.###########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    print(opts)


    ## path initialization
    dataset_path = '/wangxiao/experiments/'
    result_home = '/wangxiao/experiments/trackingResults/'

    seq_home = dataset_path + opts['set_type']
    seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
    seq_list = np.sort(seq_list) 

    iou_list=[]
    fps_list=dict()
    bb_result = dict()
    result = dict()

    iou_list_nobb=[]
    bb_result_nobb = dict()
    for num, seq in enumerate(seq_list):

        if num<-1:
            continue

        already_done = os.listdir(result_home) 
        
        if seq+"_rgbt234.txt" in already_done: 
            print("==>> Skip this video: ", seq) 
        else: 
            txtName = seq + '_rgbt234.txt'
            fid = open(result_home + txtName, 'w')
            
            seq_path = seq_home + '/' + seq
            img_list_v, img_list_i, gt = genConfig(seq_path, opts['set_type'])

            iou_result, result_bb, fps, result_nobb = run_mdnet(img_list_v, img_list_i, gt[0], gt, seq = seq, display=opts['visualize'])

            enable_frameNum = 0.
            for iidx in range(len(iou_result)):
                if (math.isnan(iou_result[iidx])==False): 
                    enable_frameNum += 1.
                else:
                    ## gt is not alowed
                    iou_result[iidx] = 0.

            iou_list.append(iou_result.sum()/enable_frameNum)
            bb_result[seq] = result_bb
            fps_list[seq]=fps

            bb_result_nobb[seq] = result_nobb
            print('{} {} : {} , total mIoU:{}, fps:{}'.format(num,seq,iou_result.mean(), sum(iou_list)/len(iou_list),sum(fps_list.values())/len(fps_list)))


            for iidex in range(len(result_bb)):
               line = result_bb[iidex]

               # pdb.set_trace() 
               fid.write(str(line[0]))
               fid.write(',')
               fid.write(str(line[1]))
               fid.write(',')
               fid.write(str(line[2]))
               fid.write(',')
               fid.write(str(line[3]))
               fid.write('\n')
            fid.close()


