# DFG_RGBT_Tracking_PyTorch
Official Implementation of DFG-RGBT-Tracker with PyTorch 

[[Project]()]   [[Paper]()]  

## Install: 
This code is developed based on Python 3.7, PyTorch 1.0, CUDA 10.1, Ubuntu 16.04, Tesla P100 * 4. Install anything it warnings. 

RoI align module needs to compile first: 

CUDA_HOME=/usr/local/cuda-10.1 python setup.py build_ext --inplace 


## Train and Test: 
Pre-trained backbone network model (imagenet-vgg-m.mat): [[Google-Drive](https://drive.google.com/open?id=1em1Aj4AyJGlT18Xw3bggBZSwQwbk_Gli)]

Our trained Models (test_CBAM_dfg_rtmdnet_trained_on_50.pth): [[Google-Drive](https://drive.google.com/open?id=15xdxh47mnCINyKAWJP-vLjgij1PtyJtI)]

TANet (TANet_rgbt_model.pkl): [[Google-Drive](https://drive.google.com/open?id=1c1XjacXoFWeKpnEJn480LoRWBS2E8nfx)] 


1. generate the "50.pkl" with prepro_rgbt.py as the training data; 

2. train the tracker with train.py; 

3. train the rgbt_TANet with train_rgbtTANet.py; 

4. Obtain the attention maps and run the test.py for rgbt-tracking. 




## Acknowledgement: 
1. https://github.com/BossBobxuan/RT-MDNet 
2. https://github.com/Ugness/PiCANet-Implementation 
3. https://github.com/NieXC/pytorch-mula 
4. https://github.com/luuuyi/CBAM.PyTorch 


## Citation: 
If you use this code for your research, please cite the following paper: 

