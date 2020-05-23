# DFG_RGBT_Tracking_PyTorch
Official Implementation of DFG-RGBT-Tracker ("Dynamic Modality-Aware Filter Generation for RGB-T Tracking") with PyTorch 


[[Project](https://sites.google.com/view/dfgtrack/)]   [[Paper]() (Coming Soon)] 


## Demo:
(Red: Ours, Blue: Ground Truth, Green: RT-MDNet)  

![rgbt_car10](https://github.com/wangxiao5791509/DFG_RGBT_Tracking_PyTorch/blob/master/rgbt_car10.gif) 

![rgbt_balancebike](https://github.com/wangxiao5791509/DFG_RGBT_Tracking_PyTorch/blob/master/rgbt_balancebike.gif) 

![rgbt_flower1](https://github.com/wangxiao5791509/DFG_RGBT_Tracking_PyTorch/blob/master/rgbt_flower1.gif)

![rgbt_kite4](https://github.com/wangxiao5791509/DFG_RGBT_Tracking_PyTorch/blob/master/rgbt_kite4.gif)


## Install: 
This code is developed based on Python 3.7, PyTorch 1.0, CUDA 10.1, Ubuntu 16.04, Tesla P100 * 4. Install anything it warnings. 

RoI align module needs to compile first: 

CUDA_HOME=/usr/local/cuda-10.1 python setup.py build_ext --inplace 



## Train and Test: 
1. generate the "50.pkl" with prepro_rgbt.py as the training data; 

2. train the tracker with train.py; 

3. train the rgbt_TANet with train_rgbtTANet.py; 

4. Obtain the attention maps and run the test.py for rgbt-tracking. 




## Acknowledgement: 
* https://github.com/BossBobxuan/RT-MDNet 
* https://github.com/NieXC/pytorch-mula 
* https://github.com/luuuyi/CBAM.PyTorch 




## Citation: 
If you use this code for your research, please cite the following paper: 

