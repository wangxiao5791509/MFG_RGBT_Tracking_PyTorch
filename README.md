# MFGNet_RGBT_Tracking_PyTorch
Official Implementation of MFGNet-RGBT-Tracker ("Dynamic Modality-Aware Filter Generation for RGB-T Tracking") with PyTorch 


Many RGB-T trackers attempt to attain robust feature representation by utilizing an adaptive weighting scheme (or attention mechanism). Different from these works, we propose a new dynamic modality-aware filter generation module (named MFGNet) to boost the message communication between visible and thermal data by adaptively adjusting the convolutional kernels for various input images in practical tracking. Our experimental results demonstrate the advantages of our proposed MFGNet for RGB-T tracking. 


[[Project](https://sites.google.com/view/dfgtrack/)]   [[Paper]()] 


![rgbt_car10](https://github.com/wangxiao5791509/DFG_RGBT_Tracking_PyTorch/blob/master/pipelinev5.png) 



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



## Results: 


![rgbt_kite4](https://github.com/wangxiao5791509/DFG_RGBT_Tracking_PyTorch/blob/master/results_on_rgbt210_234.png)

![rgbt_kite4](https://github.com/wangxiao5791509/DFG_RGBT_Tracking_PyTorch/blob/master/ComponentAnalysis.png)



## Acknowledgement: 
* https://github.com/BossBobxuan/RT-MDNet 
* https://github.com/NieXC/pytorch-mula 
* https://github.com/luuuyi/CBAM.PyTorch 




## Citation: 
If you use this code for your research, please cite the following paper: 
~~~
@article{wang2020dfgrgbttrack,
  title={Dynamic Modality-Aware Filter Generation for RGB-T Tracking},
  author={Xiao Wang, Xiujun Shu, Shiliang Zhang, Bo Jiang, Yaowei Wang, Yonghong Tian, Feng Wu},
  journal={arXiv preprint},
  year={2020}
}
~~~
