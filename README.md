# MFGNet_RGBT_Tracking_PyTorch
Official implementation of **MFGNet: Dynamic Modality-Aware Filter Generation for RGB-T Tracking**, Xiao Wang, Xiujun Shu, Shiliang Zhang, Bo Jiang, Yaowei Wang, Yonghong Tian, Feng Wu, Accepted by IEEE Transactions on Multimedia (TMM), 2022 [[Paper](https://arxiv.org/abs/2107.10433)] 


## Abstract: 
Many RGB-T trackers attempt to attain robust feature representation by utilizing an adaptive weighting scheme (or attention mechanism). Different from these works, we propose a new dynamic modality-aware filter generation module (named MFGNet) to boost the message communication between visible and thermal data by adaptively adjusting the convolutional kernels for various input images in practical tracking. Given the image pairs as input, we first encode their features with the backbone network. Then, we concatenate these feature maps and generate dynamic modality-aware filters with two independent networks. The visible and thermal filters will be used to conduct a dynamic convolutional operation on their corresponding input feature maps respectively. Inspired by residual connection, both the generated visible and thermal feature maps will be summarized with input feature maps. The augmented feature maps will be fed into the RoI align module to generate instance-level features for subsequent classification. To address issues caused by heavy occlusion, fast motion and out-of-view, we propose to conduct a joint local and global search by exploiting a new direction-aware target driven attention mechanism. The spatial and temporal recurrent neural network is used to capture the direction-aware context for accurate global attention prediction. Extensive experiments on three large-scale RGB-T tracking benchmark datasets validated the effectiveness of our proposed algorithm.



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
~~~
CUDA_HOME=/usr/local/cuda-10.1 python setup.py build_ext --inplace 
~~~


## Training and Testing: 
1. generate the "50.pkl" with prepro_rgbt.py as the training data; 

2. train the tracker with train.py; 

3. train the rgbt_TANet with train_rgbtTANet.py; 

4. Obtain the attention maps and run the test.py for rgbt-tracking. 


## Pretrained Models (CBAM_dfg_rtmdnet_trained_on_50.pth, 20200525_directionAware_TANet_rgbt_model.pkl):  
```
链接：https://pan.baidu.com/s/1Je7KB6x37Mc7ay4dCxDJvQ  提取码：AHUT 
``` 


## Results: 

![rgbt_kite4](https://github.com/wangxiao5791509/DFG_RGBT_Tracking_PyTorch/blob/master/results_on_rgbt210_234.png)

![rgbt_kite4](https://github.com/wangxiao5791509/DFG_RGBT_Tracking_PyTorch/blob/master/ComponentAnalysis.png)



## Acknowledgement: 
* https://github.com/BossBobxuan/RT-MDNet 
* https://github.com/NieXC/pytorch-mula 
* https://github.com/luuuyi/CBAM.PyTorch 
* [Survey] "**Dynamic neural networks: A survey.**" Han, Yizeng, et al.  IEEE Transactions on Pattern Analysis and Machine Intelligence (2021). [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9560049)]



## Citation: 
If you use this code for your research, please cite the following paper: 
~~~
@article{wang2021mfgnet,
  title={MFGNet: Dynamic Modality-Aware Filter Generation for RGB-T Tracking},
  author={Wang, Xiao and Shu, Xiujun and Zhang, Shiliang and Jiang, Bo and Wang, Yaowei and Tian, Yonghong and Wu, Feng},
  journal={IEEE Transactions on Multimedia},
  year={2022}
}
~~~

If you have any questions, feel free to contact me via email: **wangxiaocvpr@foxmail.com**. 




