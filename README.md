# DFG_RGBT_Tracking_PyTorch
Official Implementation of DFG-RGBT-Tracker with PyTorch 

[[Project]()]   [[Paper]()]  

## Install: 
This code is developed based on Python 3.7, PyTorch 1.0, Ubuntu 16.04, Tesla P100 * 4. Install anything it warnings. 

RoI align module needs to compile first: 

python setup.py build_ext --inplace  

CUDA_HOME=/usr/local/cuda-10.1 python setup.py build_ext --inplace 


## Acknowledgement: 
1. https://github.com/BossBobxuan/RT-MDNet 
2. https://github.com/Ugness/PiCANet-Implementation 
3. https://github.com/NieXC/pytorch-mula 
4. https://github.com/luuuyi/CBAM.PyTorch 


## Citation: 
If you use this code for your research, please cite the following paper: 

