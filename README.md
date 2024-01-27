# CV FINAL PROJECT  
* This is a class project of the Computer Vision course in Peking University.
### Gongle Xue, Yifeng Xu, Yang Li  
### Theme:Deep Image Prior
#### Introduction
This repository contains our implementation of the paper "Deep Image Prior" and some improvements inspired by the paper "On Measuring and Controlling the Spectral Bias of the Deep Image Prior". 

We propose to solve image reconstruction tasks including denoising, inpainting and super-resolution via using the structure of network as image prior.

#### Get Started
##### Prepare your python environment
We recommend you to use anaconda3.
```bash
conda create -n dip python=3.10.13
conda activate dip
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

#### Prepare your dataset
Download our dataset in the repository directory from [](). Our dataset consists of the testing images in the paper "Deep Image Prior" and our own images(the names ofen begin with "our_imgs"). 

If you want to apply our method to other images, just add their address to the function "name2add()" in [utils.py](utils.py). For denoising tasks, the "mask_address" is None; for inpainting tasks, the "mask_address" is the address of image mask; for super-resolution tasks, the "mask_address" is the address of original high-resolution image(if exits).

#### Evalutation
Run the commands in [record.md](record.md) to see the results with best settings we've ever tried. Remember to check if the file paths of shells are right.

### Reference:
[Deep Image Prior](https://arxiv.org/pdf/1711.10925.pdf) 
[On Measuring and Controlling the Spectral Bias of the Deep Image Prior](https://arxiv.org/abs/2107.01125)
[Code](https://github.com/DmitryUlyanov/deep-image-prior) 
[Untrained Neural Network Priors for Inverse Imaging Problems: A Survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9878048)  
[100-lines-code](https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code)  