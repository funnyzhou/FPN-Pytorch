# FPN-Pytorch

This repository borrowed heavily from [roytseng-tw/Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch). The main difference is that we mainly focus on Faster R-CNN instead of Mask R-CNN. We add more modules on top of Feature Pyramid Network (FPN) which recently became a common backbone in object detection. These modules include:

## Updates
- [x] Focal Loss (both biased layer initialization and loss function)
- [x] Deformable Convolution
- [x] Cosine Annealing Learning Rate
- [x] Cascade R-CNN (code finished, under test)
- [x] SNIP (code finished, under test)
- [ ] PANet

## Getting Started
Please follow the instruction to configure data preparation (including downloading pre-trained models) and code compilation.

## Train and Validation
The operations are the same as those in [roytseng-tw/Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch).
For training a Res50-FPN using 2x schedule,
```python
python tools/train_net_step.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_2x.yaml --bs 32
```
The above script performs the default action on all gpus. The total batch size is 32.

If you want to continue your training on a specific checkpoint, just add an option
```python
--load_ckpt {path/to/the/checkpoint}
```
after `tools/train_net_step.py`.

Besides, we also provide a `Windows` version. By simply replacing `tools/train_net_step.py` with `tools/train_net_step_win.py`, you can run the code on Win. We remove the resource package in `train_net_step_win.py` which at least runs correctly on one gpu in windows (tested).

For testing your trained models,
```python
python tools/test_net.py --dataset coco2017 --cfg config/baselines/e2e_faster_rcnn_R-50-FPN_2x.yaml --load_ckpt {path/to/your/checkpoint}
```
If you'd like to use the multi-gpu testing mode in [roytseng-tw/Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch), add 
```python
--multi-gpu-testing
```
behind the above script.

## Focal Loss
We implement the Focal Loss in Fast R-CNN (we also tried to put it in RPN, but it did not work). The related hyperparameters are as follows (item in bracket tells the DEFAULT value):

- `FAST_RCNN.FOCAL_LOSS (False)`: use Focal Loss in Fast R-CNN
- `FAST_RCNN.FL_INIT (False)`: use biased layer initialization proposed in [1]
- `FAST_RCNN.PRIOR (0.01)`: the initialization prior in Fast R-CNN
- `RPN.FL_INIT (False)`: use Focal Loss in RPN
- `RPN.PRIOR (0.01)`: the initialization prior in RPN
- `FAST_RCNN.GAMMA (2)` AND `FAST_RCNN.ALPHA (0.25)`: two hyparams in Focal Loss

## Deformable Convolution
We implement the Deformable Convolution presented in [2]. Note that we ignored the deformable pooling and put the deformable module into the last layers in block 3 and 4 (the same as mxnet implementation).

To use deform conv, you need to set `MODEL.USE_DEFORM (False)` to `True`. Note that deform operation usually take more time and consume more memory, so you may need a smaller batch size.

## Cosine Annealing LR
Cosine annealing lr has been proved to be effective in image classification tasks. The formula is very simple:
$$\eta_t=\eta_{min}^i+\frac{1}{2}(\eta_{max}^i-\eta_{min}^i)(1+\cos(\frac{T_{cur}}{T_i}\pi))$$
Since we only set the maximum iteration in `config` file, you need to set several hyperparameters:

- `SOLVER.COSINE_LR (False)`: whether to use cosine annealing after warm up stage
- `SOLVER.COSINE_NUM_EPOCH (13)`: the number epochs (corresponds to max_iter, like 12 or 13, etc.) you want to run. This is required to calculate $T_i$
- `SOLVER.COSINE_T0 (3)`: required to calculate $T_i$
- `SOLVER.COSINE_MULTI (2)`: required to get $T_i$

For more details, please refer to `train_net_step.py` or `train_net_step_win.py`

## Cascade R-CNN
We implemented Cascade R-CNN based on [zhaoweicai/Detectron-Cascade-RCNN](https://github.com/zhaoweicai/Detectron-Cascade-RCNN) which is written in caffe2. Since this official repository contains more details than those stated in the original paper, we may need more time to tune the model parameters. Please keep patience.

The corresponding options in `config.py` are:

- `FAST_RCNN.USE_CASCADE (False)`:  you should set it to `True` if you want to use Cascade R-CNN
- `CASCADE_RCNN.{}`: Hyperparameters in Cascade R-CNN. Please keep it original if you don't have an idea about how to set them

## SNIP: An Analysis of Scale Invariance in Object Detection
SNIP is presented as a multi-scale training method for modern object detectors. Its main idea is to assign object with different size to different models when training networks, as for size out of scope, just ignore them. 

Here, we offer a simplified version: you can train a detector by specifying the scope of size (both in RPN and Fast R-CNN), and gradients of objects out of this scope are ignored during the training stage. The options in `config.py` are:

- `{FAST_RCNN, RPN}.SNIP (False)`: whether to use SNIP in RPN and Fast R-CNN
- `{FAST_RCNN, RPN}.RES_LO (0)`: lower bound of object size after re-scaling (e.g., object size after you resize the input image to 800$\times$800)
- `{FAST_RCNN, RPN}.RES_HI (0)`: upper bound of object size
- `{FAST_RCNN,RPN}.SNIP_NEG_THRESH (0.3)`:  anchors which have an overlap greater than 0.3 with an invalid (out of size scope) ground truth box are excluded during training
- `{FAST_RCNN,RPN}.SNIP_TARGET_THRESH (0.5)`： this operation helps to ignore anchors above. Please let it be the default (perhaps, we will change it in the future)

Now you are able to train different detectors using different size scopes. Good luck!

## Benchmark

To be continued...

## Reference
[1][Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002), CVPR 2017.
[2][Deformable Convolutional Networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.pdf), ICCV 2017.
[3][SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983), ICLR 2017.
[4][Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/abs/1712.00726), CVPR 2018.
[5][An Analysis of Scale Invariance in Object Detection ­ SNIP](http://openaccess.thecvf.com/content_cvpr_2018/papers/Singh_An_Analysis_of_CVPR_2018_paper.pdf), CVPR 2018.