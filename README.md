# (RFF)

## Environments

The code has been tested in the following environments:

- Python 3.8
- PyTorch 1.8.1
- cuda 10.2
- torchsummary, torchvision, thop, scipy, sympy

## Pre-trained Models

**CIFAR-10:**

[Vgg-16] | [ResNet56] | [GoogLeNet]

**ImageNet:**

[ResNet50]

## Running Code

The experiment settings are as follows:

**1. VGG-16-bn**

| Compression Rate            | Flops($\downarrow $) | Params($\downarrow $) | Accuracy |
| --------------------------- | -------------------- | --------------------- | -------- |
| [0.25]\*5+[0.35]\*3+[0.8]*5 | 115.8M(63.2%)        | 2.11M(85.94%)         | 93.72%   |

```shell
#VGGNet
#All scripts can be cut-copy-paste from run.bat or run.sh.
python main.py \
--arch vgg_16_bn \
--resume [pre-trained model dir] \
--compress_rate [0.25]*5+[0.35]*3+[0.8]*5 \
--num_workers 1 \
--batch_size 128 \
--epochs 1 \
--lr 0.001 \
--lr_decay_step 1 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 

python main.py \
--arch vgg_16_bn \
--from_scratch True \
--resume final_pruned_model/vgg_16_bn_1.pt \
--num_workers 1 \
--epochs 200 \
--gpu 0 \
--lr 0.01 \
--lr_decay_step 100,150 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 
```

**2. ResNet-56**

| Compression Rate                                             | Flops($\downarrow $) | Params($\downarrow $) | Accuracy |
| ------------------------------------------------------------ | -------------------- | --------------------- | -------- |
| [0.]+[0.2,0.]\*1+[0.65,0.]\*8+[0.2,0.15]\*1+[0.65,0.15]\*8+[0.2,0.]\*1+[0.4,0.]\*8 | 60.0M(52.6%)         | 0.48M(43.4%)          | 93.66%   |

```shell
#ResNet-56
python main.py \
--arch resnet_56 \
--resume [pre-trained model dir] \
--compress_rate [0.]+[0.2,0.]*1+[0.65,0.]*8+[0.2,0.15]*1+[0.65,0.15]*8+[0.2,0.]*1+[0.4,0.]*8 \
--num_workers 1 \
--epochs 1 \
--lr 0.001 \
--lr_decay_step 1 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 

python main.py \
--arch resnet_56 \
--from_scratch True \
--resume final_pruned_model/resnet_56_1.pt \
--num_workers 1 \
--epochs 300 \
--gpu 0 \
--lr 0.01 \
--lr_decay_step 150,225 \
--weight_decay 0.0005 \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 
```

**3. GoogLeNet**

| Compression Rate                       | Flops($\downarrow $) | Params($\downarrow $) | Accuracy |
| -------------------------------------- | -------------------- | --------------------- | -------- |
| [0.2]+[0.75]\*15+[0.75]\*9+[0.,0.4,0.] | 0.543B(64.5%)        | 2.77M(55.1%)          | 94.75%   |

```shell
#GoogLeNet
python main.py \
--arch googlenet \
--resume [pre-trained model dir] \
--compress_rate [0.2]+[0.7]*15+[0.75]*9+[0.,0.4,0.] \
--num_workers 1 \
--epochs 1 \
--lr 0.001 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 

python main.py \
--arch googlenet \
--from_scratch True \
--resume final_pruned_model/googlenet_1.pt \
--num_workers 1 \
--epochs 200 \
--lr 0.01 \
--lr_decay_step 100,150 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--save_id 1 
```

**4. ResNet-50**

| Compression Rate                                             | Flops($\downarrow $) | Params($\downarrow $) | Top-1 Acc. | Top-5 Acc. |
| ------------------------------------------------------------ | -------------------- | --------------------- | ---------- | ---------- |
| [0.]+[0.2,0.2,0.2]\*1+[0.65,0.65,0.2]\*2+[0.2,0.2,0.15]\*1+[0.65,0.65,0.15]\*3+[0.2,0.2,0.1]\*1+[0.65,0.65,0.1]\*5+[0.2,0.2,0.1]+[0.2,0.2,0.1]\*2 | 1.83B(55.5%)         | 15.1M(41.1%)          | 75.43%     | 92.49%     |

```shell
#ResNet-50
python main.py \
--arch resnet_50 \
--resume [pre-trained model dir] \
--data_dir [dataset dir] \
--dataset ImageNet \
--compress_rate [0.]+[0.2,0.2,0.2]*1+[0.65,0.65,0.2]*2+[0.2,0.2,0.15]*1+[0.65,0.65,0.15]*3+[0.2,0.2,0.1]*1+[0.65,0.65,0.1]*5+[0.2,0.2,0.1]+[0.2,0.2,0.1]*2 \
--num_workers 4 \
--batch_size 128 \
--epochs 1 \
--lr_decay_step 1 \
--lr 0.001 \
--weight_decay 0. \
--input_size 224 \
--save_id 1 

python main.py \
--arch resnet_50 \
--from_scratch True \
--resume final_pruned_model/resnet_50_1.pt \
--num_workers 4 \
--epochs 120 \
--lr 0.01 \
--lr_decay_step 30,60,90 \
--batch_size 128 \
--weight_decay 0.0001 \
--input_size 224 \
--data_dir [dataset dir] \
--dataset ImageNet \
--save_id 1
```

