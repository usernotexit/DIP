# Pix2pix 图像变换

本次实验的模型基于全卷积网络（[Fully Convolutional Layers](https://arxiv.org/abs/1411.4038)）的跳跃连接思想，通过卷积层和最大池化层提取图像特征，在经过浅层MLP进行分类后，由反卷积的方式恢复特征图尺寸。

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

1. 采用 [Facades](https://cmp.felk.cvut.cz/~tylecr1/facade/) 作为数据集
2. 本次作业亦尝试渲染图像 [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) 作为数据集，但由于数据量大，图像尺寸大，分类数目且不均衡等原因导致训练时间过长且效果较差，因此放弃使用

## Training

To Be Done：代码整理中

```train
```

>📋 各参数含义
> 1. d 
> 2. d

## Evaluation & Pre-trained Models

对于训练好的模型，打开交互界面

```eval
python run_Pix2pix_gradio.py --model-file model/final_model.pth
```

>📋 各参数含义

## Results

训练 200 epoch 后，在验证集上的损失如下：（待完善）

| Model name         | Train Loss  | Val Loss |
| ------------------ |---------------- | -------------- |
| FCN-32s   |     。。。         |      。。。       |
| FCN-16s   |     。。。         |      。。。       |
| FCN-8s    |     。。。         |      。。。       |


>📋 
