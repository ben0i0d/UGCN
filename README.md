# UO-STGCN

*这是非常不稳定的研究性质代码*

## 数据处理

1. 数据预处理：通过`data_gen/gen_joint.py`下生成`joint`模态300帧的视频，
2. 数据模态预处理：基于上一步得到`bone`基础模态，与`motion`模态数据生成
3. 数据切片化处理：通过`data_gen/gen_ntu_50f.py`对原始数据处理生成50帧的视频,或者使用`data_gen/gen_ntu_50f.py`生成64帧视频

*使用300帧的原因是，存在的最大帧数为299*

*平均帧数为84.36*

![alt text](.resources/image.png)

## 模型训练

注意：
1. `main_train.py`更名为`pretrain.py`，对于`baseline`基准，后缀为`_bl`
2. 默认启用了`checkpoint`机制,注意`checkpoint`只支持单阶段，例如预训练得到的权重只服务于预训练阶段，如果要将权重用到测试阶段，则必须使用`stgcn.pth`这样的权重
3. torchlight不再需要`pip install`，改写了代码使用文件直接链接
4. 默认将生成文件统一放入`runs`，包括测试生成的`.npy`文件等，当前我们仅考虑NTU60数据集，因此没有细化目录结构

**代码仍然在梳理中，对没有用到的代码也仍在简化**
