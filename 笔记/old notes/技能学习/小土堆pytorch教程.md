---

---
---
# 重要函数`dir()`与`help()`
dir(torch)函数可以查看指定方法包下的所有方法
```python
dir(torch)
...
#就会列出torch中的各个子模块
```
而help()可以查询官方注释

# 如何加载数据？
## Dataset
	提供一种方式获取数据以及label，读取一个数据集合
### 本地数据集获取
代码位于[这里](D:\Doc\Python Practise\pytorch\read_data.py)
```python
from torch.utils.data import Dataset
from PIL import Image
import os

#使用必须继承Dataset类
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir     
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
        
    def __len__(self):
        return len(self.img_path)

root_dir = "hymenoptera_data\hymenoptera_data\\train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
train_dataset = ants_dataset + bees_dataset

print(len(train_dataset))

image, label = bees_dataset[1]

image.show()
```
### torchvision数据获得
```python
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(
    root="./dataset", train=True, transform=dataset_transform, download=True
)
test_set = torchvision.datasets.CIFAR10(
    root="./dataset", train=False, transform=dataset_transform, download=True
)

writer = SummaryWriter("P10")

for i in range(10):
    img,target=test_set[i]
    writer.add_image("test_set",img,i)
writer.close()
```
## Dataloader
	为后续神经网络提供不同数据形式

# transforms
![[Pasted image 20240523135211.png]]

## 常见的transforms方法

```python
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
  
writer = SummaryWriter("logs")
img=Image.open("hymenoptera_data/hymenoptera_data/train/ants/5650366_e22b7e1065.jpg")
print(img)

trans_totensor = transforms.ToTensor()      #获得转换工具
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1, 1, 1], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 2)

# resize
print(img.size)
trans_resize = transforms.Resize((100, 100))
# img PIL->resize->img_resize PIL
img_resize = trans_resize(img)
print(type(img_resize))
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize)

# Compose -resize -2

trans_resize2 = transforms.Resize(100)
trans_compose = transforms.Compose([trans_resize2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2,1)
  
#RandomCrop
tran_random = transforms.RandomCrop(100)
trans_compose_2 = transforms.Compose([tran_random,trans_totensor])
for i in range(10):
    img_crop=trans_compose_2(img)

writer.close()
```

# nn卷积操作
使用[**torch.nn**](https://pytorch.org/docs/stable/nn.html)中封装好的函数进行操作，**类似**矩阵相乘
## Conv2d
较为常用的参数解释如[链接](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)动画所演示，这里着重说一下**out_channels**参数
![[Pasted image 20240602192425.png]]当out_channels=2时，将使用两个卷积核对同一输入操作，并输出结果。
**注意！**：卷积核的内容是随机生成的，每一次运行都会改变


# 池化操作

## Maxpool2d
>保留数据特征并减少数据量，[MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d)]

![[Pasted image 20240603090253.png]]
输入时为5*5图像，输出后可以大大降低图像数据量。有点类似数据压缩。具体效果如下图所示
![[Pasted image 20240603100408.png]]

# 非线性激活[Non-linear Activations (weighted sum, nonlinearity)](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
>对图像矩阵执行一系列非线性操作

## nn.ReLu![[Pasted image 20240604093754.png]]

# Flatten
将25个像素（图左）以线性方式转换至目标像素个数（图右）
![[Pasted image 20240604145230.png]]

# Loss Functions（损失函数）
>其两大作用为
>1. 计算实际输出与目标之间的差距
>2. 为我们更新输出提供依据

![[Pasted image 20240613193459.png]]
在实际的应用中，我们希望预测结果越来越准确时就需要损失函数
![[Pasted image 20240615141346.png]]
`output`的三个数据分别指代Person，dog，cat三者的相似概率，此时通过下图算法
![[Pasted image 20240615141401.png]]
可以得到各个类别的损失函数值，之后通过此数值调整网络（使用优化器）

# 优化器[torch.optim](https://pytorch.org/docs/stable/optim.html)
优化器需要配合损失函数使用，首先需要计算损失
```python
  
loss = torch.nn.CrossEntropyLoss()
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
		result_loss = loss(outputs, targets)    
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        # print(result_loss)
        running_loss = running_loss + result_loss
    print(running_loss)
```

# 如何使用现有网络
```python
import torchvision
from torch import nn
  
#使用现有神经网络
vgg16_false = torchvision.models.vgg16()                    #不使用预处理
vgg16_true = torchvision.models.vgg16(weights="DEFAULT")    #使用预处理

#CRFAR10数据集
train_data = torchvision.datasets.CIFAR10(
    "./data", True, torchvision.transforms.ToTensor(), download=True
)

#修改现有网络的两个方法

#1.添加处理
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)

#2.替换处理
print(vgg16_false)
vgg16_false.classifier[6]=nn.Linear(4096,10)
print(vgg16_false)
```

# 模型的保存与读取
模型的保存共有两种方法
```python
# model_save.py
import torchvision
import torch
from torch import nn

vgg16=torchvision.models.vgg16()
```
1. 模型结构+参数
```python
torch.save(vgg16,"vgg16_method1.pth")
```
读取方法
```python
model = torch.load("vgg16_method1.pth")
```
2. 仅保留模型参数
```python
torch.save(vgg16.state_dict(),"vgg16_method2.pth")
```
读取方法
```python
vgg16=torchvision.models.vgg16()
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
```
**注意！**：本方法在保存自己创建的模型时必须保证类存在
```python
#陷阱
class Tudui(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1=nn.Conv2d(3,64,3)
    def forward(self,x):
        x=self.conv1(x)
        return x
tudui=Tudui()
torch.save(tudui,'tudui_method1.pth')

#model_load.py
model=torch.load('tudui_method1.pth')#直接读取会导致错误，如需使用则必须保证本文件中存在Tudui类
print(model)
```
解决方案：
* 在`model_load.py`中添加Tudui类定义
* import model_save.py（方法一的变体）  

# 基本套路
```python
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *

# 1.准备训练数据集
train_data = torchvision.datasets.CIFAR10(
    "./data", True, torchvision.transforms.ToTensor(), download=True
)

# 2.准备测试数据集
test_data = torchvision.datasets.CIFAR10(
    "./data", False, torchvision.transforms.ToTensor(), download=True
)

# 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"各长度:1.{train_data_size} 2.{test_data_size}")

# 3.利用dataloader加载数据集
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)
  
# 4.搭建神经网络

# 5.创建网络模型
stone = Stone()
  
# 6.损失函数
loss_fn = torch.nn.CrossEntropyLoss()

# 7.优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(stone.parameters(), lr=learning_rate)
  
# 8.设置训练网络的参数
total_train_step = 0
total_test_step = 0
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./logs_train")

#9.训练步骤开始
for i in range(epoch):
    print("----第{}次训练开始----".format(i + 1))
    for data in train_dataloader:
        imgs, targets = data
        outputs = stone(imgs)
        
        #损失函数计算误差
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #输出展示
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练{}次时，Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
            
    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = stone(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            total_accuracy = (outputs.argmax(1) == targets).sum()
    print("整体测试集loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step += 1

    # 保存训练模型
    torch.save(stone, "stone_{}.path".format(i))
    print("模型已保存")

writer.close()
```
# 如何使用显卡训练
cpu训练很慢，那么我们怎样使用显卡训练呢？
## 调用.cuda()
1. 网络模型
```python
stone = Stone()
if torch.cuda.is_available():
    stone = stone.cuda()
```
2. 数据（输入，标注）
```python
if torch.cuda.is_available():
	imgs = imgs.cuda()
	targets = targets.cuda()
```
3. 损失函数
```python
loss_fn = torch.nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
```

## 使用.to(device)
