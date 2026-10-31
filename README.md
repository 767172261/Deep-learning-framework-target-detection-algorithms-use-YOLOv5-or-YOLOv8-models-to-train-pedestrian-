深度学习框架目标检测算法 使用YOLOv5或YOLOv8模型训练行行人检测数据集通过训练出的模型进行推理 识别行人的检测
@[toc]

使用YOLOv5或YOLOv8训练行人目标检测模型，

行人目标检测图像数据集yolo格式（图像.jpg+标签.txt）目标检测，用于训练YOLOv5、YOLOv8等

已按7：2：1的比例划分好训练集（3056张）：验证集（873张）：测试集（436）

类别：1类：person
### 1. 环境准备
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f878fb6bc93949bcb1e128bbf8e55e71.jpeg)

准备好环境，可以通过以下命令来安装：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ca0e387c77cf420789eb9969adaef13f.png)

```bash
# 安装PyTorch（以CUDA 11.1为例）
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111

# 克隆YOLOv5仓库并安装依赖
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

![对于YOLOv8，由于它可能是Ultralytics未来发布的版本，请关注其官方GitHub页面获取最新安装指南。](https://i-blog.csdnimg.cn/direct/b8d95aa22a13414c8d915da07a73bdaf.png)


### 2. 准备数据集

按照7:2:1比例划分好的数据集，包含图像文件（.jpg）和标签文件（.txt）。接下来需要创建一个YAML文件来描述数据集结构。

`dataset.yaml`示例内容如下：

```yaml
train: /path/to/train/images/
val: /path/to/val/images/
test: /path/to/test/images/

nc: 1  # 类别数量
names: ['person']  # 类别名称
```

请替换路径为实际的数据集位置。

### 3. 开始训练

在YOLOv5中，可以使用以下命令开始训练：

```bash
python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5s.pt
```

参数说明：
- `--img` 设置输入图片大小。
- `--batch` 批处理大小。
- `--epochs` 训练周期数。
- `--data` 数据集配置文件路径。
- `--weights` 预训练权重路径。

### 4. 推理代码

训练完成后，可以使用以下代码对新图片进行推理：

```python
from pathlib import Path
import torch

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')

# 图片路径
img_path = 'path/to/image.jpg'

# 推理
results = model(img_path)

# 显示结果
results.print()  # 输出结果到控制台
results.show()   # 显示结果图片
```


使用YOLOv8进行行人目标检测，从数据准备、环境搭建、模型训练、推理以及性能评估等步骤来详细介绍。假设您已经有了图像和标签文件，并且按照7:2:1的比例划分好了训练集、验证集和测试集。

### 数据准备

#### 格式转换
确保数据集符合YOLO格式，每个图像都有一个对应的`.txt`文件，其中包含边界框信息。每行代表一个对象，格式如下：

```
class_id x_center y_center width height
```

- `x_center`, `y_center`, `width`, 和 `height` 都是相对于图像宽度和高度的相对值（范围从0到1）。
- `class_id` 是类别的索引，在您的案例中仅为`0`（因为只有一个类别：person）。

#### 数据划分
将数据集划分为训练集（3056张）、验证集（873张）和测试集（436张）。接下来需要创建一个YAML文件来描述这些数据集。

```yaml
train: /path/to/train/images/
val: /path/to/val/images/
test: /path/to/test/images/

nc: 1  # 类别数量
names: ['person']  # 类别名称
```

替换路径为实际的数据集位置。

### 环境搭建

首先，克隆YOLOv8仓库并安装依赖：

```bash
git clone https://github.com/ultralytics/yolov8.git
cd yolov8
pip install -r requirements.txt
```

确保您已安装PyTorch和其他必要的库。

### 模型训练

配置超参数并开始训练：

```bash
python train.py --img 640 --batch 16 --epochs 100 --data path/to/dataset.yaml --weights yolov8s.pt --cfg yolov8s.yaml
```

这里：
- `--img` 设置输入图片大小。
- `--batch` 批处理大小。
- `--epochs` 训练周期数。
- `--data` 数据集配置文件路径。
- `--weights` 预训练权重路径。
- `--cfg` 模型配置文件路径。

### 模型推理

训练完成后，可以使用以下代码对新图片进行推理：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/train/exp/weights/best.pt')

# 单个图片推理
results = model('path/to/image.jpg')

# 批量推理
results = model(['path/to/image1.jpg', 'path/to/image2.jpg'])

# 显示结果
for r in results:
    print(r.boxes)  # 输出预测框信息
```

### 性能评估

对于性能评估，您可以直接使用YOLOv8提供的评估工具：

```bash
python val.py --weights runs/train/exp/weights/best.pt --data path/to/dataset.yaml --img 640
```

输出mAP（mean Average Precision）等指标，用于评估模型性能。

请注意，请参考[Ultralytics的官方GitHub页面](https://github.com/ultralytics/yolov8)进行详细指导
