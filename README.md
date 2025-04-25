# BART_Modern_Classical_Chinese_Translation

基于BART预训练模型的现代文和文言文的转换模型

## 概述

此仓库仅包含三个模型配置参数，没有模型的权重参数  
其中两个经过训练的模型参数相同，与预训练模型参数有些许区别  
预训练权重参数，经过训练的C2M和M2C权重参数，以及训练集和测试集都存储在百度网盘  
网盘链接：[https://pan.baidu.com/s/1wGJh9IxXMMzJA3BsKwAUbw](https://pan.baidu.com/s/1wGJh9IxXMMzJA3BsKwAUbw) 提取码: bart

## 环境

PyTorch: 2.7.0  
transformers: 4.51.3  
datasets: 3.5.0  
不严格要求版本必须一致，但差距过大可能会有警告

## 用法

第一步，克隆该存储库
```
git clone https://github.com/Hamletlx/BART_Modern_Classical_Chinese_Translation.git
```
第二步，从百度网盘中下载对应的模型权重参数放到对应的文件夹，如果需要运行训练代码，把数据集也下载下来  
网盘链接：[https://pan.baidu.com/s/1wGJh9IxXMMzJA3BsKwAUbw](https://pan.baidu.com/s/1wGJh9IxXMMzJA3BsKwAUbw) 提取码: bart

第三步，运行predict.py进行测试

拓展，根据predict.py文件里示例，可以自行将其部署到其他应用，如：  
demo1.py：使用 PySide6 简单部署桌面应用程序  
demo2.py：使用 Flask 简单部署Web应用程序  
