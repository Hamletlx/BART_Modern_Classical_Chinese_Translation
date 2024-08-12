# BART_Modern_Classical_Chinese_Translation

基于BART预训练模型的现代文和文言文的转换模型

## 概述

此仓库包含[预训练参数](bart_base_chinese)，以及[文言文转换为现代文](final_model_C2M)和[现代文转换为文言文](final_model_M2C)的参数
还有[训练集](data_train.csv)和[测试集](data_test.csv)

## 环境

PyTorch: 2.3.1
transformers: 4.44.0

## 注意

[预训练参数文件](bart_base_chinese/model.safetensors)，[文言文转换为现代文参数文件](final_model_C2M/model.safetensors)，[现代文转换为文言文参数文件](final_model_M2C/model.safetensors)的大小都为**500MB**左右

## 演示

https://github.com/user-attachments/assets/d4e006a9-d656-4e8a-8dba-9d9fcea70537
