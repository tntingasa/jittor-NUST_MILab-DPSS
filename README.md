
# Jittor 大规模无监督语义分割模型 DPSS

![主要结果](https://github.com/tntingasa/jittor-NUST_MILab-DPSS/blob/main/pic/%E5%9B%BE%E7%89%871.png)


## 简介

本项目包含了第二届计图挑战赛计图 - 大规模无监督语义分割赛题的代码实现。本项目的特点是：针对聚类中心有误差的问题，提出基于聚类距离的样本去噪的方法，使聚类性能得到提升；针对伪标签存在噪声干扰的问题，提出基于去噪的特征微调的方法，校正了噪声标签；针对边缘定位不准确的问题，提出SAM-Auto伪标签增强的方法，令轮廓更加清晰；针对前景定位不准问题，提出基于像素注意力的SAM伪标签生成的方法，提高前景物体定位的准确性；提出的大规模无监督语义分割模型在决赛取得第二，表明了所提方法的有效性。

## 安装 
本项目的安装细节请见 **[USAGE](USAGE.md)**

#### 数据集
本项目的数据集下载地址：[数据集](https://github.com/LUSSeg/ImageNet-S#prepare-the-imagenet-s-dataset-with-one-command)

#### 模型权重
本项目的模型模型下载地址为xxx，下载后放入目录 `<root>/weights/` 下。

## 性能分析
![image](https://user-images.githubusercontent.com/20515144/196449430-5ac6a88c-24ea-4a82-8a45-cd244aeb0b3b.png)


## 致谢
| 对参考的论文、开源库予以致谢，可选

此项目基于论文 *A Style-Based Generator Architecture for Generative Adversarial Networks* 实现，部分代码参考了 [jittor-gan](https://github.com/Jittor/gan-jittor)。

## 注意事项

![image-20220419164035639](https://s3.bmp.ovh/imgs/2022/04/19/6a3aa627eab5f159.png)
