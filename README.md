# 代码说明  
这个程序是基于Pytorch实现Seq2Seq(attention)功能(用于序列图片的识别)。     
本代码目的：方便学习Seq2Seq(attention)细节，所以重点关注model的实现部分。  
　　　　　　理论上：model的实现部分和原论文中的公式对比过，应该没大问题。  
　　　　　　实践上：自己在实验室项目中已经使用过此模型，从实验效果来看模型没问题。  
　　　　　　为了便于学习，我把原始程序上做了一些删减，应该不会影响程序的正确性。  

### 开发环境    
Pytorch0.4.0      
tensorlfow  
keras  
window或linux系统皆可  
此代码是cpu版，若想使用gpu可以在代码把注释行改掉   

### 文件夹说明  
1、piture_row:	存放原始图片文件  
2、piture_npy:	存放经过inceptionv3模型提取的2048维特征向量    
3、picture_info:存放label、train、val的信息   

### 程序文件说明
1、extractor.py：使用keras调用inceptionv3从原始图片中提取2048维特征向量   
2、data.py:　　　数据处理程序   
3、picture_info：存放label、train、val的信息   
4、config.py:　　记录整个模型的一些超参数   
5、train.py：　　训练程序入口   

### 程序运行   
先运行extractor.py提取特征向量。再运行train.py  
把所需要的安装包都pip安装好就没啥大问题，就是如此so easy！  

# Seq2Seq(attention)模型详解   
### 论文链接  
[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
### 模型框架图   
在一些细节上和代码有些出入，但是不影响对模型的理解
![picture1](https://github.com/Liu-Yicheng/seq2seq-attention-_on_picture/raw/master/structure.jpg)  
