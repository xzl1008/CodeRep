# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 22:01:54 2018

@author: xiezhilong
"""
from numpy import exp, array, random, dot
#训练集数据
training_set_inputs = array([[0, 0 , 1],[1, 1, 1],[1, 0, 1], [0, 1, 1]])
#训练标签初始化
training_set_outputs = array([[0, 1, 1, 0]]).T
#偏置项初始化
bias = 0
#学习率
alpha = 0.01
#设置随机种子
random.seed(1)
#权重参数初始化
synaptic_weights = 2 * random.random((3, 1)) - 1
#训练1万次，每次将所有训练数据都用于计算
for iteration in range(10000):
    #用sigmoid作为激活函数，求出当前参数条件下的训练数据的输出
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights) + bias)))
    #根据梯度下降，更新权重
    synaptic_weights += (1 / 4) * alpha * dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
    #更新偏置项
    bias += (1 / 4) * alpha * ((training_set_outputs - output) * output * (1 - output)).sum()
    
#输出样本[1, 0, 0]的预测值   
print(1/ (1 + exp( + exp(-dot(array([1, 0, 0]), synaptic_weights) + bias))))

l = [i[0] for i in synaptic_weights]
l.append(bias)
#显示多项式
print("y = %.2f*x1 + %.2f*x2 + %.2f*x3 + %.2f" % tuple(l))