#!/usr/bin/env python
# -*- coding: utf-8 -*-

from calcmfcc import *
import scipy.io.wavfile as wav
import numpy
import os
import scipy.io as io

file_path = 'D:\developers\_Machine_learning\casia1'
mat_path="mfcc_mat_save_path.mat"
pathnames = []
for (dirpath, dirnames, filenames) in os.walk(file_path):
    for filename in filenames:
        if os.path.splitext(filename)[-1] ==".wav":
            pathnames += [os.path.join(dirpath, filename)]

mat= numpy.zeros((len(pathnames),39*5))  #生成一个3行4列全部元素为0的矩阵
N = 0
for file in pathnames:
    (rate,sig) = wav.read(file)
    mfcc_feat = calcMFCC_delta_delta(sig,rate)
    # axis=0，计算每一列的均值
    c1=numpy.mean(mfcc_feat,axis=0) #平均值
    c2=numpy.min(mfcc_feat,axis=0)  # 最小值
    c3=numpy.max(mfcc_feat,axis=0)  # 最大值
    c4=numpy.median(mfcc_feat,axis=0)  # 中值
    c5=numpy.std(mfcc_feat,axis=0)    # 标准偏差
    tt=numpy.concatenate((c1,c2 ,c3,c4,c5),axis=0)
    mat[N,:]=tt
    N+=1
io.savemat(mat_path, {'mfcc_name': mat})
print(mfcc_feat.shape)
print(mat.shape)
