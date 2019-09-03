# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:04:10 2017

@author: hosery
"""

# !/usr/bin/evn python
# coding:utf-8
import os
import cv2
import numpy as np

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import sys


folder = '/home/ltdev/lingtian/jhzheng/s3data/image-classification-garbage/data/garbage_classify/train_data/'
file_train = open('./' + "train_color.txt", 'w')  # 写文件
file_val = open('./' + "val_color.txt", 'w')  # 写文件
file_test = open('./' + "test_color.txt", 'w')  # 写文件

filelist =os.listdir(folder)
cnt = 0
test_cnt = np.zeros((1, 40),dtype=np.int32)
val_cnt = np.zeros((1, 40),dtype=np.int32)
train_cnt = np.zeros((1, 40),dtype=np.int32)
for classfile in filelist:
    if classfile[-3:] == "txt":
        clpath = folder+classfile
        f = open(clpath,'r')
        s = f.read()
        lab =int(s.split(',')[1])


        cnt +=1
        if cnt % 20 == 0:
            pic = clpath.replace('txt','jpg')+" "+str(lab)+"\n"
            val_cnt[0, lab] += 1
            file_val.write(pic)
        elif cnt % 20 == 1:
            pic = clpath.replace('txt','jpg') + " " + str(lab) + "\n"
            test_cnt[0, lab] += 1
            file_test.write(pic)
        else:
            pic = clpath.replace('txt','jpg') + " " + str(lab) + "\n"
            train_cnt[0, lab] += 1
            file_train.write(pic)

print(train_cnt)
print(val_cnt)
print(test_cnt)
file_train.close()
file_val.close()
file_test.close()


