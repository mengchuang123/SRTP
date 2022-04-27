# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:50:34 2020

@author: user
"""
import os
import random

a=os.listdir('/homeB/zhuangxianwei/data/yolo3_for_phone/cell-phone-detection-using-yolov3-master/data/train/image/')
image_txt=[]
vaild_txt=[]
#image.txt
picklabelnum =  int(0.2*len(a))
vaild = random.sample(a,picklabelnum)
for i in a:
    if i in vaild:
        vaild_txt.append(os.path.join('/homeB/zhuangxianwei/data/yolo3_for_phone/cell-phone-detection-using-yolov3-master/data/train/image/', i))
    else:
        image_txt.append(os.path.join('/homeB/zhuangxianwei/data/yolo3_for_phone/cell-phone-detection-using-yolov3-master/data/train/image/', i))
    
with open('data/train/image.txt', 'w') as filehandle:
    count = 0
    for listitem in image_txt:
        filehandle.write('%s\n' % listitem)
        count+=1
    print(f"image len :{count}")
with open('data/train/valid.txt', 'w') as filehandle:
    count = 0
    for listitem in vaild_txt:
        filehandle.write('%s\n' % listitem)
        count+=1
    print(f"vaild len :{count}")
