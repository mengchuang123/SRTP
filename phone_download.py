# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:33:12 2020

@author: user
"""

from pycocotools.coco import COCO

import requests
import csv

#COCO 是一个类, 因此, 使用构造函数创建一个 COCO 对象, 构造函数首先会加载 json 文件, 然后解析图片和标注信息的 id, 根据 id 来创建其关联关系

coco = COCO('./data/coco/annotations/instances_val2017.json')
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]   #获得类名
print('COCO categories: \n{}\n'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['cell phone'])
imgIds = coco.getImgIds(catIds=catIds )
images = coco.loadImgs(imgIds)
print("imgIds: ", imgIds)
print("images: ", images)

# for im in images:
#     print("im: ", im)
#     with open('downloaded_images/' + im['file_name'], 'wb') as handler:
#         handler.write(img_data)

#Download annotations
with open('annotations_download_' + 'cell phone' + '.csv', mode='w', newline='') as annot:
    for im in images:
        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for i in range(len(anns)):
            annot_writer = csv.writer(annot)
            #annot_writer.writerow([im['coco_url'], anns[i]['bbox'][0], anns[i]['bbox'][1], anns[i]['bbox'][0] + anns[i]['bbox'][2], anns[i]['bbox'][1] + anns[i]['bbox'][3], classes])
            annot_writer.writerow(['downloaded_images/' + im['file_name'], int(round(anns[i]['bbox'][0])), int(round(anns[i]['bbox'][1])), int(round(anns[i]['bbox'][0] + anns[i]['bbox'][2])), int(round(anns[i]['bbox'][1] + anns[i]['bbox'][3])), 'cell phone'])
            #print("anns: ", im['coco_url'], anns[i]['bbox'][0], anns[i]['bbox'][1], anns[i]['bbox'][0] + anns[i]['bbox'][2], anns[i]['bbox'][1] + anns[i]['bbox'][3], 'person')
        annot.close()