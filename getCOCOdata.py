from pycocotools.coco import COCO

import requests
import csv
import cv2

#COCO 是一个类, 因此, 使用构造函数创建一个 COCO 对象, 构造函数首先会加载 json 文件, 然后解析图片和标注信息的 id, 根据 id 来创建其关联关系

coco = COCO('./data/coco/annotations/instances_train2017.json')
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]   #获得类名
print('COCO categories: \n{}\n'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['cell phone'])
print(catIds)
imgIds = coco.getImgIds(catIds=catIds )
images = coco.loadImgs(imgIds)
print("imgIds: ", imgIds)
#print("images: ", images)
print(len(imgIds))
print(images[0]["file_name"])

#获取train2017中的全部图像文件名字

import os
a=os.listdir('/homeB/zhuangxianwei/data/yolo3_for_phone/cell-phone-detection-using-yolov3-master/data/coco/images/train2017/train2017/')
print(len(a))
for i  in range(len(a)):
    if a[i]==images[0]["file_name"]:
        print("Finded !!")

if(i == len(a)-1) :
    print("fail!")

# #选出val中的符合要求的图像
for i in range(len(images)):
    file = '/homeB/zhuangxianwei/data/yolo3_for_phone/cell-phone-detection-using-yolov3-master/data/coco/images/train2017/train2017/' + images[i]["file_name"]
    img = cv2.imread(file)
    dir = "/homeB/zhuangxianwei/data/yolo3_for_phone/cell-phone-detection-using-yolov3-master/data/coco/phone_images/" + images[i]["file_name"]
    cv2.imwrite(dir,img)






# #Download annotations
# with open('annotations_download_' + 'cell phone' + '.csv', mode='w', newline='') as annot:
#     for im in images:
#         annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
#         anns = coco.loadAnns(annIds)
#         for i in range(len(anns)):
#             annot_writer = csv.writer(annot)
#             #annot_writer.writerow([im['coco_url'], anns[i]['bbox'][0], anns[i]['bbox'][1], anns[i]['bbox'][0] + anns[i]['bbox'][2], anns[i]['bbox'][1] + anns[i]['bbox'][3], classes])
#             annot_writer.writerow(['downloaded_images/' + im['file_name'], int(round(anns[i]['bbox'][0])), int(round(anns[i]['bbox'][1])), int(round(anns[i]['bbox'][0] + anns[i]['bbox'][2])), int(round(anns[i]['bbox'][1] + anns[i]['bbox'][3])), 'cell phone'])
#             #print("anns: ", im['coco_url'], anns[i]['bbox'][0], anns[i]['bbox'][1], anns[i]['bbox'][0] + anns[i]['bbox'][2], anns[i]['bbox'][1] + anns[i]['bbox'][3], 'person')
#         annot.close()