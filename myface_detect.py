from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1 

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
from PIL import Image

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder('data/test_images_aligned')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        # print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])
# 数据库数据
aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

#测试数据
img=Image.open('data/test_images/angelina_jolie/1.jpg')

img_aligned,pro =  mtcnn(img, return_prob=True)
print('Face detected with probability: {:8f}'.format(pro))

# 
img_list=[]
img_list.append(img_aligned)
img_list = torch.stack(img_list).to(device)
img_embeddings = resnet(img_list).detach().cpu()
img_name=[]
img_name.append('1')
#计算距离
dists = [[(e1 - e2).norm().item() for e2 in img_embeddings] for e1 in embeddings]

print(pd.DataFrame(dists, columns=img_name, index=names))

