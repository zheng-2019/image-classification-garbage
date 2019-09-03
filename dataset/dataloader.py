from torch.utils.data import Dataset
from torchvision import transforms as T 
from config import config
from PIL import Image 
from dataset.aug import *
from itertools import chain 
from glob import glob
from tqdm import tqdm
import random 
import numpy as np 
import pandas as pd 
import os 
import cv2
import torch 

#1.set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

#2.define dataset
class SheinDataset(Dataset):
    def __init__(self,label_list,transforms=None,train=True,test=False):
        self.test = test 
        self.train = train 
        imgs = []
        labels = []
        if self.test:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"]))
            self.imgs = imgs 
        else:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"],row["label1"]))
                labels.append(row["label1"])
            self.imgs = imgs
            self.labels = labels
        if transforms is None:
            if self.test or not self.train:
                self.transforms = T.Compose([
                    T.Resize((config.img_weight,config.img_height)),
                    T.ToTensor(),
                    T.Normalize(mean = [0.485,0.456,0.406],
                                std = [0.229,0.224,0.225])])
            else:
                self.transforms  = T.Compose([
                    T.Resize((config.img_weight,config.img_height)),
                    T.RandomRotation(30),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomAffine(45),
                    T.ToTensor(),
                    T.Normalize(mean = [0.485,0.456,0.406],
                                std = [0.229,0.224,0.225])])
        else:
            self.transforms = transforms
    def __getitem__(self,index):
        if self.test:
            filename = self.imgs[index]
            #img = cv2.imread(filename)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = Image.open(filename)
            img = self.transforms(img)
            return img,filename
        else:
            filename,label1 = self.imgs[index] 
            #img = cv2.imread(filename)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = Image.open(filename)
            img = self.transforms(img)
            return img,label1
    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    imgs = []
    label1 = []
    for sample in batch:
        imgs.append(sample[0])
        label1.append(sample[1])

    return torch.stack(imgs, 0), \
           label1

def get_files(root,mode):
    #for test
    if mode == "test":
        files = []
        f_lab = open(root)
        lines_lab = f_lab.readlines()
        for line in tqdm(lines_lab):
            files.append(line[:-1])
        files = pd.DataFrame({"filename":files})
        return files
    elif mode != "test": 
        #for train and val       
        all_data_path,labels1,labels2,labels3,labels4 = [],[],[],[],[]
        print("loading train dataset")
        f_lab = open(root)
        lines_lab = f_lab.readlines()
        for line in tqdm(lines_lab):
            strlist = line.split(' ', 5)
            all_data_path.append(strlist[0])
            #labels1.append(int(strlist[1]))
            #labels2.append(int(strlist[2]))
            #labels3.append(int(strlist[3]))
            labels1.append(int(strlist[1][:-1]))
        all_files = pd.DataFrame({"filename":all_data_path,"label1":labels1})
        return all_files
    else:
        print("check the mode please!")
    
