import os 
import random 
import time
import json
import torch
import torchvision
import numpy as np 
import pandas as pd 
import warnings
from datetime import datetime
from torch import nn,optim
from config import config 
from collections import OrderedDict
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from dataset.dataloader import *
from sklearn.model_selection import train_test_split,StratifiedKFold
from timeit import default_timer as timer
from models.model import *
#from models.resnet import *
from utils import *
from IPython import embed
#1. set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

def eval(val_loader,model):
    csv_map1 = OrderedDict({"filename":[],"result":[],"probability":[]})
    model.eval()
    eq1 = 0
    items = 0
    for i,(input,target1) in enumerate(val_loader):
        with torch.no_grad():
            image_var = Variable(input).cuda()
            y_pred1 = model(image_var)
            _, predicted1 = torch.max(y_pred1.data, 1)
            pred1 = predicted1.cpu().numpy().tolist()
            ann1 = target1
            items = items+1
            if pred1 == ann1 :
                eq1=eq1+1
    Accuracy1 = float(eq1)/float(items)
    print("f+ Class %s : %f"%(str(1),Accuracy1))


def main():
    if config.model_name =='densenet169':
        model = densenet_model()
        print('train densenet169')
    elif config.model_name =='densenet161':
        model = densenet_model()
        print('train densenet161')
    elif config.model_name =='densenet121':
        model = densenet_model()
        print('train densenet121')
    elif config.model_name =='densenet201':
        model = densenet_model()
        print('train densenet201')
    elif config.model_name =='resnet101':
        model = resnet_model()
        print('train resnet101')
    elif config.model_name =='resnet152':
        model = resnet_model()
        print('train resnet152')
    elif config.model_name =='resnet50':
        model = resnet_model()
        print('train resnet50')
    else:
        print('load net err !!! ')
    #model = torch.nn.DataParallel(model)
    model.cuda()

    print('load model :' + config.best_models + config.model_name+ os.sep + str(0) + "/model_best.pth.tar" )
    best_model = torch.load(config.best_models + config.model_name+ os.sep + str(0) + "/model_best.pth.tar")
    model.load_state_dict(best_model["state_dict"])
    val_data_list = get_files(config.test_data,"val")
    val_data = SheinDataset(val_data_list,train=False)
    val_dataloader = DataLoader(val_data,batch_size=1,shuffle=False,collate_fn=collate_fn,pin_memory=False)
    eval(val_dataloader,model)

if __name__ =="__main__":
    main()

