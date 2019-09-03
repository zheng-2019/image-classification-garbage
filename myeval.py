
import os 
import random 
import time
import torch
import numpy as np 
import pandas as pd 
from datetime import datetime
from config import config 
from collections import OrderedDict
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from dataset.dataloader import *
from config import config
from models.model import *

garbages_Type = (
"其他垃圾/一次性快餐盒",
"其他垃圾/污损塑料",
"其他垃圾/烟蒂",
"其他垃圾/牙签",
"其他垃圾/破碎花盆及碟碗",
"其他垃圾/竹筷",
"厨余垃圾/剩饭剩菜",
"厨余垃圾/大骨头",
"厨余垃圾/水果果皮",
"厨余垃圾/水果果肉",
"厨余垃圾/茶叶渣",
"厨余垃圾/菜叶菜根",
"厨余垃圾/蛋壳",
"厨余垃圾/鱼骨",
"可回收物/充电宝",
"可回收物/包",
"可回收物/化妆品瓶",
"可回收物/塑料玩具",
"可回收物/塑料碗盆",
"可回收物/塑料衣架",
"可回收物/快递纸袋",
"可回收物/插头电线",
"可回收物/旧衣服",
"可回收物/易拉罐",
"可回收物/枕头",
"可回收物/毛绒玩具",
"可回收物/洗发水瓶",
"可回收物/玻璃杯",
"可回收物/皮鞋",
"可回收物/砧板",
"可回收物/纸板箱",
"可回收物/调料瓶",
"可回收物/酒瓶",
"可回收物/金属食品罐",
"可回收物/锅",
"可回收物/食用油桶",
"可回收物/饮料瓶",
"有害垃圾/干电池",
"有害垃圾/软膏",
"有害垃圾/过期药物"
)

def myeval(val_loader,model1):
    print('get start!!!')
    print(len(garbages_Type))
    print()
    ann1 = np.zeros((1, config.num_garbages),dtype=np.int32)
    pre1 = np.zeros((1, config.num_garbages),dtype=np.int32)
    confus1 = np.zeros((config.num_garbages, config.num_garbages),dtype=np.int32)
    for iter,(input,target1) in enumerate(val_loader):
        with torch.no_grad():
            if config.testwithgpu:
                image_var = Variable(input).cuda()
            else:
                image_var = Variable(input)
            det_tic = time.time()
            y_pred1 = model1(image_var)
            misc_toc = time.time()
            nms_time = misc_toc - det_tic
            print('Inference_Time: {:.5f} s/image'.format(nms_time))

            _, predicted1 = torch.max(y_pred1.data, 1)
            ann1[0,target1.item()]+=1
            pre1[0,target1.item()]+= (predicted1.cpu()==target1)
            confus1[target1.item(),predicted1.cpu().item()]+=1

    if not os.path.exists(config.submit):
        os.mkdir(config.submit)
    
    print(len(garbages_Type))
    print('garbages eval value !!!')    
    for i in range(len(garbages_Type)):
        if ann1[0,i]>0:
            prate = float(pre1[0,i])/float(ann1[0,i])
            print(f"+ Class '{garbages_Type[i]}' pre:{pre1[0,i]} ann:{ann1[0,i]} - AP: {prate}")
    print("garbages Precision : {:.4%}".format(float(pre1.sum())/float(ann1.sum())))


    pd_data=pd.DataFrame(confus1,index=garbages_Type,columns=garbages_Type)
    pd_data.to_csv('./submit/conf_garbages_data.csv')


def loadnet(model_name,model_path,task_name):
    if model_name =='densenet121':
        model = densenet_model()
        print('train'+config.model_name)
    else:
        print('load net err !!! ')
    if config.testwithgpu:
        model.cuda()
    print(model_path)
    best_model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(best_model["state_dict"])
    model.eval()
    return model

def evaltest():   
    modelpath = config.best_models + config.model_name+ os.sep + str(0) + "/model_best.pth.tar"
    print(config.best_models + config.model_name+ os.sep + str(0) + "/model_best.pth.tar")
    model_garbages = loadnet(config.model_name,modelpath,'num_garbages')

    val_files = get_files(config.test_data,"val")
    val_dataloader = DataLoader(SheinDataset(val_files,test=False),batch_size=1,shuffle=False,pin_memory=False)
    myeval(val_dataloader,model_garbages)
    

if __name__ =="__main__":
    evaltest()




