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
from dataset.sampler import *
from torch.utils.data.sampler import  WeightedRandomSampler
from dataset.sampler import *
from sklearn.model_selection import train_test_split,StratifiedKFold
from timeit import default_timer as timer
from models.model import *
from utils import *
#from IPython import embed
#1. set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

is_train = True
#is_train = False

#2. evaluate func
def evaluate(val_loader,model,criterion,epoch):
    # define meters
    losses = AverageMeter()
    top1 = AverageMeter()
    #progress bar
    val_progressor = ProgressBar(mode="Val  ",epoch=epoch,total_epoch=config.epochs,model_name=config.model_name,total=len(val_loader))
    # switch to evaluate mode and confirm model has been transfered to cuda
    model.cuda()
    model.eval()#for test
    with torch.no_grad():
        for i,(input,target1) in enumerate(val_loader):
            val_progressor.current = i 
            input = Variable(input).cuda()
            target1 = Variable(torch.from_numpy(np.array(target1)).long()).cuda()
            # compute output
            output1 = model(input)
            loss1 = criterion(output1,target1)
            loss = loss1
            # measure accuracy and record loss
            precision11,precision21 = accuracy(output1,target1,topk=(1,2))
            losses.update(loss.item(),input.size(0))
            top1.update(precision11[0],input.size(0))
            val_progressor.current_loss = losses.avg
            val_progressor.current_top1 = top1.avg
            val_progressor()
        val_progressor.done()
    return [losses.avg,top1.avg]

#3. test model on public dataset and save the probability matrix
def testing(test_loader,model):
    # confirm the model converted to cuda
    csv_map1 = OrderedDict({"filename":[],"result":[],"probability":[]})
    model.eval()
    for i,(input,filepath) in enumerate(test_loader):
        # change everything to cuda and get only basename
        with torch.no_grad():
            image_var = Variable(input).cuda()
            y_pred1 = model(image_var)
            _, predicted1 = torch.max(y_pred1.data, 1)
            #print(predicted1)

            smax = nn.Softmax(1)
            smax_out1 = smax(y_pred1)

        # save probability to csv files
        csv_map1["filename"].extend(filepath)
        csv_map1["result"].extend(predicted1.cpu().numpy())
        for output in smax_out1:
            prob = ";".join([str(i) for i in output.data.tolist()])
            csv_map1["probability"].append(prob)
    result = pd.DataFrame(csv_map1)
    result["probability"] = result["probability"].map(lambda x : [float(i) for i in x.split(";")])
    result.to_csv("./submit/{}_submission.csv" .format(config.model_name + "_" + str(0)),index=False,header = None)

#4. more details to build main function    
def main():
    fold = 0
    # mkdirs
    if not os.path.exists(config.submit):
        os.mkdir(config.submit)
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.weights + config.model_name + os.sep +str(fold) + os.sep):
        os.makedirs(config.weights + config.model_name + os.sep +str(fold) + os.sep)
    if not os.path.exists(config.best_models + config.model_name + os.sep +str(fold) + os.sep):
        os.makedirs(config.best_models + config.model_name + os.sep +str(fold) + os.sep)       
    # get model and optimizer
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
    #model = resnet_model()
    #model = densenet_model()
    #model = torch.nn.DataParallel(model)
    model.cuda()

    #optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=config.weight_decay)
    optimizer = optim.Adam(model.parameters(),lr = config.lr,amsgrad=True,weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()
    # some parameters for  K-fold and restart model
    start_epoch = 0
    best_precision1 = 0
    best_precision_save = 0
    #resume = False
    #resume = True
    
    # restart the training process
    if config.resume:
        checkpoint = torch.load(config.best_models + config.model_name+ os.sep + str(0) + "/model_best.pth.tar")
        start_epoch = checkpoint["epoch"]
        fold = checkpoint["fold"]
        best_precision1 = checkpoint["best_precision1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # get files and split for K-fold dataset
    # read files
    train_data_list = get_files(config.train_data,"train")
    val_data_list = get_files(config.val_data,"val")
    #test_files = get_files(config.test_data,"test")

    train_data = SheinDataset(train_data_list)
    val_data = SheinDataset(val_data_list,train=False)
    weights = make_weights_for_balanced_classes(train_data.labels,config.num_garbages)

    # load dataset
    if config.randomsimple==True:#权重采样
        samples_weight=WeightedRandomSampler(weights,num_samples=len(train_data.labels),replacement=True)
        train_dataloader = DataLoader(train_data,sampler=samples_weight,batch_size=config.batch_size,collate_fn=collate_fn,pin_memory=True)
    else:
        train_dataloader = DataLoader(train_data,batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn,pin_memory=True)
    val_dataloader = DataLoader(val_data,batch_size=config.batch_size * 2,shuffle=True,collate_fn=collate_fn,pin_memory=False)

    # optim
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,"max",verbose=1,patience=3)
    scheduler =  optim.lr_scheduler.StepLR(optimizer,step_size = 10,gamma=0.1)
    # define metrics
    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    valid_loss = [np.inf,0,0]
    model.train()#for train
    # train
    start = timer()
    for epoch in range(start_epoch,config.epochs):
        scheduler.step(epoch)
        train_progressor = ProgressBar(mode="Train",epoch=epoch,total_epoch=config.epochs,model_name=config.model_name,total=len(train_dataloader))
        # train
        model.train()#for train mode
        #cont = [0]*62
        for iter,(input,target1) in enumerate(train_dataloader):
            # switch to continue train process
            #for lab in target1:
            #   cont[lab]=cont[lab]+1
            #print(cont) 

            train_progressor.current = iter
            model.train()
            input = Variable(input).cuda()
            target1 = Variable(torch.from_numpy(np.array(target1)).long()).cuda()
            output1 = model(input)

            loss1 = criterion(output1,target1)
            precision1_train1,precision2_train1 = accuracy(output1,target1,topk=(1,2))

            train_losses.update(loss1.item(),input.size(0))
            train_top1.update(precision1_train1[0],input.size(0))
            train_progressor.current_loss = train_losses.avg
            train_progressor.current_top1 = train_top1.avg
            #backward
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            train_progressor()

        train_progressor.done()
        #evaluate
        lr = get_learning_rate(optimizer)
        #evaluate every half epoch
        valid_loss = evaluate(val_dataloader,model,criterion,epoch)
        is_best = valid_loss[1] > best_precision1
        best_precision1 = max(valid_loss[1],best_precision1)
        try:
            best_precision_save = best_precision1.cpu().data.numpy()
        except:
            pass
        save_checkpoint({
                    "epoch":epoch + 1,
                    "model_name":config.model_name,
                    "state_dict":model.state_dict(),
                    "best_precision1":best_precision1,
                    "optimizer":optimizer.state_dict(),
                    "fold":fold,
                    "valid_loss":valid_loss,
        },is_best,fold)
    
    best_model = torch.load(config.best_models +config.model_name+ os.sep+ str(fold) +os.sep+ 'model_best.pth.tar')
    model.load_state_dict(best_model["state_dict"])


def test():
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
    model = torch.nn.DataParallel(model)
    model.cuda()

    print('load model :' + config.best_models + config.model_name+ os.sep + str(0) + "/model_best.pth.tar" )
    best_model = torch.load(config.best_models + config.model_name+ os.sep + str(0) + "/model_best.pth.tar")
    model.load_state_dict(best_model["state_dict"])
    test_files = get_files(config.test_data,"test")
    test_dataloader = DataLoader(SheinDataset(test_files,test=True),batch_size=1,shuffle=False,pin_memory=False)
    testing(test_dataloader,model)

if __name__ =="__main__":
    if is_train:
        main()
    else:
        test()







