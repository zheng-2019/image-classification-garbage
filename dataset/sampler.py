import torch
import torch.utils.data
import torchvision
import numpy as np

def make_weights_for_balanced_classes(labels,classes):
    labels = np.array(labels)
    #print(labels)
    weights=[]
    outs=[]
    weight = 1./ len(labels)
    for ic in range(classes):
        weights.append(weight/(labels==ic).sum())
    for sim in labels:
        #print(sim)
        outs.append(weights[sim])

 
    return  torch.DoubleTensor(outs)

