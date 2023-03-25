
##%%

import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from sklearn.model_selection import train_test_split
from PIL import Image
from data_module import *
from cnn_module import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
if device =='cuda':
    torch.cuda.manual_seed_all(0) 


###########################################################
###########################################################
###########################################################

# FLAGS

train_flag_cnn = True
test_flag_cnn = True



# HYPS & PARAMETERS

num_epochs = 1 ######5
batch_size = 1 # number of samples in one batch
learning_rate = 0.1 ###0.001

supervised_ratio = 0.2
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.04

classes = ('cat','dog')

train_dir = 'data/train'
test_dir = 'data/test'
model_path = 'model/cnn.pth'
results_path = 'results/results_cnn.npy'

train_list = glob.glob(os.path.join(train_dir,'*.jpg')) 

# INSPECT DATA

show_data(train_list)
# Class frequencies



# LOAD AND SPLIT DATA

unsupervised_list, train_list, val_list, test_list = data_split(train_list,supervised_ratio,val_ratio, test_ratio, random_state=0)

unsupervised_transforms, train_transforms, val_transforms, test_transforms = transf()

train_data = dataset(train_list, transform=train_transforms)
test_data = dataset(test_list, transform=test_transforms)
val_data = dataset(val_list, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=batch_size, shuffle=True)

#for img, labels in test_loader:
#    x=img

# check range of values in image tensor
dataiter = iter(test_loader)
x, y = dataiter.next()

class test_auto(nn.Module):

    def __init__(self):

        super().__init__() 

        self.conv1 = nn.Conv2d(3, 16, 8)
        self.conv2 = nn.Conv2d(16,32,8)
        self.de_conv2 = nn.ConvTranspose2d(32, 16, 8)
        self.de_conv1 = nn.ConvTranspose2d(16,3,8)
    

        self.pool1 = nn.MaxPool2d(4, return_indices=True)
        self.pool2 = nn.MaxPool2d(8, return_indices=True)
        self.unpool2= nn.MaxUnpool2d(8)
        self.unpool1= nn.MaxUnpool2d(4)   
                

    def forward(self, x):

        # encoder
        print(f'original size {x.size()}')

        conv1_output= self.conv1(x)
        print(f'conv1_output size {conv1_output.size()}')

        max1_output, indices1 = self.pool1(conv1_output)
        print(f'max1_output size {max1_output.size()}')

        conv2_output = self.conv2(max1_output)
        print(f'conv2_output size {conv2_output.size()}')

        max2_output, indices2 = self.pool2(conv2_output)
        print(f'max2_output size {max2_output.size()}')

        # decoder
        unmax2_output = self.unpool2(max2_output, indices2, output_size=conv2_output.size())
        de_conv2_output = self.de_conv2(unmax2_output)

        unmax1_output = self.unpool1(de_conv2_output, indices1, output_size=conv1_output.size())
        de_conv1_output = self.de_conv1(unmax1_output)

        return de_conv1_output
    

model=test_auto()
model(x)