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
from eval_viz_module import *

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
batch_size = 5 # number of samples in one batch
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
#test_list = glob.glob(os.path.join(test_dir, '*.jpg'))



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
'''
print(f'original dataloader: {x.size()}')

conv1 = nn.Conv2d(3, 16, 8)
x= conv1(x)
print(f'conv1: {x.size()}')

pool = nn.MaxPool2d(4, 4)
x = pool(x)
print(f'pool: {x.size()}')

conv2 = nn.Conv2d(16,32,8)
x = conv2(x)
print(f'conv2: {x.size()}')

pool = nn.MaxPool2d(8, 8)
x = pool(x)
print(f'pool: {x.size()}')

#conv3 = nn.Conv2d(16,32,8)
#x = conv3(x)
#print(f'conv3: {x.size()}')




print('------------------------------------')

size_flattened_x = x.size()[1]*x.size()[2]*x.size()[3]
x = x.view(-1, size_flattened_x)                          
print(f'reshape with view(): {x.size()}')


fc1 = nn.Linear(in_features= size_flattened_x, out_features=120)
x = fc1(x)
print(f'fc1: {x.size()}')

fc2 = nn.Linear(120, 84)
x = fc2(x)
print(f'fc2: {x.size()}')

fc3 = nn.Linear(84, 2)
x = fc3(x)
print(f'fc3: {x.size()}')'''





'''conv1 = nn.Conv2d(3, 16, 8)
x= conv1(x)
print(f'conv1: {x.size()}')

pool = nn.MaxPool2d(4)
x = pool(x)
print(f'pool: {x.size()}')

conv2 = nn.Conv2d(16,32,8)
x = conv2(x)
print(f'conv2: {x.size()}')

pool = nn.MaxPool2d(kernel_size=8)
x = pool(x)
print(f'pool: {x.size()}')



print('------------------------------------')

conv2_T = nn.ConvTranspose2d(32,16,8, stride=12)
x = conv2_T(x)
print(f'deconv2: {x.size()}')'''

print(f'original: {x.size()}')

conv1 = nn.Conv2d(3, 16, 8)
x= conv1(x)
print(f'conv1: {x.size()}')

pool = nn.MaxPool2d(4)
x = pool(x)
print(f'pool: {x.size()}')

conv2 = nn.Conv2d(16,32,8)
x = conv2(x)
print(f'conv2: {x.size()}')

pool = nn.MaxPool2d(8)
x = pool(x)
print(f'pool: {x.size()}')


print('------------------------------------')

conv2_T = nn.ConvTranspose2d(32,16,8, stride = 11, output_padding=3) #11, 2
x = conv2_T(x)
print(f'deconv2: {x.size()}')

conv2_T = nn.ConvTranspose2d(16,3,8, stride = 4, output_padding=1)
x = conv2_T(x)
print(f'deconv2: {x.size()}')


