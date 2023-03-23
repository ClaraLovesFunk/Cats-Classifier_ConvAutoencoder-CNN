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
print(f'original: {x.size()}')

conv1 = nn.Conv2d(3, 16, 8)
x= conv1(x)
print(f'conv1: {x.size()}')

pool = nn.MaxPool2d(kernel_size=4, return_indices=True)
x,indices = pool(x)
#indices,x = pool(x)
print(f'pool: {x.size()}')

print('------------------------------------')

pool2 = nn.MaxUnpool2d(4)
x = pool2(x,indices)
#print(f'pool: {x.size()}')


pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=2)
input = torch.tensor([[[[ 1.,  2.,  3.,  4.],
                        [ 5.,  6.,  7.,  8.],
                        [ 9., 10., 11., 12.],
                        [13., 14., 15., 16.]]]])

print(f'input original: {x.size()}')
output, indices = pool(input)
print(f'after pooling: {output.size()}')
output_unpool = unpool(output, indices)
print(f'after unpooling: {output_unpool.size()}')



print(f'original: {x.size()}')

conv1 = nn.Conv2d(3, 16, 8)
x= conv1(x)
print(f'conv1: {x.size()}')

pool = nn.MaxPool2d(2, stride=2, return_indices=True)
x, indices = pool(x)
print(f'pool output: {x.size()}')

unpool = nn.MaxUnpool2d(2, stride=2)
x = unpool(x, indices)
print(f'after unpooling: {x.size()}')'''






input = x # Variable(torch.rand(1,1,64,64))
print(f'original: {x.size()}')

conv1 = nn.Conv2d(3, 16, 8)
conv_output= conv1(x)
print(f'conv1: {conv_output.size()}')

###################################

pool1 = nn.MaxPool2d(2, stride=2, padding=1, return_indices=True)
pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool1= nn.MaxUnpool2d(2, stride=2)
unpool2= nn.MaxUnpool2d(2, stride=2, padding=1)

output1, indices1 = pool1(conv_output)
print(f'pool1: {output1.size()}')
output2, indices2 = pool2(output1)
print(f'pool2: {output2.size()}')

output3 = unpool1(output2, indices2, output_size=output1.size())
print(f'unpool1: {output3.size()}')
output4 = unpool2(output3, indices1, output_size=conv_output.size())
print(f'unpool2: {output4.size()}')

###################################

conv2_T = nn.ConvTranspose2d(16,3,8)
x = conv2_T(output4)
print(f'deconv2: {x.size()}')

#print(f'output: {output4.size()}')