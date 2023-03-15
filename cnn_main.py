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
from eval_module import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
if device =='cuda':
    torch.cuda.manual_seed_all(0) 


###########################################################
###########################################################
###########################################################

# FLAGS

train_flag = True
test_flag = True



# HYPS & PARAMETERS

num_epochs = 0 ######5
batch_size = 4
learning_rate = 0.1 ###0.001

supervised_ratio = 0.2
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

classes = ('cat','dog')

train_dir = 'data/train'
test_dir = 'data/test'
model_path = 'model/cnn.pth'
results_path = 'results/results_cnn.npy'

train_list = glob.glob(os.path.join(train_dir,'*.jpg')) 
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))



# INSPECT DATA

show_data(train_list)
# Class frequencies



# LOAD AND SPLIT DATA

unsupervised_list, train_list, val_list, test_list = data_split(train_list,supervised_ratio,val_ratio, test_ratio)

train_transforms, val_transforms, test_transforms = transf()

train_data = dataset(train_list, transform=train_transforms)
test_data = dataset(test_list, transform=test_transforms)
val_data = dataset(val_list, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=batch_size, shuffle=True)



# DEFINE MODEL

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss() #####FOR MULTICLASS, softmax already included in loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



# TRAIN

if train_flag == True:

    model = cnn_cats_train(train_loader, num_epochs, model, criterion, optimizer, model_path)
    torch.save(model.state_dict(), model_path)



# EVAL 

if test_flag == True:
    
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(model_path))

    acc, loss = eval_cats_clf(val_loader, model, criterion)

    results = {
        'acc': acc,
        'loss': loss
    }
    np.save(results_path, results) 
    results = np.load(results_path,allow_pickle='TRUE').item()

    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')

    for metric in results:
        print(f'{metric}: {results[metric]}')
