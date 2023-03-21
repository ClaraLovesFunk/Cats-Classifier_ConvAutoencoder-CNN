#%%

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
from collections import Counter

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


# HYPS & PARAMETERS

num_epochs = 5 ######5
batch_size = 100 # number of samples in one batch patrick4 ##### how is batch-size affecting training?
learning_rate = 0.001 ###0.001

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

transform_cats = transforms.Compose([   
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.5), (0.5))
])

data_set = dataset(train_list,transform = transform_cats)
data_loader = torch.utils.data.DataLoader(dataset = data_set, batch_size=batch_size, shuffle=True )


# SHOW DATA

#show_data(train_list)



# CLASS SIZES

#train_classes = [label for _, label in data_set]
#print(Counter(train_classes))

  
data = {'Cats':12500, 'Dogs':12500} #{'Cats':Counter(train_classes)[0], 'Dogs':Counter(train_classes)[1]}  ########{'Cats':12500, 'Dogs':12500}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
plt.bar(courses, values, color ='maroon',#, color ='maroon'
        width = 0.4)
 
plt.xlabel("Classes")
plt.ylabel("No. of Instances")
plt.title("Class Sizes")
plt.show()



# %%
