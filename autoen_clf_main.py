'''# ganzer autoencoder
model

# autoencoder
model.encoder(daten)

# autoen + clfhead
model_clf_head(model.encoder(daten))'''


#%%

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from data_module import *
from autoen_module import *
from eval_viz_module import *



device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
if device =='cuda':
    torch.cuda.manual_seed_all(0) 


###########################################################
###########################################################
###########################################################



# FLAGS

train_flag = False
test_flag = True



# HYPS & PARAMETERS

num_epochs = 1 ######5
batch_size = 32 # 64     devide sample in smaller batches
learning_rate = 1e-3 #########1e-3 
weight_decay=1e-5

supervised_ratio = 0.2# 0.99 ######0.9996 #0.2
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

train_dir = 'data/train'
model_path = 'model/autoencoder.pth'
results_path = 'results/results_autoencoder.npy'

train_list = glob.glob(os.path.join(train_dir,'*.jpg')) 

classes = ('cat','dog')


# LOAD AND SPLIT DATA

unsupervised_list, train_list, val_list, test_list = data_split(train_list, supervised_ratio,val_ratio, test_ratio, random_state=0) 

transform_cats = transforms.Compose([   
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.5), (0.5))
])


train_set = dataset(unsupervised_list, transform=transform_cats) 
test_set = dataset(test_list, transform=transform_cats) 

train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size=batch_size, shuffle=True )
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size=batch_size, shuffle=True )

print(f'dataset size: {len(train_set)}') ##########

# check range of values in image tensor
#dataiter = iter(train_loader)
#images_train, labels_train = dataiter.next()
#print(images_train, labels_train)
#print(f'range of values of image tensor: {torch.min(images)}, {torch.max(images)}') # based on original image values that were put in tensor and all the stuff like cropping, flipping...



    
# DEFINE MODEL

model_untrained = Autoencoder()



# TRAIN

if train_flag == True:

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_untrained.parameters(),
                                lr=learning_rate,             
                                weight_decay=weight_decay)

    model, output = autoen_train(num_epochs, train_loader, model_untrained, criterion, optimizer)
    torch.save(model.state_dict(), model_path)



# EVAL & VIZ

if test_flag == True:

    model = Autoencoder()
    model.load_state_dict(torch.load(model_path))

    #for (img, _) in test_loader: # iterating over the batches in dataloader
    dataiter = iter(test_loader) # same but for one batch?
    img, labels = dataiter.next()

    recon = model.encoder(img)
    print(recon.size())

    print('Finished Training')

# %%






























'''
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
from autoen_module import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
if device =='cuda':
    torch.cuda.manual_seed_all(0) 


###########################################################
###########################################################
###########################################################

# FLAGS

train_flag_cnn = False
test_flag_cnn = True



# HYPS & PARAMETERS

num_epochs = 10 ######5
batch_size = 32 # number of samples in one batch patrick4 ##### how is batch-size affecting training?
learning_rate = 1e-3 #####1e-5     #0.001 ###0.001

supervised_ratio = 0.2 ##### 0.2
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

classes = ('cat','dog')

train_dir = 'data/train'
test_dir = 'data/test'
model_path = 'model/autoencoder.pth'
results_path = 'results/results_autoen_clf.npy'

train_list = glob.glob(os.path.join(train_dir,'*.jpg')) 




# LOAD AND SPLIT DATA

unsupervised_list, train_list, val_list, test_list = data_split(train_list,supervised_ratio,val_ratio, test_ratio, random_state=0)

unsupervised_transforms, train_transforms, val_transforms, test_transforms = transf() 

unsupervised_data = dataset(unsupervised_list, transform=unsupervised_transforms)
train_data = dataset(train_list, transform=train_transforms)
val_data = dataset(val_list, transform=val_transforms)
test_data = dataset(test_list, transform=test_transforms)

unsupervised_loader = torch.utils.data.DataLoader(dataset = unsupervised_data, batch_size=batch_size, shuffle=False)
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)



# DEFINE MODEL

model = Autoencoder()
model.load_state_dict(torch.load(model_path))

dataiter = iter(test_loader) # same but for one batch?
img, labels = dataiter.next()

recon = model(img)'''