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
batch_size = 32 # 64
learning_rate = 1e-3 
weight_decay=1e-5

supervised_ratio = 0.2
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

unsupervised_transforms, train_transforms, val_transforms, test_transforms = transf() 

train_set = dataset(unsupervised_list, transform = unsupervised_transforms) 
test_set = dataset(test_list, transform = test_transforms)  

train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size=batch_size, shuffle=True )
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size=batch_size, shuffle=True )

print(f'dataset size: {len(train_set)}') 

    
# DEFINE MODEL

model = Autoencoder()
criterion = nn.MSELoss()


# TRAIN

if train_flag == True:

    optimizer = torch.optim.Adam(model.parameters(),
                            lr=learning_rate,             
                            weight_decay=weight_decay)

    model, output = autoen_train(num_epochs, train_loader, model, criterion, optimizer)
    torch.save(model.state_dict(), model_path)



# EVAL & VIZ

if test_flag == True:

    model = Autoencoder()
    model.load_state_dict(torch.load(model_path))

    # plot reconstructed images
    #viz_autoen(test_loader, model, classes)

    # evaluate 
    test_loss = autoen_test(test_loader, model, criterion)

        
# %%