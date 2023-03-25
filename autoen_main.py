#%%

import torch
import torch.nn as nn
import os
import glob
from data_module import *
from autoen_module import *



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

num_epochs = 5 ######5
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

data_list = glob.glob(os.path.join(train_dir,'*.jpg')) ############## RENAME DATALIST! (also in dataset,)

classes = ('cat','dog')


# LOAD AND SPLIT DATA

unsupervised_list, train_list, val_list, test_list = data_split(data_list, supervised_ratio,val_ratio, test_ratio, random_state=0) 

unsupervised_transforms, train_transforms, val_transforms, test_transforms = transf() 

data_data = dataset(data_list, transform = unsupervised_transforms) 
unsupervised_data = dataset(unsupervised_list, transform = unsupervised_transforms) 
train_data = dataset(train_list, transform=train_transforms)
val_data = dataset(val_list, transform=val_transforms)
test_data = dataset(test_list, transform=test_transforms)

data_loader = torch.utils.data.DataLoader(dataset = data_data, batch_size=batch_size, shuffle=False)
unsupervised_loader = torch.utils.data.DataLoader(dataset = unsupervised_data, batch_size=batch_size, shuffle=False)
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)

print(f'dataset size: {len(data_data)}') 
    


# DEFINE MODEL

model = Autoencoder()
criterion = nn.MSELoss()


# TRAIN

if train_flag == True:

    optimizer = torch.optim.Adam(model.parameters(),
                            lr=learning_rate,             
                            weight_decay=weight_decay)

    model, output = autoen_train(num_epochs, data_loader, model, criterion, optimizer)
    torch.save(model.state_dict(), model_path)



# EVAL & VIZ

if test_flag == True:

    model = Autoencoder()
    model.load_state_dict(torch.load(model_path))

    # plot reconstructed images
    viz_autoen(train_loader, model, classes)

    # evaluate 
    test_loss = autoen_test(test_loader, model, criterion)

        
# %%