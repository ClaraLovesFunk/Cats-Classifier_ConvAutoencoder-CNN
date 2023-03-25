#%%

import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
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

train_flag_cnn = False
test_flag_cnn = True



# HYPS & PARAMETERS

num_epochs = 10 ######5
batch_size = 32 
learning_rate = 1e-3 #####1e-5     

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



# LOAD AND SPLIT DATA

unsupervised_list, train_list, val_list, test_list = data_split(train_list, supervised_ratio,val_ratio, test_ratio, random_state=0) 

unsupervised_transforms, train_transforms, val_transforms, test_transforms = transf() 

unsupervised_data = dataset(unsupervised_list, transform = unsupervised_transforms) 
train_data = dataset(train_list, transform=train_transforms)
val_data = dataset(val_list, transform=val_transforms)
test_data = dataset(test_list, transform=test_transforms)

unsupervised_loader = torch.utils.data.DataLoader(dataset = unsupervised_data, batch_size=batch_size, shuffle=True )
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)

print(f'dataset size: {len(train_data)}') 



# DEFINE MODEL

model = cnn_cats().to(device)

optimizer = optim.Adam(params = model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss() 

model.train()



# TRAIN

if train_flag_cnn == True:

    model = train_cnn(num_epochs, train_loader, model, criterion, optimizer, val_loader)

    torch.save(model.state_dict(), model_path)




# EVAL 


if test_flag_cnn == True:
    
    model.load_state_dict(torch.load(model_path))

    test_accuracy, test_loss = test_cnn(val_loader, model, criterion)
    
 
# %%
