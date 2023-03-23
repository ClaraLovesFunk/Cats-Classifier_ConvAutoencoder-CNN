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

train_flag_autoen_clf = False
test_flag_autoen_clf = True



# HYPS & PARAMETERS

num_epochs = 10 ######5
batch_size = 32 # 64
learning_rate = 1e-3 
weight_decay=1e-5

supervised_ratio = 0.2
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

train_dir = 'data/train'
autoen_path = 'model/autoencoder.pth'
autoen_clf_head_path = 'model/autoencoder_clf_head.pth'
results_path = 'results/results_autoencoder.npy'

train_list = glob.glob(os.path.join(train_dir,'*.jpg')) 

classes = ('cat','dog')



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

    
# MODEL

# load trained autoencoder
autoen = Autoencoder()
autoen.load_state_dict(torch.load(autoen_path))
autoen.to(device)

# define classification head
autoen_clf_head = Autoencoder_Clf_Head()
optimizer = optim.Adam(params = autoen_clf_head.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss() 



# TRAIN

if train_flag_autoen_clf == True:

    autoen_clf_head.to(device)

    autoen_clf_head = train_autoen_clf_head(num_epochs, train_loader, autoen, autoen_clf_head, criterion, optimizer, val_loader)

    torch.save(autoen_clf_head.state_dict(), autoen_clf_head_path)



# EVAL

if test_flag_autoen_clf == True:
 
    autoen_clf_head.load_state_dict(torch.load(autoen_clf_head_path))
    autoen_clf_head.to(device)
    
    with torch.no_grad():
            test_accuracy=0
            test_loss =0
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device)


                # encoder steps
                conv1_output= autoen.conv1(data)
                max1_output, indices1 = autoen.pool1(conv1_output)

                conv2_output = autoen.conv2(max1_output)
                max2_output, indices2 = autoen.pool2(conv2_output)

                # classification head steps
                val_outputs = autoen_clf_head(max2_output)


                val_loss = criterion(val_outputs,label)
                acc = ((val_outputs.argmax(dim=1) == label).float().mean())
                test_accuracy += acc/ len(val_loader)
                test_loss += val_loss/ len(val_loader)
                
            print('test_accuracy : {}, test_loss : {}'.format(test_accuracy,test_loss))

