#%%

import torch
import torch.nn as nn
import os
import glob
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from matplotlib.pyplot import imread
from joblib import dump, load

import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os


from data_module import *
from autoen_module import *
from PCA_module import *



device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
if device =='cuda':
    torch.cuda.manual_seed_all(0) 


###########################################################
###########################################################
###########################################################



# FLAGS

train_PCA_flag = False

train_PCA_Clf_flag = False
test_PCA_Clf_flag = False



# HYPS & PARAMETERS

num_epochs = 10 
batch_size = 32  
learning_rate = 1e-3 
weight_decay=1e-5

supervised_ratio = 0.2
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

train_dir = 'data/train'
model_PCA_path = 'model/pca.joblib'
model_PCA_Clf_head_path = 'model/pcar_clf_head.pth'

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

data_loader = torch.utils.data.DataLoader(dataset = data_data, batch_size=len(data_data), shuffle=False) # entire data in one batch
unsupervised_loader = torch.utils.data.DataLoader(dataset = unsupervised_data, batch_size=batch_size, shuffle=False)
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)




# PCA

if train_PCA_flag == True:

    # access all data
    dataiter = iter(data_loader) 
    img_all, labels = dataiter.next()

    # prepare data for PCA
    img_all = torch.flatten(img_all, start_dim=2, end_dim=-1) # flatten only img, not over samples
    img_all = img_all.numpy() # to numpy
    img_all = img_all.astype(np.uint8) # ?
    img_all = img_all.mean(axis=1) # transform to greyscales by taking mean of r,g,b values

    # train PCA
    tswizzle_pca = PCA(n_components=800).fit(img_all)
    dump(tswizzle_pca, model_PCA_path)

    print('PCA fitted and saved')



    
# CLASSIFICATION HEAD

# load PCA-model
model_PCA = load(model_PCA_path) 

# define classification head
PCA_clf_head = Autoencoder_Clf_Head()
optimizer = optim.Adam(params = PCA_clf_head.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss() 


if train_PCA_Clf_flag == True:

    PCA_clf_head = train_PCA_clf_head(num_epochs, train_loader, model_PCA, PCA_clf_head, criterion, optimizer, val_loader)

    torch.save(PCA_clf_head.state_dict(), model_PCA_Clf_head_path)


if test_PCA_Clf_flag == True:
 
    PCA_clf_head.load_state_dict(torch.load(model_PCA_Clf_head_path))

    test_accuracy, test_loss = test_PCA_clf(val_loader, model_PCA, PCA_clf_head, criterion)
    
