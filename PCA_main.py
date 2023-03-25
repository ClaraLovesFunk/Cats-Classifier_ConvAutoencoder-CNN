#%%

import torch
import torch.nn as nn
import os
import glob
from sklearn.decomposition import PCA
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
test_flag = False



# HYPS & PARAMETERS

num_epochs = 5 ######5
batch_size = 25000 # 64
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

dataiter = iter(test_loader) 
img, labels = dataiter.next()

img = torch.flatten(img, start_dim=2, end_dim=-1) # flatten only img, not over samples
img = img.numpy()
print(f' size data_loader: {img.shape}')


'''#%%
#!pip install sklearn numpy matplotlib
#pip install matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import os
import glob
from data_module import *
from autoen_module import *
from sklearn.decomposition import PCA

import matplotlib as plt
from matplotlib.pyplot import imread
import numpy as np


def imshow(img, cmap = None):
    img = img / 2 + 0.5  
    plt.imshow(np.transpose(img, (1, 2, 0)),cmap = cmap)


A = np.random.rand(4, 4) # 4 instances, 4 features
B = np.dot(A, A.transpose())
#print(B)

samples = 500
covariance_matrix = B
X = np.random.multivariate_normal(mean=[0,0,0,0], cov=covariance_matrix, size=samples) # instances 4 dimensional
print(X[0])


pca = PCA(n_components=4).fit(X)
# Now let’s take a look at our components and our explained variances:
print(pca.components_)
print(f'explained var: {pca.explained_variance_ratio_}')

pca_2 = PCA(n_components=2).fit(X)
transformed = pca_2.fit_transform(X)
#plt.scatter(transformed.T[0], transformed.T[1])
'''





#'''
from mpl_toolkits import mplot3d
#pca_3 = PCA(n_components=3).fit(X)
#transformed = pca_3.fit_transform(X)
#fig = plt.figure()
#ax = plt.axes(projection = '3d')
#ax.scatter(transformed.T[0], transformed.T[1], transformed.T[2], alpha=0.3)

from matplotlib.pyplot import imread 
#img = imread("data/train/cat.1.jpg")
img = img.astype(np.uint8)
#print(type(img))
print(img.shape)
img = img.mean(axis=1) # transform to greyscales by taking mean of r,g,b values

#plt.imshow(img, cmap="gray")
#imshow(img, cmap="gray")



print(f'grey image shape {img.shape}')
#img = img.flatten()
#print(f'flattened image shape {img.shape}')
#img = img.reshape(-1, 1)
#print(f'reshaped flattened image shape {img.shape}')

tswizzle_pca = PCA(n_components=400).fit(img)

transformed = tswizzle_pca.transform(img)
print(f'transformed {transformed.shape}')

projected = tswizzle_pca.inverse_transform(transformed)
print(f'projected {projected.shape}')
#'''


# %%

