
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

# HYPS
num_epochs = 1 ######5
batch_size = 4
learning_rate = 0.1 ###0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1234)
if device =='cuda':
    torch.cuda.manual_seed_all(1234)


train_dir = 'data/train'
test_dir = 'data/test'

#print(os.listdir(train_dir)[:5])

train_list = glob.glob(os.path.join(train_dir,'*.jpg')) ############
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

#print(len(train_list))


# show data
from PIL import Image
random_idx = np.random.randint(1,25000,size=10)

fig = plt.figure()
i=1
for idx in random_idx:
    ax = fig.add_subplot(2,5,i)
    img = Image.open(train_list[idx])
    plt.imshow(img)
    i+=1

plt.axis('off')
plt.show()




# SPLIT DATA

# supervised vs unsupervised

# train_val vs test

# train vs val
train_list, val_list = train_test_split(train_list, test_size=0.2)




# data transformation                             
train_transforms =  transforms.Compose([     #common image transformations, that can be chained together via .compose
        transforms.Resize((224, 224)), #Resize the input image to the given size.
        transforms.RandomResizedCrop(224), #Crop a random portion of image and resize it to a given size
        transforms.RandomHorizontalFlip(), #Horizontally flip the given image randomly with a given probability.
        transforms.ToTensor(), # Convert a PIL Image or numpy.ndarray to tensor.
    ])

val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

test_transforms = transforms.Compose([   
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

class dataset(torch.utils.data.Dataset):

    def __init__(self,file_list,transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self,idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        label = img_path.split('/')[-1].split('.')[0]
        if label == 'dog':
            label=1
        elif label == 'cat':
            label=0
            
        return img_transformed,label
        
train_data = dataset(train_list, transform=train_transforms)
test_data = dataset(test_list, transform=test_transforms)
val_data = dataset(val_list, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=batch_size, shuffle=True)

#print(len(train_data), len(train_loader))
#print(len(val_data), len(val_loader))

#print(train_data[0][0].shape)




classes = ('cat','dog')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))




print(images.shape) #images 224*224

conv1 = nn.Conv2d(3, 6, 5)
x=conv1(images)
print(x.shape)

pool = nn.MaxPool2d(2, 2)
x=pool(x)
print(x.shape)

conv2 = nn.Conv2d(6,16,5)
x=conv2(x)
print(x.shape)

pool = nn.MaxPool2d(2, 2)
x=pool(x)
print(x.shape)