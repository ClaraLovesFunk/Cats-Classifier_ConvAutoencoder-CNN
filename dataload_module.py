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
    


def transf():# data transformation                             
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
    
    return train_transforms, val_transforms, test_transforms


def show_data(train_list):

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


def data_split(train_list,supervised_ratio,val_ratio, test_ratio):
    
    # supervised vs unsupervised
    unsupervised_list, supervised_list = train_test_split(train_list, test_size=supervised_ratio)

    # train_val vs test
    train_val_list, test_list = train_test_split(supervised_list, test_size=test_ratio)

    # train vs val
    train_list, val_list = train_test_split(train_val_list, test_size=val_ratio)

    return unsupervised_list, train_list, val_list, test_list


supervised_ratio = 0.2
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1