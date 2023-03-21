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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
if device =='cuda':
    torch.cuda.manual_seed_all(0) 



def eval_cats_clf(val_loader, model, criterion):

    with torch.no_grad():
        accuracy=0
        loss =0
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)
            
            val_output = model(data)
            val_loss = criterion(val_output,label)
            
            
            acc = ((val_output.argmax(dim=1) == label).float().mean()) #########
            accuracy += acc/ len(val_loader)
            loss += val_loss/ len(val_loader)
            
        print('val_accuracy : {}, val_loss : {}'.format(accuracy,loss))

    return accuracy, loss



def imshow(img):
    img = img / 2 + 0.5  
    plt.imshow(np.transpose(img, (1, 2, 0))) 



def viz_autoen_output(test_loader, model, classes,output):

    #Batch of test images
    #dataiter = iter(test_loader)
    #images, labels = dataiter.next()
    #print(labels[0])

    #Sample outputs
    #output = output[0][2]#model(images)
    #images = output[0][1]#images.numpy()

    #output = output.view(batch_size, 3, 32, 32)
    images_recon = images_recon.numpy()
    images = images.numpy()

    #images_recon = images_recon.detach().numpy()
    #images = images.detach().numpy()
    #print(images[0])
    #print(output[0])

    #Original Images
    print("Original Images")
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
    for idx in np.arange(5): 
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        #ax.set_title(classes[labels[idx]])
    plt.show()

    #Reconstructed Images
    print('Reconstructed Images')
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
    for idx in np.arange(5):
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        imshow(images_recon[idx])
        #ax.set_title(classes[labels[idx]])
    plt.show() 
    print('Finished Training')