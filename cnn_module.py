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


###########################################################
###########################################################
###########################################################




class cnn_cats(nn.Module):


    def __init__(self):

        super(cnn_cats, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 8)
        self.conv2 = nn.Conv2d(16,32,8)
        #self.conv3 = nn.Conv2d(32,64,8)

        self.pool1 = nn.MaxPool2d(4, 4)
        self.pool2 = nn.MaxPool2d(8, 8)

        self.fc1 = nn.Linear(in_features=800, out_features=128)
        self.fc2 = nn.Linear(128, 2) #####self.fc2 = nn.Linear(128, 64)
        #self.fc3 = nn.Linear(64, 2)


    def forward(self, x):
        
        x = self.pool1(F.relu(self.conv1(x)))  
        x = self.pool2(F.relu(self.conv2(x)))  
        #x = F.relu(self.conv3(x))

        x = x.view(-1, 800)

        x = F.relu(self.fc1(x))               
        x = self.fc2(x)            #######x = F.relu(self.fc2(x))               
        #x = self.fc3(x)  

        return x
    



def cnn_cats_train(train_loader, num_epochs, model, criterion, optimizer):

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # origin shape: 
            # input_layer: 
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad() # empty the gradients
            loss.backward()
            optimizer.step()

            if (i+1) % int(n_total_steps/3) == 0: ############ 10 mal pro epoche soll geprinted werden
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    print('Finished Training')
    
    return model