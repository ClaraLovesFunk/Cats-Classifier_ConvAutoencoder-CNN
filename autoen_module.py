import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from data_module import *


class Autoencoder(nn.Module):
    def __init__(self):

        super().__init__() 

        self.conv1 = nn.Conv2d(3, 16, 8)
        self.conv2 = nn.Conv2d(16,32,8)
        self.de_conv2 = nn.ConvTranspose2d(32, 16, 8)
        self.de_conv1 = nn.ConvTranspose2d(16,3,8)
    

        self.pool1 = nn.MaxPool2d(4, return_indices=True)
        self.pool2 = nn.MaxPool2d(8, return_indices=True)
        self.unpool2= nn.MaxUnpool2d(8)
        self.unpool1= nn.MaxUnpool2d(4)   
                


    def forward(self, x):

        # encoder
        conv1_output= self.conv1(x)
        max1_output, indices1 = self.pool1(conv1_output)

        conv2_output = self.conv2(max1_output)
        max2_output, indices2 = self.pool2(conv2_output)

        # decoder
        unmax2_output = self.unpool2(max2_output, indices2, output_size=conv2_output.size())
        de_conv2_output = self.de_conv2(unmax2_output)

        unmax1_output = self.unpool1(de_conv2_output, indices1, output_size=conv1_output.size())
        de_conv1_output = self.de_conv1(unmax1_output)

        return de_conv1_output
    


def autoen_train(num_epochs, data_loader, model, criterion, optimizer):
    
    outputs = []
    for epoch in range(num_epochs):
        for (img, _) in data_loader:

            recon = model(img) 
            loss = criterion(recon, img) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if (i+1) % int(n_total_steps/3) == 0: 
            #    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}') #where do we store loss over the batches?
        outputs.append((epoch, img, recon))
    
    return model, outputs








'''
class Autoencoder(nn.Module):
    def __init__(self):

        super().__init__() 

        self.conv1 = nn.Conv2d(3, 16, 8)
        self.conv2 = nn.Conv2d(16,32,8)
        self.de_conv2 = nn.ConvTranspose2d(32, 16, 8)
        self.de_conv1 = nn.ConvTranspose2d(16,3,8)
    

        self.pool1 = nn.MaxPool2d(2, stride=2, padding=1, return_indices=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool2= nn.MaxUnpool2d(2, stride=2)
        self.unpool1= nn.MaxUnpool2d(2, stride=2, padding=1)   
                


    def forward(self, x):

        # encoder
        conv1_output= self.conv1(x)
        max1_output, indices1 = self.pool1(conv1_output)

        conv2_output = self.conv2(max1_output)
        max2_output, indices2 = self.pool2(conv2_output)

        # decoder
        unmax2_output = self.unpool2(max2_output, indices2, output_size=conv2_output.size())
        de_conv2_output = self.de_conv2(unmax2_output)

        unmax1_output = self.unpool1(de_conv2_output, indices1, output_size=conv1_output.size())
        de_conv1_output = self.de_conv1(unmax1_output)

        return de_conv1_output
    '''