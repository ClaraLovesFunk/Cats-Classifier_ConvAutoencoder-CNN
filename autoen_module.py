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

# same dims as cnn
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
    


def autoen_train(num_epochs, data_loader, model, criterion, optimizer):
    
    outputs = []
    for epoch in range(num_epochs):
        for (img, _) in data_loader:
            # img = img.reshape(-1, 28*28) # -> use for Autoencoder_Linear

            recon = model(img) # reconstrcuted image
            loss = criterion(recon, img) # reconstructed image vs original image and calc mean squaere error
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            

        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        outputs.append((epoch, img, recon))
    
    return model, outputs



'''
# successfully trained autoen with wrong dims
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()       
        # oritinal size: N, 3, 224, 224
        self.encoder = nn.Sequential(
            # input-channels, output-channels, kernelsize
            # strategy: increase input-channels, reduce image size to ultimately 1*1
            nn.Conv2d(3, 16, 5), # stride =1, padding =0 
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),  
            nn.ReLU(),
            nn.Conv2d(32, 64, 5) 
            # why dont we activate the last thing?
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 5), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 5), 
            nn.Sigmoid() # Why???????
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded'''