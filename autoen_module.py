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
        # oritinal size: N, 3, 224, 224
        self.encoder = nn.Sequential(
            # input-channels, output-channels, kernelsize
            # strategy: increase input-channels, reduce image size to ultimately 1*1
            nn.Conv2d(3, 16, 5),  
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
        return decoded
    


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
