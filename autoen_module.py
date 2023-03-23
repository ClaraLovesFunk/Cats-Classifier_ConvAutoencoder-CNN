import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from data_module import *
from autoen_module import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
if device =='cuda':
    torch.cuda.manual_seed_all(0) 


###########################################################
###########################################################
###########################################################


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



class Autoencoder_Clf_Head(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(in_features=800, out_features=128)
        self.fc2 = nn.Linear(128, 2) 
    
    def forward(self, x):
        
        x = x.view(-1, 800)

        x = F.relu(self.fc1(x))               
        x = self.fc2(x)                        
  
        return x
    


def autoen_train(num_epochs, data_loader, model, criterion, optimizer):
    
    outputs = []
    for epoch in range(num_epochs):
        for (img, _) in data_loader:

            img.to(device)
            recon = model(img) 
            loss = criterion(recon, img) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}') 
        outputs.append((epoch, img, recon))
    
    return model, outputs



def train_autoen_clf_head(num_epochs, train_loader, autoen, autoen_clf_head, criterion, optimizer, val_loader):

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            autoen = autoen.to(device)
            
            # encoder steps
            conv1_output= autoen.conv1(data)
            max1_output, indices1 = autoen.pool1(conv1_output)

            conv2_output = autoen.conv2(max1_output)
            max2_output, indices2 = autoen.pool2(conv2_output)

            # classification head steps
            outputs = autoen_clf_head(max2_output)

            loss = criterion(outputs, label)            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = ((outputs.argmax(dim=1) == label).float().mean())
            epoch_accuracy += acc/len(train_loader)
            epoch_loss += loss/len(train_loader)
            
        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
        
        with torch.no_grad():
            epoch_val_accuracy=0
            epoch_val_loss =0
            for data, label in val_loader:
                data = data.to(device)
                label = label.to(device)
                
                # encoder steps
                conv1_output= autoen.conv1(data)
                max1_output, indices1 = autoen.pool1(conv1_output)

                conv2_output = autoen.conv2(max1_output)
                max2_output, indices2 = autoen.pool2(conv2_output)

                # classification head steps
                val_outputs = autoen_clf_head(max2_output)

                #val_output = model(data)
                val_loss = criterion(val_outputs,label)
                
                
                acc = ((val_outputs.argmax(dim=1) == label).float().mean())
                epoch_val_accuracy += acc/ len(val_outputs)
                epoch_val_loss += val_loss/ len(val_outputs)
                
            print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))

    return autoen_clf_head



def autoen_test(data_loader, model, criterion):

    model.to(device)
    with torch.no_grad():

        recon_all = []
        img_all = []

        for img_batch, _ in data_loader:

            img_batch = img_batch.to(device)
            recon_batch = model(img_batch)

            recon_all.append(recon_batch.cpu())
            img_all.append(img_batch.cpu())

        recon_all = torch.cat(recon_all)
        img_all = torch.cat(img_all) 

        test_loss = criterion(recon_all, img_all)
        print(f'test loss: {test_loss}')

    return test_loss.data



def test_autoen_clf(test_loader, autoen, autoen_clf_head, criterion):
        
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
            test_outputs = autoen_clf_head(max2_output)


            val_loss = criterion(test_outputs,label)
            acc = ((test_outputs.argmax(dim=1) == label).float().mean())
            test_accuracy += acc/ len(test_loader)
            test_loss += val_loss/ len(test_loader)
            
        print('test_accuracy : {}, test_loss : {}'.format(test_accuracy,test_loss))

    return test_accuracy, test_loss



def viz_autoen(test_loader, model, classes):

    dataiter = iter(test_loader) 
    img, labels = dataiter.next()

    recon = model(img)

    imgs = img.detach().numpy()
    recon = recon.detach().numpy()

    # plot original imgs
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
    for idx in np.arange(5):
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        imshow(imgs[idx])
        ax.set_title(classes[labels[idx]])

    # plot reconstr. imgs
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
    for idx in np.arange(5):
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        imshow(recon[idx])
        ax.set_title(classes[labels[idx]])

    plt.show() 
