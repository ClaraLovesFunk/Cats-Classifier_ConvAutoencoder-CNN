
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



def train_PCA_clf_head(num_epochs, train_loader, model_PCA, PCA_clf_head, criterion, optimizer, val_loader):

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        
        for data, label in train_loader:
            #data = data.to(device)
            #label = label.to(device)
            #model_PCA = model_PCA.to(device)
            #PCA_clf_head = PCA_clf_head.to(device)

            # prepare data for PCA
            data = torch.flatten(data, start_dim=2, end_dim=-1) # flatten only img, not over samples
            data = data.numpy() # to numpy
            data = data.astype(np.uint8) # ?
            data = data.mean(axis=1) # transform to greyscales by taking mean of r,g,b values
            
            # compress data & turn to tensor
            data_compressed= model_PCA.transform(data)
            data_compressed = torch.from_numpy(data_compressed)

            # classification head
            outputs = PCA_clf_head(data_compressed.float())

            loss = criterion(outputs, label)            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = ((outputs.argmax(dim=1) == label).float().mean())
            epoch_accuracy += acc/len(train_loader)
            epoch_loss += loss/len(train_loader)
            
        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
        
        with torch.no_grad():

            epoch_loss = 0
            epoch_accuracy = 0
            
            for data, label in val_loader:
                #data = data.to(device)
                #label = label.to(device)
                #model_PCA = model_PCA.to(device)
                #PCA_clf_head = PCA_clf_head.to(device)

                # prepare data for PCA
                data = torch.flatten(data, start_dim=2, end_dim=-1) # flatten only img, not over samples
                data = data.numpy() # to numpy
                data = data.astype(np.uint8) # ?
                data = data.mean(axis=1) # transform to greyscales by taking mean of r,g,b values
                
                # compress data & turn to tensor
                data_compressed= model_PCA.transform(data)
                data_compressed = torch.from_numpy(data_compressed)

                # classification head
                outputs = PCA_clf_head(data_compressed.float())

                loss = criterion(outputs, label)            
                #optimizer.zero_grad()
                #loss.backward()
                #optimizer.step()
                
                acc = ((outputs.argmax(dim=1) == label).float().mean())
                epoch_accuracy += acc/len(val_loader)
                epoch_loss += loss/len(val_loader)
                
            print('Epoch : {}, val accuracy : {}, val loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))

    return PCA_clf_head




def test_PCA_clf(test_loader, model_PCA, autoen_clf_head, criterion):
        
    with torch.no_grad():

        test_accuracy=0
        test_loss =0
        for data, label in test_loader:
            #data = data.to(device)
            #label = label.to(device)
            #model_PCA = model_PCA.to(device)
            #autoen_clf_head = autoen_clf_head.to(device)

            # prepare data for PCA
            data = torch.flatten(data, start_dim=2, end_dim=-1) # flatten only img, not over samples
            data = data.numpy() # to numpy
            data = data.astype(np.uint8) # ?
            data = data.mean(axis=1) # transform to greyscales by taking mean of r,g,b values
            
            # compress data & turn to tensor
            data_compressed= model_PCA.transform(data)
            data_compressed = torch.from_numpy(data_compressed)

            # classification head
            test_outputs = autoen_clf_head(data_compressed.float())


            val_loss = criterion(test_outputs,label)
            acc = ((test_outputs.argmax(dim=1) == label).float().mean())
            test_accuracy += acc/ len(test_loader)
            test_loss += val_loss/ len(test_loader)
            
        print('test_accuracy : {}, test_loss : {}'.format(test_accuracy,test_loss))

    return test_accuracy, test_loss