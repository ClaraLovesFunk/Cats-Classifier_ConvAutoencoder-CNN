import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.pool1 = nn.MaxPool2d(4, 4)
        self.pool2 = nn.MaxPool2d(8, 8)

        self.fc1 = nn.Linear(in_features=800, out_features=128)
        self.fc2 = nn.Linear(128, 2) 
    
    def forward(self, x):
        
        x = self.pool1(F.relu(self.conv1(x)))  
        x = self.pool2(F.relu(self.conv2(x)))  

        x = x.view(-1, 800)

        x = F.relu(self.fc1(x))               
        x = self.fc2(x) 

        return x




def train_cnn(num_epochs, train_loader, model, criterion, optimizer, val_loader):

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            
            output = model(data)
            loss = criterion(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = ((output.argmax(dim=1) == label).float().mean())
            epoch_accuracy += acc/len(train_loader)
            epoch_loss += loss/len(train_loader)
            
        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
        
        with torch.no_grad():
            epoch_val_accuracy=0
            epoch_val_loss =0
            for data, label in val_loader:
                data = data.to(device)
                label = label.to(device)
                
                val_output = model(data)
                val_loss = criterion(val_output,label)
                
                
                acc = ((val_output.argmax(dim=1) == label).float().mean())
                epoch_val_accuracy += acc/ len(val_loader)
                epoch_val_loss += val_loss/ len(val_loader)
                
            print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))

    return model



def test_cnn(test_loader, model, criterion):

    with torch.no_grad():
            
        test_accuracy=0
        test_loss =0

        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            
            test_output = model(data)
            test_loss = criterion(test_output,label)
            
            acc = ((test_output.argmax(dim=1) == label).float().mean())
            test_accuracy += acc/ len(test_loader)
            test_loss += test_loss/ len(test_loader)
            
        print('test_accuracy : {}, test_loss : {}'.format(test_accuracy,test_loss))

    return test_accuracy, test_loss


