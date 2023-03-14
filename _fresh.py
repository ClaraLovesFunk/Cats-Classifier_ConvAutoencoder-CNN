
##%%

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
num_epochs = 0 ######5
batch_size = 4
learning_rate = 0.1 ###0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1234)
if device =='cuda':
    torch.cuda.manual_seed_all(1234)


train_dir = 'data/train'
test_dir = 'data/test'

print(os.listdir(train_dir)[:5])

train_list = glob.glob(os.path.join(train_dir,'*.jpg')) ############
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

print(len(train_list))


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

print(len(train_data), len(train_loader))
print(len(val_data), len(val_loader))

print(train_data[0][0].shape)




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


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(in_features=16*53*53, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        #print(x.shape)
        x = x.view(-1, 16*53*53)       #            # to flatten the output of conv2, the -1 will give us our batch_size
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss() #####FOR MULTICLASS, softmax already included in loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)







#########################################################################
#########################################################################
# THE REST

#########################################################################
#########################################################################

# training loop of batch optimization 
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad() # empty the gradients
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

'''# wrap evaluation
with torch.no_grad(): # disabling gradient calculation
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device) # to get GPU support
        #labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')'''

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
        
    print('val_accuracy : {}, val_loss : {}'.format(epoch_val_accuracy,epoch_val_loss))