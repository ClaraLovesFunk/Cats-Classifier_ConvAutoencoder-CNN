##%%

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
if device =='cuda':
    torch.cuda.manual_seed_all(0) 


###########################################################
###########################################################
###########################################################

'''transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))
    ])

transform = transforms.ToTensor() #turn images to tensor'''
# FLAGS

train_flag = True
test_flag = True



# HYPS & PARAMETERS

#num_epochs = 0 ######5
batch_size=64
#learning_rate = 0.1 ###0.001

supervised_ratio = 0.2
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

#classes = ('cat','dog')

train_dir = 'data/train'
#test_dir = 'data/test'
model_path = 'model/autoencoder.pth'
results_path = 'results/results_autoencoder.npy'

train_list = glob.glob(os.path.join(train_dir,'*.jpg')) 
#test_list = glob.glob(os.path.join(test_dir, '*.jpg'))



# INSPECT DATA

show_data(train_list)
# Class frequencies



# LOAD AND SPLIT DATA
#dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

unsupervised_list, train_list, val_list, test_list = data_split(train_list,supervised_ratio,val_ratio, test_ratio, random_state=0)

transform_cats = transforms.Compose([   
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

unsupervised_data = dataset(unsupervised_list, transform=transform_cats)
#train_data = dataset(train_list, transform=train_transforms)
#test_data = dataset(test_list, transform=test_transforms)
#val_data = dataset(val_list, transform=test_transforms)

data_loader= torch.utils.data.DataLoader(dataset = unsupervised_data, batch_size=batch_size, shuffle=True )
#train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
#test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
#val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=batch_size, shuffle=True)





'''
# TAKE ONLY PARTIAL DATA
# Split the indices in a stratified way
indices = np.arange(len(dataset))
train_indices, test_indices = train_test_split(indices, train_size=100*10, stratify=dataset.targets)

# Warp into Subsets and DataLoaders
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)








data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=64, #### ???????
                                          shuffle=True)'''

print(len(data_loader.dataset))

# view images
dataiter = iter(data_loader)
images, labels = dataiter.next()
print(f'range of values of image tensor: {torch.min(images)}, {torch.max(images)}') # based on original image values that were put in tensor and based on normalization






'''
# repeatedly reduce the size
class Autoencoder_Linear(nn.Module):
    def __init__(self):
        super().__init__()        
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), # number of pixels, outputsize (here it reduces the size from 784 to 128) #####(N, 784) -> (N, 128)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3) # -> N, 3
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid() ###### activation function that puts the values in the range 0-1 (bc thats the min and max values for the tensors we tested earlier)
        )                ##### we min max values of our input data would be -1,1, we need tanh

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# Input [-1, +1] -> use nn.Tanh
'''


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()        
        # N, 1, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # -> N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7) # -> N, 64, 1, 1
        )
        
        # N , 64, 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7), # -> N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # N, 16, 14, 14 (N,16,13,13 without output_padding)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # N, 1, 28, 28  (N,1,27,27)
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
 
# Note: nn.MaxPool2d -> use nn.MaxUnpool2d, or use different kernelsize, stride etc to compensate...
# Input [-1, +1] -> use nn.Tanh





model = Autoencoder()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.5,              #######1e-3
                             weight_decay=1e-5)




# Point to training loop video
num_epochs = 1 #######??????
outputs = []
for epoch in range(num_epochs):
    for (img, _) in data_loader:
        # img = img.reshape(-1, 28*28) # -> use for Autoencoder_Linear
        recon = model(img)
        loss = criterion(recon, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    outputs.append((epoch, img, recon))



for k in range(0, num_epochs, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])
            
    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1) # row_length + i + 1
        # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])

print('Finished Training')
PATH = 'model/autoencoder_cats.pth'
torch.save(model.state_dict(), PATH)