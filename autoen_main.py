#%%

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from data_module import *
from autoen_module import *



device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
if device =='cuda':
    torch.cuda.manual_seed_all(0) 


###########################################################
###########################################################
###########################################################



# FLAGS

train_flag = True
test_flag = True



# HYPS & PARAMETERS

num_epochs = 1 ######5
batch_size=64
learning_rate = 1e-3 ##########
weight_decay=1e-5

supervised_ratio = 0.2
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

train_dir = 'data/train'
model_path = 'model/autoencoder.pth'
results_path = 'results/results_autoencoder.npy'

train_list = glob.glob(os.path.join(train_dir,'*.jpg')) 

classes = ('cat','dog')


# LOAD AND SPLIT DATA

unsupervised_list, train_list, val_list, test_list = data_split(train_list,supervised_ratio,val_ratio, test_ratio=0.8, random_state=0) ############# CHANGE TEST RATIO

transform_cats = transforms.Compose([   
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])



unsupervised_data = dataset(test_list, transform=transform_cats) ######## CHANGE TEST LIST TO UNSUPERVISED LIST!!!!!!!!

#print(f'unsupervised_data shape after dataset {unsupervised_data.size()}')
train_loader = torch.utils.data.DataLoader(dataset = unsupervised_data, batch_size=batch_size, shuffle=True )
test_loader = torch.utils.data.DataLoader(dataset = unsupervised_data, batch_size=batch_size, shuffle=True )


#print(f'train_loader shape {train_loader.shape()}')
#print(f'len(train_loader): {len(train_loader.dataset)}')

# view images
dataiter = iter(train_loader)
images, labels = dataiter.next()
#print(f'range of values of image tensor: {torch.min(images)}, {torch.max(images)}') # based on original image values that were put in tensor and all the stuff like cropping, flipping...



    
# DEFINE MODEL

model = Autoencoder()



# TRAIN

if train_flag == True:

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.9,    ####################### 1e-3          
                                weight_decay=1e-5)

    model, outputs = autoen_train(num_epochs, train_loader, model, criterion, optimizer)
    torch.save(model.state_dict(), model_path)



# EVAL & VIZ

model = Autoencoder()
model.load_state_dict(torch.load(model_path))

#Batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
print(labels[0])

#Sample outputs
output = model(images)
images = images.numpy()

#output = output.view(batch_size, 3, 32, 32)
output = output.detach().numpy()

def imshow(img):
    img = img / 2 + 0.5  
    plt.imshow(np.transpose(img, (1, 2, 0))) 

#Original Images
print("Original Images")
fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
for idx in np.arange(5):
    ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])
plt.show()


#Reconstructed Images
print('Reconstructed Images')
fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
for idx in np.arange(5):
    ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
    imshow(output[idx])
    ax.set_title(classes[labels[idx]])
plt.show() 
print('Finished Training')

# %%
