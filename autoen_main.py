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

train_flag = False
test_flag = True



# HYPS & PARAMETERS

num_epochs = 1 ######5
batch_size=64
learning_rate = 0.5 ##########1e-3
weight_decay=1e-5

supervised_ratio = 0.2
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

train_dir = 'data/train'
model_path = 'model/autoencoder.pth'
results_path = 'results/results_autoencoder.npy'

train_list = glob.glob(os.path.join(train_dir,'*.jpg')) 



# LOAD AND SPLIT DATA

unsupervised_list, train_list, val_list, test_list = data_split(train_list,supervised_ratio,val_ratio, test_ratio=0.01, random_state=0) ############# CHANGE TEST RATIO

transform_cats = transforms.Compose([   
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

unsupervised_data = dataset(test_list, transform=transform_cats) ######## CHANGE TEST LIST TO UNSUPERVISED LIST!!!!!!!!
data_loader= torch.utils.data.DataLoader(dataset = unsupervised_data, batch_size=batch_size, shuffle=True )
print(f'len(data_loader): {len(data_loader.dataset)}')

# view images
dataiter = iter(data_loader)
images, labels = dataiter.next()
print(f'range of values of image tensor: {torch.min(images)}, {torch.max(images)}') # based on original image values that were put in tensor and all the stuff like cropping, flipping...


    
# DEFINE MODEL

model = Autoencoder()



# TRAIN

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.9,    ####################### 1e-3          
                             weight_decay=1e-5)

model, outputs = autoen_train(num_epochs, data_loader, model, criterion, optimizer)
torch.save(model.state_dict(), model_path)



# EVAL

#def plot_reconstructed_img(num_epochs, plot_every_i_epoch):

for k in range(0, num_epochs, 4): 
    plt.figure(figsize=(5, 2))
    #plt.gray()
    imgs = outputs[k][1].detach().numpy() # transforms it from tensor to np array
    lum_img = imgs[:, :, 0]

    recon = outputs[k][2].detach().numpy()
    lum_recon = recon[:, :, 0]

    for i, item in enumerate(lum_img):
        if i >= 5: break 
        #plt.subplot(2, 5, i+1)
        #plt.imshow(item[0])
        plt.subplot(2, 5, i+1)
        plt.imshow(item[0])
            
    for i, item in enumerate(lum_recon):
        if i >= 9: break
        plt.subplot(2, 5, 5+i+1) 
        plt.imshow(item[0])
        fig = plt.figure()

    #plt.imshow(  outputs[0][2].permute(1, 2, 0)  )
    #plt.show()

    #lum_img = imgs[:, :, 0]
    #plt.imshow(lum_img)

print('Finished Training')

# %%
