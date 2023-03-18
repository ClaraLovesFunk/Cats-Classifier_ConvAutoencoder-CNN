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
from eval_viz_module import *



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
learning_rate = 0.5 #########1e-3 
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

unsupervised_list, train_list, val_list, test_list = data_split(train_list, supervised_ratio,val_ratio, test_ratio, random_state=0) 

transform_cats = transforms.Compose([   
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])


train_set = dataset(unsupervised_list, transform=transform_cats) 
test_set = dataset(test_list, transform=transform_cats) 

train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size=batch_size, shuffle=True )
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size=batch_size, shuffle=True )

# check range of values in image tensor
dataiter = iter(train_loader)
images_train, labels_train = dataiter.next()
#print(f'range of values of image tensor: {torch.min(images)}, {torch.max(images)}') # based on original image values that were put in tensor and all the stuff like cropping, flipping...



    
# DEFINE MODEL

model_untrained = Autoencoder()



# TRAIN

if train_flag == True:

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_untrained.parameters(),
                                lr=learning_rate,             
                                weight_decay=weight_decay)

    model, output = autoen_train(num_epochs, train_loader, model_untrained, criterion, optimizer)
    torch.save(model.state_dict(), model_path)



# EVAL & VIZ
if test_flag == True:

    model = Autoencoder()
    model.load_state_dict(torch.load(model_path))

    viz_autoen_output(train_loader, model, classes)




'''model = Autoencoder()
model.load_state_dict(torch.load(model_path))

dataiter = iter(train_loader) # SHOULD BE TEST LOADER
images_test, labels_test = dataiter.next()

outputs = model(images_test)

for k in range(0, num_epochs, 4): 
    plt.figure(figsize=(9, 2))
    plt.gray()

    imgs = images_test.detach().numpy() # transforms it from tensor to np array
    recon = outputs.detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9: break # plot first 9 images
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

print('Finished Training')'''



# %%
