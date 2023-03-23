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

train_flag = True
test_flag = True



# HYPS & PARAMETERS

num_epochs = 1 ######5
batch_size = 32 # 64     devide sample in smaller batches
learning_rate = 1e-3 #########1e-3 
weight_decay=1e-5

supervised_ratio = 0.2# 0.99 ######0.9996 #0.2
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
    transforms.ToTensor(),
    #transforms.Normalize((0.5), (0.5))
])


train_set = dataset(unsupervised_list, transform=transform_cats) 
test_set = dataset(test_list, transform=transform_cats) 

train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size=batch_size, shuffle=True )
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size=batch_size, shuffle=True )

print(f'dataset size: {len(train_set)}') ##########

# check range of values in image tensor
#dataiter = iter(train_loader)
#images_train, labels_train = dataiter.next()
#print(images_train, labels_train)
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

    #for (img, _) in test_loader: # iterating over the batches in dataloader
    dataiter = iter(test_loader) # same but for one batch?
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
    print('Finished Training')

# %%