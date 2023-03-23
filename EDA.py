#%%

import torch
from torchvision import transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import glob
from data_module import *
from cnn_module import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
if device =='cuda':
    torch.cuda.manual_seed_all(0) 


###########################################################
###########################################################
###########################################################



# HYPS & PARAMETERS

num_epochs = 5 
batch_size = 32
learning_rate = 0.001 

supervised_ratio = 0.2
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

classes = ('cat','dog')

train_dir = 'data/train'
test_dir = 'data/test'
model_path = 'model/cnn.pth'
results_path = 'results/results_cnn.npy'

train_list = glob.glob(os.path.join(train_dir,'*.jpg')) 

transform_cats = transforms.Compose([   
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

data_set = dataset(train_list,transform = transform_cats)
data_loader = torch.utils.data.DataLoader(dataset = data_set, batch_size=batch_size, shuffle=True )


# SHOW DATA

 

show_data(train_list)



# CLASS SIZES

train_classes = [label for _, label in data_set]
print(Counter(train_classes))

  
data = {'Cats':12500, 'Dogs':12500} #{'Cats':Counter(train_classes)[0], 'Dogs':Counter(train_classes)[1]}
classes = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
plt.bar(classes, values, color ='maroon',
        width = 0.4)
 
plt.xlabel("Classes")
plt.ylabel("No. of Instances")
plt.title("Class Sizes")
plt.show()



# %%
