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

unsupervised_list, train_list, val_list, test_list = data_split(train_list,supervised_ratio,val_ratio, test_ratio=0.001, random_state=0) ############# CHANGE TEST RATIO

transform_cats = transforms.Compose([   
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])



unsupervised_data = dataset(test_list, transform=transform_cats) ######## CHANGE TEST LIST TO UNSUPERVISED LIST!!!!!!!!
#print(f'unsupervised_data shape after dataset {unsupervised_data.size()}')
data_loader= torch.utils.data.DataLoader(dataset = unsupervised_data, batch_size=batch_size, shuffle=True )
#print(f'data_loader shape {data_loader.shape()}')
#print(f'len(data_loader): {len(data_loader.dataset)}')

# view images
dataiter = iter(data_loader)
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

    model, outputs = autoen_train(num_epochs, data_loader, model, criterion, optimizer)
    torch.save(model.state_dict(), model_path)



# EVAL

model = Autoencoder()
model.load_state_dict(torch.load(model_path))


for (img, _) in data_loader:
    # img = img.reshape(-1, 28*28) # -> use for Autoencoder_Linear
    
    recon = model(img[0]) # reconstrcuted image
    plt.imshow(  recon.permute(1, 2, 0)  ) 
    
    #print(recon.size())
        #loss = criterion(recon, img) # reconstructed image vs original image and calc mean squaere error
        
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()

    #print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    #outputs.append((epoch, img, recon))









'''
for (img, _) in data_loader:

    # original image
    print(f'img size in dataloader {img[0].size()}')
    plt.imshow(  img[0].permute(1, 2, 0)  ) 
    
    # reconstructed image
    recon = model(img) # reconstrcuted image

#print(f'original img size {images[0].size()}')
#plt.imshow(  images[0].permute(1, 2, 0)  ) 


dataiter = iter(data_loader)
images, labels = dataiter.next()

         

'''







'''
model = Autoencoder().to(device)
model.load_state_dict(torch.load(model_path))



img = outputs[0][1][0]
recon = outputs[0][2][0]


print(f'size original img {img.size()}')
print(f'size reconst. img {recon.size()}')

trans = torchvision.transforms.ToPILImage()
out = trans(img)
out.show()

out_recon = trans(recon)
out_recon.show()


#img_permute = img.permute(1, 2, 0)
#print(f'size original img_permuted {img_permute.size()}')

#plt.imshow(  img.permute(1, 2, 0)  )
#plt.imshow(  recon.permute(1, 2, 0)  )



'''

print('Finished Training')

# %%
