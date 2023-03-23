import torch
from torchvision import transforms
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
if device =='cuda':
    torch.cuda.manual_seed_all(0) 


###########################################################
###########################################################
###########################################################



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
    


def transf():
    unsupervised_transforms = transforms.Compose([   
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_transforms =  transforms.Compose([     
            transforms.Resize((224, 224)), 
            transforms.RandomResizedCrop(224), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
        ])

    val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            #transforms.RandomResizedCrop(224), 
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    test_transforms = transforms.Compose([   
            transforms.Resize((224, 224)),
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    
    return unsupervised_transforms, train_transforms, val_transforms, test_transforms




def imshow(img):
    img = img / 2 + 0.5  
    plt.imshow(np.transpose(img, (1, 2, 0)))




def show_data(train_list):

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


def data_split(train_list,supervised_ratio,val_ratio, test_ratio, random_state):

    # supervised vs unsupervised
    unsupervised_list, supervised_list = train_test_split(train_list, test_size=supervised_ratio, random_state=random_state, shuffle=True)

    # train_val vs test
    train_val_list, test_list = train_test_split(supervised_list, test_size=test_ratio,random_state=random_state, shuffle=True)

    # train vs val
    train_list, val_list = train_test_split(train_val_list, test_size=val_ratio,random_state=random_state, shuffle=True)

    return unsupervised_list, train_list, val_list, test_list