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
from PIL import Image
from data_module import *
from cnn_module import *
from eval_viz_module import *


input = Variable(torch.rand(1,1,64,64))
pool1 = nn.MaxPool2d(2, stride=2, padding=1, return_indices=True)
pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool1= nn.MaxUnpool2d(2, stride=2)
unpool2= nn.MaxUnpool2d(2, stride=2, padding=1)

output1, indices1 = pool1(input)
output2, indices2 = pool2(output1)

output3 = unpool1(output2,indices2)
output4 = unpool2(output3, indices1)