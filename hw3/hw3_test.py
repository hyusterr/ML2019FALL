#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import torch
import torchvision
import os
import random
import numpy as np
import pandas as pd
import glob
import torch.nn as nn
from torch.optim import Adam
# pillow; python image library
from PIL import Image
from torch.utils.data import Dataset, DataLoader
# there a lot of pre-train models and datasets in torchvision
import torchvision.models as models
from torchvision import transforms

# In[2]:

use_gpu = torch.cuda.is_available() 

# In[19]:


class refAlexNet(nn.Module):
    
    # all model needs to super the torch.nn.Module
    def __init__(self):
        
        super(refAlexNet, self).__init__()
        
        # define layers as model class's attributes
        # nn.Conv2d(in_channel, out_channel)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),     
        )
        
        # batch normalization is a new trick
        # tensor x > convolution > activation function > dropout/BN/maxpool > output tensor
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),            
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),    
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            
        )
        
        self.adapool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        
        
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(3*3*256, 2*2*256),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    # define the feed forward flow as a method: the function eat model and input x, return the final tensor
    def forward(self, x):
        #image size (48,48)
        x = self.conv1(x) #(24,24)
        x = self.conv2(x)#(12,12)
        x = self.conv3(x) #(6,6)
        x = self.conv4(x) #(3,3)
        x = x.view(-1, 3*3*256)
#         print(x.shape) 
        x = self.fc(x)
        return x
    
# RuntimeError: size mismatch, m1: [256 x 1152], m2: [4096 x 1024]


# In[8]:


model = refAlexNet()

print(model)


# In[21]:


model.load_state_dict(torch.load('./refAlexNet_model_epoch300_300.pth'))
model.cuda()
model.eval()

# In[33]:



# In[38]:


transform = transforms.Compose([
    #transforms.RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10, fillcolor=0),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize([mean], [std], inplace=False)
    ])
    

pred_list = []

def load_test_data( img_path ):
    
    '''
    this function is simply to get all (image-filename, label) tuples 
    '''
    
    
    # build image filename list, and make sure they are sorted
    # glob is a module that support linux shell regex filename usage
    # os.path.join >>> './ml2019fall-hw3-private/train_img/*.jpg'
    # glob will expand it
    test_image = sorted( glob.glob(os.path.join(img_path, '*.jpg' ) ) )

    return test_image


class hw3_test_dataset(Dataset):
    
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        # since Resnet18 expected to eat RGB image, we need to convert it
        img = Image.open(self.data[idx])
        img = self.transform(img)
        
        return img


testing_set = load_test_data(sys.argv[1])
test_dataset = hw3_test_dataset(testing_set, transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


for idx, img in enumerate(test_loader):
    if use_gpu:
        img = img.cuda()
        
    output = model(img)
    predict = torch.max(output, 1)[1]
    
    pred_list.append(predict)
    

lst = [str(i) + ',' + str(int(pred_list[i])) + '\n' for i in range(len(pred_list))]

with open(sys.argv[2], 'w') as f:
    f.write('id,label\n')
    f.writelines(lst)
    f.close()
