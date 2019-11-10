#!/usr/bin/env python
# coding: utf-8



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


print('I have CUDA: ', torch.cuda.is_available() )


# In[3]:


def load_data( img_path, label_path ):
    
    '''
    this function is simply to get all (image-filename, label) tuples 
    '''
    
    
    # build image filename list, and make sure they are sorted
    # glob is a module that support linux shell regex filename usage
    # os.path.join >>> './ml2019fall-hw3-private/train_img/*.jpg'
    # glob will expand it
    train_image = sorted( glob.glob(os.path.join(img_path, '*.jpg' ) ) )
    
    # build labels as a list
    # remember they are sorted
    train_label = pd.read_csv( label_path ).iloc[:,1].values.tolist()
    
    # a list of tuple: [(image-filename, label)]
    train_data = list( zip( train_image, train_label ) )
    
    
    random.shuffle( train_data )
    
    train_set = train_data[:26000]
    valid_set = train_data[26000:]
    
    return train_set, valid_set


# In[4]:


print('All training data available:', len( pd.read_csv( sys.argv[2] ).iloc[:,1].values.tolist() ) )


# In[5]:


# need to custom dataset in every pytorch task
# they basically need to contain:
# __init__
# __len__
# __getitem__

class hw3_dataset( Dataset ): # this is inherence
    
    def __init__( self, data, transform ):
        
        # define attributes
        self.data = data
        # transfrom may use pytorch's transforms that help us reshape image and transform it into tensors
        # we store the operation in attribute, make it easilier to call
        self.transform = transform
        
    # define what len(this-class) will return    
    def __len__(self):
        return len( self.data )
    
    # this is a fixed form
    # make it possible for user to get specific data by calling index
    def __getitem__(self, idx):
        
        # remember they are tuples 
        img = Image.open( self.data[idx][0] )
        img = self.transform( img )
        label = self.data[idx][1]
        
        return img, label


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
        x = self.fc(x)
        print(x.size())
        return x


# In[8]:


model = refAlexNet()

print(model)


# In[21]:


# model.load_state_dict(torch.load('./refAlexNet_model_epoch300_300.pth'))


# In[7]:


if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()

    # get data 
    train_set, valid_set = load_data(sys.argv[1], sys.argv[2])

    #transform to tensor, data augmentation
    
    transform = transforms.Compose([
    #transforms.RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10, fillcolor=0),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize([mean], [std], inplace=False)
    ])
    
    # read data and preprocess
    
    train_dataset = hw3_dataset(train_set,transform)
    train_loader = DataLoader(train_dataset, batch_size=72, shuffle=True)

    valid_dataset = hw3_dataset(valid_set,transform)
    valid_loader = DataLoader(valid_dataset, batch_size=72, shuffle=False)

    model = refAlexNet()
    if use_gpu:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    num_epoch = 300
    
    # record every epoch's performance
    
    train_epoch_acc  = []
    train_epoch_loss = []
    val_epoch_acc    = []
    val_epoch_loss   = []
    
    print('start training')
    
    for epoch in range(num_epoch):
        
        print( epoch + 1, 'starting' )
        
        model.train()
        
        train_loss = []
        train_acc = []
        
        for idx, (img, label) in enumerate(train_loader):
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            
            # backpropgate 
            loss.backward()
            optimizer.step()

            predict = torch.max(output, 1)[1]
            acc = np.mean((label == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())

        train_epoch_loss.append( (epoch + 1, np.mean(train_loss)) )
        train_epoch_acc.append( (epoch + 1, np.mean(train_acc)) )
        
        print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))


        # nn.module subclasses have two mode: train and eval, that is convenient when containing dropout, BN, etc.
        model.eval()
        with torch.no_grad():
            valid_loss = []
            valid_acc = []
            for idx, (img, label) in enumerate(valid_loader):
                if use_gpu:
                    img = img.cuda()
                    label = label.cuda()
                output = model(img)
                loss = loss_fn(output, label)
                predict = torch.max(output, 1)[1]
                acc = np.mean((label == predict).cpu().numpy())
                valid_loss.append(loss.item())
                valid_acc.append(acc)
                
            val_epoch_acc.append( ( epoch + 1, np.mean(np.mean(valid_acc)) ) )
            val_epoch_loss.append( ( epoch + 1, np.mean(np.mean(valid_loss)) ) )
            
            print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, np.mean(valid_loss), np.mean(valid_acc)))
        
        if np.mean(train_acc) > 0.95:
            checkpoint_path = 'refAlexNet_model_epoch300_{}.pth'.format(epoch + 1) 
            torch.save(model.state_dict(), checkpoint_path)
            print('model saved to %s' % checkpoint_path)
