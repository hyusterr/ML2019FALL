#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import numpy as np
import torch
from PIL import Image
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torch import nn


# In[ ]:


trainX = np.load(sys.argv[1])
trainX = np.transpose(trainX, (0, 3, 1, 2)) / 255. * 2 - 1
trainX = torch.Tensor(trainX)


# In[ ]:


use_gpu = torch.cuda.is_available()
print('I have CUDA: ', use_gpu)


# In[ ]:


# define model

from torch.autograd import Variable
from torchvision import transforms

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # b, 16, 5, 5
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2), # b, 8, 2, 2
#             nn.Conv2d(128, 256, 3, stride=1, padding=1),  # b, 8, 3, 3
#             nn.ReLU(True),
#             nn.MaxPool2d(2) # b, 8, 2, 2
        )
 
        # i think should not add Dropout
#         self.flatten = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128*4*4, 128*2*2),
#             nn.ReLU(),
# #             nn.Dropout()
#         )
        
#         self.reverse_flatten = nn.Sequential(
#             nn.Linear(128*2*2, 128*4*4),
#             nn.ReLU() # output 128*4*4 neuron
#         )
        
        self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 3, stride=1),  # b, 16, 5, 5
#             nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 5, stride=1),  # b, 16, 5, 5
            nn.ReLU(True),
#             nn.BatchNorm2d(64),
            nn.ConvTranspose2d(128, 64, 9, stride=1),  # b, 8, 15, 15
            nn.ReLU(True),
#             nn.BatchNorm2d(32),
            nn.ConvTranspose2d(64, 3, 17, stride=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
#         x1 = self.flatten(x1) # this is a 512-dim vector
#         x  = self.reverse_flatten(x1) # this is a 2048 vector
#         x  = x.view(-1, 128, 4, 4)
        x  = self.decoder(x1)
        return x1, x


model = autoencoder().cuda()

print(model)


# In[ ]:


# load trained models

model.load_state_dict(torch.load('./3+3conv_autoencoder-with3BN-deeperCONV.pth'))


# In[ ]:


# get 1st dim reduction result

model.eval()

X = []

for i in trainX:
    
    vec, img = model(i[None].cuda())
    
    X.append( vec.view(-1).cpu().detach().numpy() )
    

X = np.array( X )


# In[ ]:


from sklearn.decomposition import KernelPCA
# the best is n=200
transformer = KernelPCA(n_components=200, kernel='rbf', n_jobs=-1)
KPCA_X = transformer.fit_transform(X)
# KPCA_X.lambda_


# In[ ]:


from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2).fit_transform(KPCA_X)

X_embedded.shape


# In[ ]:


# clustering
from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(n_clusters=2, random_state=0).fit(pca_X)
print( kmeans.labels_[:10] )

# print( kmeans.cluster_centers_ )


# In[ ]:


# need reverse
lst = [1 if str(i) == '0' else 0 for i in kmeans.labels_]

lst = [str(i) + ',' + str(lst[i]) + '\n' for i in range(len(lst))]

print(lst)
# output

with open(sys.argv[2], 'w') as f:
    f.write('id,label\n')
    f.writelines(lst)
    f.close()

