#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import numpy as np
import pandas as pd
import spacy
import torch
from torch import nn
import random
from gensim.models import Word2Vec


# In[6]:


# there are nlp.pipe in spacy, including tokenizer, parser, ner, postagger...
nlp = spacy.load("en_core_web_sm")


# In[9]:


EMBEDDING_DIM = 200
# there are paras can adjust for stopwords, based on frequency
W2Vmodel = Word2Vec.load("200-dim-word2vec.model")

w2v = []
for _, key in enumerate(W2Vmodel.wv.vocab):
    w2v.append((key, W2Vmodel.wv[key]))
special_tokens = ["<PAD>", "<UNK>"]
for token in special_tokens:
    w2v.append((token, [0.0] * EMBEDDING_DIM))


# In[5]:


w2v_dict = { i[0]: i[1] for i in w2v }


# In[34]:


def load_data( xs, ys ):
    
    '''
    this function is simply to get all (image-filename, label) tuples 
    '''
    
    # a list of tuple: [(image-filename, label)]
    train_data = list( zip( xs, ys ) )
    
    random.shuffle( train_data )
    
    train_set = train_data[:11916]
    valid_set = train_data[11916:]
    
    return train_set, valid_set


# In[35]:


class hw5_dataset( torch.utils.data.Dataset ): # this is inherence
    
    def __init__( self, data ):
        
        # define attributes
        self.data = data
        # transfrom may use pytorch's transforms that help us reshape image and transform it into tensors
        # we store the operation in attribute, make it easilier to call
        
    # define what len(this-class) will return    
    def __len__(self):
        return len( self.data )
    
    # this is a fixed form
    # make it possible for user to get specific data by calling index
    def __getitem__(self, idx):
        
        sentence = self.data[idx][0]
        label    = self.data[idx][1]
        
        return sentence, label


# In[10]:


# this is my own try
# epoch = 1500
# lr = 0.0002

HIDDEN_DIM    = 200 

class GRUmodel(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, stacked_num, target_size):
        super(GRUmodel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim
        self.stacked_num   = stacked_num
        self.target_size   = target_size

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # (15, 200), 200
        self.gru = nn.GRU( self.embedding_dim, self.hidden_dim, self.stacked_num, bidirectional=True, 
                          dropout=0.3, batch_first=True )

        # The linear layer that maps from hidden state space to tag space
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2),
            nn.Softmax()
        )

    def forward(self, x):
        
        y, h = self.gru(x)
        h    = h[-1,:,:]
#         print(h.shape)
#         y    = y.mean(1)
#         print(y.shape)
        y    = self.classifier(h)
#         print(y.shape)
        
        # out.size() --> 100, 10
        return y


# In[11]:


model = GRUmodel(EMBEDDING_DIM, HIDDEN_DIM, 2, 2)


# In[13]:


model.load_state_dict(torch.load('gru_class-dropout_model_epoch400_1363.pth'))


# In[21]:


with open(sys.argv[1], 'r', encoding='utf-8') as f:
    test_x = [i.split(',')[1] for i in f.readlines()]
    f.close()
    
test_x = list(nlp.pipe(test_x[1:]))
test_x = [list(i.text for i in j) for j in test_x]
test_x = [ i[:15] if len(i) >= 15 else i + [ '<PAD>' ] * ( 15 - len(i) ) for i in test_x ]
test_x = torch.Tensor( np.array([[ w2v_dict[i] if i in w2v_dict else w2v_dict['<UNK>'] for i in j ] for j in test_x]) )


# In[22]:


model.cuda()
model.eval()
out = torch.argmax( model(test_x.cuda()), dim = 1 )

with open(sys.argv[2], 'w') as f:
    f.write('id,label\n')
    for i in range(len(out)):
        f.write(str(i) + ',' + str(int(out[i])) + '\n')

