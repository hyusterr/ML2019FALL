#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import pandas as pd
import spacy
import torch
from torch import nn
import random
from gensim.models import Word2Vec


# In[2]:


with open(sys.argv[1], 'r', encoding='utf-8') as f:
    train_X = [tuple(i.split(',')) for i in f.readlines()]
    f.close()


# In[3]:


with open(sys.argv[2], 'r', encoding='utf-8') as f:
    train_Y = [tuple(i.split(',')) for i in f.readlines()]
    f.close()


# In[4]:


Y = [int(i[1]) for i in train_Y[1:]]


# In[5]:


texts = [i[1] for i in train_X[1:]]


# In[6]:


# there are nlp.pipe in spacy, including tokenizer, parser, ner, postagger...
nlp = spacy.load("en_core_web_sm")
docs = list(nlp.pipe(texts))


# In[7]:


# transfer docs into list of lists
X = [list(i.text for i in j) for j in docs] # even can take emoji!
# how long is the dictionary?


# In[29]:


EMBEDDING_DIM = 100
# there are paras can adjust for stopwords, based on frequency
W2Vmodel = Word2Vec(X, size=EMBEDDING_DIM, window=5, min_count=1, workers=-1)
W2Vmodel.save( str(EMBEDDING_DIM) + "split-dim-word2vec.model" )

w2v = []
for _, key in enumerate(W2Vmodel.wv.vocab):
    w2v.append((key, W2Vmodel.wv[key]))
special_tokens = ["<PAD>", "<UNK>"]
for token in special_tokens:
    w2v.append((token, [0.0] * EMBEDDING_DIM))


# In[30]:


w2v_dict = { i[0]: i[1] for i in w2v }


# In[31]:


XX = [ i[:10] if len(i) >= 10 else i + [ '<PAD>' ] * ( 10 - len(i) ) for i in X ]


# In[32]:


train_x = torch.Tensor( np.array([[ w2v_dict[i] for i in j ] for j in XX]) )


# In[33]:


train_y = torch.Tensor( np.array(Y) ).long() # long is very very critical


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


# In[23]:


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


# In[37]:


if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()

    # get data 
    train_set, valid_set = load_data(train_x, train_y)
    
    # read data and preprocess
    
    train_dataset = hw5_dataset(train_set)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=72, shuffle=True)

    valid_dataset = hw5_dataset(valid_set)
    valid_loader  = torch.utils.data.DataLoader(valid_dataset, batch_size=72, shuffle=False)

    model = GRUmodel(EMBEDDING_DIM, HIDDEN_DIM, 2, 2)
    
    if use_gpu:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    num_epoch = 1500
    
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
        
        for idx, (sent, label) in enumerate(train_loader):
            
            if use_gpu:
                sent  = sent.cuda()
                label = label.cuda()
            
#             print(sent.shape)
            
            optimizer.zero_grad()
            output = model(sent) #.view(-1, sent.shape[0], HIDDEN_DIM))
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
            valid_acc  = []
            for idx, (sent, label) in enumerate(valid_loader):
                
#                 print(label)
                if use_gpu:
                    sent  = sent.cuda()
                    label = label.cuda()
                
#                 print(label.shape)
                
                output = model(sent)
                loss = loss_fn(output, label)
                predict = torch.max(output, 1)[1]
                
                
                acc = np.mean((label == predict).cpu().numpy())
                valid_loss.append(loss.item())
                valid_acc.append(acc)
                
            val_epoch_acc.append( ( epoch + 1, np.mean(np.mean(valid_acc)) ) )
            val_epoch_loss.append( ( epoch + 1, np.mean(np.mean(valid_loss)) ) )
            
            print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, np.mean(valid_loss), np.mean(valid_acc)))
        
        if np.mean(train_acc) > 0.75:
            checkpoint_path = 'gru_small_class-dropout_model_{}.pth'.format(epoch + 1) 
            torch.save(model.state_dict(), checkpoint_path)
            print('model saved to %s' % checkpoint_path)


# In[229]:


# model.load_state_dict(torch.load('gru_class-dropout_model_epoch400_1363.pth'))


# In[21]:


with open(sys.argv[3], 'r', encoding='utf-8') as f:
    test_x = [i.split(',')[1] for i in f.readlines()]
    f.close()
    
test_x = list(nlp.pipe(test_x[1:]))
test_x = [list(i.text for i in j) for j in test_x]
test_x = [ i[:15] if len(i) >= 15 else i + [ '<PAD>' ] * ( 15 - len(i) ) for i in test_x ]
test_x = torch.Tensor( np.array([[ w2v_dict[i] if i in w2v_dict else w2v_dict['<UNK>'] for i in j ] for j in test_x]) )


# In[22]:


out = torch.argmax( model(test_x.cuda()), dim = 1 )

with open('out.csv', 'w') as f:
    f.write('id,label\n')
    for i in range(len(out)):
        f.write(str(i) + ',' + str(int(out[i])) + '\n')

