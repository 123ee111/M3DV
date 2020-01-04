import torch
import numpy as np
import pandas as pd   
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import csv
from torch import tensor
import os
from torch.autograd import Variable
import data_loader as Dataloader                                                                
from sklearn.externals import joblib
from Model import ResNet18

BATCH_SIZE = 32
NUM_EPOCHS = 40

model = ResNet18()
    

train_path=('train_val')
train_data = pd.read_csv("train_val.csv").values
train_loader=Dataloader.Trainloader(train_data,train_path,BATCH_SIZE,shuffle=True)

    
loss_func = torch.nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(NUM_EPOCHS):
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    
    for i,(x,y) in enumerate(tqdm(train_loader)):
        model.train()
        length = len(x)
        batch_x = torch.unsqueeze(Variable(x),1)
        batch_y=y
        batch_y=batch_y.view(batch_y.size(0),1)
        batch_y=Variable(batch_y)
        output=model(batch_x)
        opt.zero_grad() 

        loss = loss_func(output,batch_y)
       
        
        loss.backward() 
        opt.step()

        for i in range(len(output)):
            if(output[i]>0.5):
                output[i] = 1
            else:
                output[i] = 0
        sum_loss += loss.item()
        predicted =output
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum()

    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                % (epoch + 1, (i + 1 + epoch * length), sum_loss / total, 100. * correct / total))

joblib.dump(model, 'saved_model.pkl')
