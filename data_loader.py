import torch
import numpy as np
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import random

class myDataset(Dataset.Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __len__(self):
        return len(self.Data)
    def __getitem__(self, index):
        data=torch.Tensor(self.Data[index])

        label = torch.Tensor(self.Label[index])

        return data, label



def cut(image):
    image0 = np.pad(image, ((10, 10), (10, 10), (10, 10)), 'constant',constant_values=0)
    ref1=random.randint(0,20)
    ref2=random.randint(0,20)
    ref3=random.randint(0,20)
    image=image0[ref1:ref1+100,ref2:ref2+100,ref3:ref3+100]
    return image



def reflect(image):
    ref = random.randint(0,1)
    if(ref == 1):
        axis = random.randint(0,2) # 0:xÖá·­×ª£¬1:yÖá·­×ª£¬2:zÖá·­×ª
        center = 50
        if(axis == 0):
            for j in range(center): 
                image[j,:,:],image[99-j,:,:] = image[99-j,:,:],image[j,:,:]
        elif(axis == 1):
            for j in range(center):
                image[:,j,:],image[:,99-j,:] = image[:,99-j,:],image[:,j,:]
        elif(axis == 2):
            for j in range(center):
                image[:,:,j],image[:,:,99-j] = image[:,:,99-j],image[:,:,j]
    return image

def Trainloader(data,path,BATCH_SIZE,shuffle):
    dataList=[]
    labels=[]
    for i in range(len(data)):
        file=path+'/'+data[i][0]+'.npz'
        img=np.load(file)['voxel']
        img=cut(img)
        img=reflect(img)
        dataList.append(img)
        labels.append(data[i][1])
        Labels=np.array(labels)
        labelList=Labels.reshape((len(Labels),1,1))
    dataset=myDataset(dataList,labelList)
    dataloader=DataLoader.DataLoader(dataset,batch_size=BATCH_SIZE, shuffle = shuffle)
    return dataloader

def Testloader(name,path,BATCH_SIZE,shuffle):
    dataList=[]
    labels=[]
    for i in range(len(name)):
        file=path+'/'+name[i][0]+'.npz'
        dataList.append(np.load(file)['voxel'])
        labels.append(0)
        Labels=np.array(labels)
        labelList=Labels.reshape((len(Labels),1,1))
    dataset=myDataset(dataList,labelList)
    dataloader=DataLoader.DataLoader(dataset,batch_size=BATCH_SIZE, shuffle = shuffle)
    return dataloader
