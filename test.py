import torch
from tqdm import tqdm
import csv
import data_loader as Dataloader
import pandas as pd 
from sklearn.externals import joblib

model = joblib.load('saved_model.pkl')
BATCH_SIZE=32
test_path=('work/test')
test_name=pd.read_csv('sampleSubmission.csv').values
test_loader=Dataloader.Testloader(test_name,test_path,BATCH_SIZE,shuffle=False)

predict=[]

for i,(x,y) in enumerate(tqdm(test_loader)):
    test_x=torch.unsqueeze(x,1)
    output = model(test_x)
    predict.append(output.tolist())


predict=str(predict)
predict=predict.replace('[','')
predict=predict.replace(']','')
predict=predict.replace('\n','')

Predict=list(eval(predict))
headers=['Id','Predicted']
l=[]
for i in range(len(test_name)):
    l.append((test_name[i][0],Predict[i]))
file=open('test.csv', "w+",newline='')
writer = csv.writer(file)
writer.writerow(headers)
writer.writerows(l)
file.close()
