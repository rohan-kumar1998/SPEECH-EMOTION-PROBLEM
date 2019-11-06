import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from sklearn.metrics import f1_score

from collections import OrderedDict
from tqdm import tqdm_notebook as tqdm 
import math, copy, time,os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context(context="talk")
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

from data_loader_test import loader

class LSTM(nn.Module):
    def __init__(self, output_dim, n_mfcc, hid_dim, dropout = 0.5):
        super().__init__()
        self.rnn = nn.LSTM(n_mfcc, hid_dim, bidirectional = True)
        self.fc = nn.Linear(2*hid_dim, output_dim)
        self.hid_dim = 2*hid_dim
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        src = src.permute(1,0,2)
        output, (hidden, c_n) = self.rnn(src)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return hidden

class RNN(nn.Module):
    def __init__(self, output_dim, n_mfcc, hid_dim, dropout = 0.5):
        super().__init__()
        self.rnn = nn.GRU(n_mfcc, hid_dim, bidirectional = True)
        self.fc = nn.Linear(2*hid_dim, output_dim)
        self.hid_dim = 2*hid_dim 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        src = src.permute(1,0,2)
        output, hidden = self.rnn(src)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return hidden

class RNN_LSTMmodel(nn.Module):
    def __init__(self, lstm, rnn, output_dim, dropout):
        super().__init__()
        self.lstm = lstm
        self.rnn = rnn 
        self.fc1 = nn.Linear(lstm.hid_dim + rnn.hid_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out1 = self.lstm(x)
        out2 = self.rnn(x)
        out = torch.cat((out1,out2), dim = 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

output_dim  =  5
n_mfcc = 40
hid_dim = 100 
dropout = 0.5
rnn = RNN(output_dim, n_mfcc , hid_dim ,dropout = 0.5)

output_dim  =  5
n_mfcc = 40
hid_dim = 100 

lstm = LSTM(output_dim, n_mfcc , hid_dim ,dropout = 0.5)

model = RNN_LSTMmodel(lstm, rnn, 5, dropout)

print("Please Enter the Test directory address :", end = "")
test_src = input()

test_data = loader(test_src)

MODEL_DIR = './weights'
MODEL_DIR = os.path.join(MODEL_DIR, 'emotion_model_gru_lstm.pt')
model.load_state_dict(torch.load(MODEL_DIR, map_location = 'cpu'))
# model.load_state_dict(torch.load(MODEL_DIR))

def evaluate(model, test_data):
	model.eval()
	filename = []
	predicted = []
	with torch.no_grad():
		for batch in test_data:
			src = Variable(batch['mfcc2'])
			out = model(src)
			out = out.squeeze() 
			out = out.data  
			_,out1 = torch.max(out, dim = 1) 
			out1 = out1.data.cpu().numpy() 
			for i in range(len(out1)):
				filename.append(batch['filename'][i])
				predicted.append(out1[i])
	
	return filename,predicted

filename, predicted = evaluate(model, test_data)

def predict_output(answer):
    output = ""
    answer = int(answer)
    if(answer == 0):
        output = "disgust"
    elif(answer == 1):
        output = "fear"
    elif(answer == 2):
        output = "happy"
    elif(answer == 3):
        output = "neutral"
    elif(answer == 4):
        output = "sad"
    return output

lines = []
for wav,output in zip(filename,predicted):
    lines.append(str(wav)+","+str(predict_output(output))+'\n')

file1 = open("test_output.txt","w") 
file1.writelines(lines) 
file1.close()