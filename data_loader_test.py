import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook as tqdm 
import math, copy, time, os
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

import numpy as np
import pandas as pd

class dataset(Dataset):
    def __init__(self, df):
        super().__init__() 
        data = []
        for i in range(len(df)):
            mfcc1, mfcc2 = self.feature_extractor(df['address'][i])
            data.append({"mfcc1" : mfcc1, "mfcc2" : mfcc2, "filename" : df['address'][i].split('/')[-1]})
        self.data = data
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def feature_extractor(self, filename):
        waveform, sample_rate = torchaudio.load(filename)
        mfcc = torchaudio.transforms.MFCC()(waveform)
        mfcc1 = mfcc.reshape(1,-1,40).squeeze()
        mfcc2 = torch.FloatTensor(np.nan_to_num(mfcc1.log2().detach().numpy()))
        return mfcc1, mfcc2
    
    def one_hot(self,idx, mx):
        hot = np.zeros(mx)
        hot[idx] = 1 
        return torch.Tensor(hot)

def collater(batch):
    mfcc1 = [item['mfcc1'] for item in batch]
    mfcc2 = [item['mfcc2'] for item in batch]
    filename = [item['filename'] for item in batch]
    mfcc1 = pad_sequence(mfcc1, batch_first= True, padding_value= 0)
    mfcc2 = pad_sequence(mfcc2, batch_first=True, padding_value=0)
    new_batch = {'mfcc1' : mfcc1, 'mfcc2' : mfcc2, 'filename':filename}
    return new_batch

def convert_to_df(speech):
    speech = np.array(speech)
    df = pd.DataFrame(speech)
    df = df.rename(index=str, columns={0: 'address'})
    return df

def data_split(df):    
    train, test = train_test_split(df, test_size=0.3)
    return train.reset_index(), test.reset_index()


def loader(test_src, batch_size = 10):
    speech = []
    labels = []
    for i in os.listdir(test_src):
        speech.append(os.path.join(test_src,i))
    test_df = convert_to_df(speech)
    test_data = dataset(test_df)    
    test_data = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, collate_fn= collater ,drop_last=True)
    return test_data