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
        for i in tqdm(range(len(df))):
            mfcc1, mfcc2 = self.feature_extractor(df['address'][i])
            data.append({"mfcc1" : mfcc1, "mfcc2" : mfcc2, "label_one_hot" : self.one_hot(df["labels"][i], 5) ,"labels" : df["labels"][i]})
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
    labels = [item['labels'] for item in batch]
    labels = torch.LongTensor(labels)
    label_one_hot = [item['label_one_hot'] for item in batch]
    mfcc1 = pad_sequence(mfcc1, batch_first= True, padding_value= 0)
    mfcc2 = pad_sequence(mfcc2, batch_first=True, padding_value=0)
    label_one_hot = pad_sequence(label_one_hot, batch_first=True, padding_value=0)
    new_batch = {'mfcc1' : mfcc1, 'mfcc2' : mfcc2, 'labels' : labels, 'label_one_hot' : label_one_hot}
    return new_batch

def convert_to_df(speech, labels):
    df = pd.DataFrame(speech)
    df[1] = labels
    df = df.rename(index=str, columns={0: 'address', 1 : 'labels'})
    return df

def data_split(df):    
    train, test = train_test_split(df, test_size=0.3)
    return train.reset_index(), test.reset_index()


def loader(train_src, val_src, batch_size):
    speech = []
    labels = []
    for i in tqdm(range(len(os.listdir(train_src)))):
        wav_dir = os.path.join(train_src, os.listdir(train_src)[i])
        for k in range(min(len(os.listdir(wav_dir)), 1000)):
            j = os.listdir(wav_dir)[k]
            speech.append(os.path.join(wav_dir, j))
            labels.append(i)
    train_df = convert_to_df(speech, labels)
    train_data, test_data = data_split(train_df)
    speech = []
    labels = []
    for i in range(len(os.listdir(val_src))):
        wav_dir = os.path.join(val_src, os.listdir(val_src)[i])
        for j in os.listdir(wav_dir):
            speech.append(os.path.join(wav_dir, j))
            labels.append(i)
    val_data = convert_to_df(speech,labels)
    train_data = dataset(train_data)
    test_data = dataset(test_data)
    val_data = dataset(val_data)
    train_data = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn= collater ,drop_last=True)
    test_data = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, collate_fn= collater ,drop_last=True)
    val_data = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, collate_fn= collater ,drop_last=True)
    return train_data, val_data, test_data