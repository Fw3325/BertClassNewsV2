import re
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import torch
from torch.utils.data import Dataset, DataLoader
import os
import logging
import sys
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

sys.path.append("/root/autodl-tmp/BertClassNews/")

from config.config import *
from train.utils import Logger

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# class loadDat_Config():
#     def __init__(self):
#         self.path = os.path.abspath(os.path.join(os.getcwd(), '..'))
#         self.Trpath = '{}/data/train_data/train_cls-sample.txt'.format(self.path)
#         self.Testpath = '{}/data/train_data/dev_cls-sample.txt'.format(self.path)
#         self.train_dir = '{}/data/train_data/train_cls.json'.format(self.path)
#         self.val_dir = '{}/data/train_data/val_cls.json'.format(self.path)
#         self.test_dir = '{}/data/train_data/test_cls.json'.format(self.path)
#         self.batch_size = 16
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        # self.Readpath = 
        

        
class BERTDataset(Dataset):

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    
def generateDataloader(df, lbl, config):
    inputs = config.tokenizer(df, padding=True, truncation=True,return_tensors='pt')
    dataset = BERTDataset(inputs, lbl)
    dataloader = DataLoader(dataset, config.batch_size)
    return dataloader


def remove_part_fPred_train(train_df, train_pred):
    cond_acc = []
    for i,j in train_pred:
        cond_acc.append((i==j) |((i!=j)&(j not in ['本地旅游','基建民生','社会热点'])))
    # cond_acc = pd.Series(cond_acc)
    new_train = train_df[cond_acc]
    new_train = new_train.reset_index(drop = True)
    return new_train

def read_process_cls_dat(Readpath):
    try:
        with open(Readpath) as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(Readpath, errors='ignore') as f:
            text = f.read()
    pattern = r'[\t\n]'
    result = re.split(pattern, text)
    contentIndx = 0
    lblIndx = 1
    n = len(result) - 1
    content, label = [], []
    while lblIndx < n:
        content.append(result[contentIndx])
        label.append(result[lblIndx])
        contentIndx += 2
        lblIndx += 2
    df = pd.DataFrame()
    df['tag'] = label
    df['content'] = content
    df['len'] = df['content'].apply(lambda x: sum([i.isalpha() for i in x]))
    df = df[df['len']>27]
    return df

def split_train_val_test_df(train_val_df, test_df,config, is_save = True):
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df['tag'], random_state=42)
    lblEncode = {j:i for i,j in zip(range(len(train_df['tag'].unique())), train_df['tag'].unique())}
    reverse_lblEncode = {lblEncode[i]:i for i in lblEncode}
    if is_save == True:
        save_file(train_df,config.train_dir)
        save_file(val_df, config.val_dir)
        save_file(test_df, config.test_dir)

    train_df['tag'] = train_df['tag'].map(lblEncode)
    val_df['tag'] = val_df['tag'].map(lblEncode)
    test_df['tag'] = test_df['tag'].map(lblEncode)
    return train_df, val_df, test_df, lblEncode, reverse_lblEncode

def save_file(df, path):
    df.to_json(path)

if __name__ == '__main__':
    config = Config().loadDat_config
    log = Logger('./log/2023/app.log')
    train_val_df  = read_process_cls_dat(config.Trpath)
    test_df = read_process_cls_dat(config.Testpath)
    train_df, val_df, test_df, lblEncode, reverse_lblEncode = split_train_val_test_df(train_val_df, test_df,config, is_save = False)
    log.logger.info (train_df.shape, test_df.shape, val_df.shape)
    
    train_dataloader = generateDataloader(train_df['content'].tolist(), train_df['tag'].tolist(),config) 
    val_dataloader = generateDataloader(val_df['content'].tolist(), val_df['tag'].tolist(), config) 
    test_dataloader = generateDataloader(test_df['content'].tolist(), test_df['tag'].tolist(), config) 
    log.logger.info ('DataLoader loading success')
    
    
    