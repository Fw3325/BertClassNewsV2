import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import sklearn.model_selection
from torch.nn import CrossEntropyLoss
import time
import subprocess
import numpy as np
# from predict import Config
import sys
import os
import logging
import warnings
sys.path.append("/root/autodl-tmp/BertClassNews/")
from config.config import *
from load_dataset.load_data_cls import *

from utils import Logger


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore")

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value
        
# class Config():
#     def __init__(self):
#         self.lr = 5e-5
#         self.batch_size = 16
#         self.num_epochs = 1
#         self.Datpath = os.path.abspath(os.path.join(os.getcwd(), '..', 'load_dataset'))
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.lblEncode = {'本地旅游': 0,
#                              '通报查处': 1,
#                              '基建民生': 2,
#                              '社会热点': 3,
#                              '暖新闻': 4,
#                              '人事任免': 5,
#                              '政策类型': 6,
#                              '产业金融': 7,
#                              '人文历史': 8,
#                              '数据排名': 9}
#         self.reverse_lblEncode= {0: '本地旅游',
#                                  1: '通报查处',
#                                  2: '基建民生',
#                                  3: '社会热点',
#                                  4: '暖新闻',
#                                  5: '人事任免',
#                                  6: '政策类型',
#                                  7: '产业金融',
#                                  8: '人文历史',
#                                  9: '数据排名'}
#         self.criterion = torch.nn.CrossEntropyLoss()
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#         self.model = BertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=len(self.lblEncode))
#         self.lr = 5e-5
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, dataloader, config):
    model.to(config.device)
    model.eval()
    final_pred = []
    total = 0
    correct = 0
    pred = {i:[] for i in range(len(config.lblEncode))}
    with torch.no_grad():

        for batch in dataloader:
            inputs = torch.stack([t for t in batch['input_ids']]).to(config.device)
            labels = torch.tensor(batch['labels']).to(config.device)
            attention_mask =torch.stack([t for t in batch['attention_mask']]).to(config.device)
            outputs = model(input_ids=inputs, attention_mask=attention_mask,
                      labels=labels)
            predictions = outputs[1].argmax(dim=1)
            for i in range(len(predictions)):
                predRes = predictions[i].item()
                pred[predRes].append((predictions[i] == labels[i]).item())
                final_pred.append((config.reverse_lblEncode[predRes], config.reverse_lblEncode[labels[i].item()]))
                # final_pred.append((predRes, labels[i].item()))

            correct += (predictions == labels).sum().item()
            total += labels.shape[0]
      
    accuracy = correct/total
    res = {config.reverse_lblEncode[i]:sum(pred[i])/len(pred[i]) if len(pred[i]) > 0 else np.nan for i in pred }
    return accuracy, res, final_pred


def model_train(model, train_dataloader, val_dataloader, config):
    model.to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.lr)
    for epoch in range(config.num_epochs):
      start = time.time()
      losses = 0
      model.train()
      idx = 0
      for batch in train_dataloader:
        inputs, token_type_ids, attention_mask, labels = batch['input_ids'], batch['token_type_ids'], batch['attention_mask'], batch['labels']

        input_ids = torch.stack([t for t in batch['input_ids']]).to(config.device)

        attention_mask =torch.stack([t for t in batch['attention_mask']]).to(config.device)
        labels = torch.tensor(batch['labels']).to(config.device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,labels=labels)
        loss = config.criterion(outputs['logits'],labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses += loss.item()
        acc = (outputs['logits'].argmax(1) == labels).float().mean()
        idx += 1
        curr = time.time()
        timeConsume = curr - start
        if idx % 300 == 0:
            print (f'Epoch {epoch+1}, batch {idx}, so far takes {timeConsume}')
          # los  =outputs['loss']
            print (f'Loss: {losses}')

      print (acc)
      val_acc, val_cat_acc, val_final_pred= evaluate(model, val_dataloader, config)
      print ('val acc:', val_acc)
        # if idx % 10 == 0:
        #     logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
        #                   f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")

      print(f'Epoch {epoch+1} complete, so far takes {timeConsume}')
    return model


def main():
    config = Config().loadDat_config
    train_val_df  = read_process_cls_dat(config.Trpath)
    test_df = read_process_cls_dat(config.Testpath)
    train_df, val_df, test_df, lblEncode, reverse_lblEncode = split_train_val_test_df(train_val_df, test_df,config, is_save = False)
    log = Logger('./log/2023/app.log')
    log.logger.info (train_df.shape, test_df.shape, val_df.shape)
    
    train_dataloader = generateDataloader(train_df['content'].tolist(), train_df['tag'].tolist(),config) 
    val_dataloader = generateDataloader(val_df['content'].tolist(), val_df['tag'].tolist(), config) 
    test_dataloader = generateDataloader(test_df['content'].tolist(), test_df['tag'].tolist(), config) 
    log.logger.info ('DataLoader loading success')
    
    config2 = Config().train_eval_config 
    model = config2.model
    model_train(model, train_dataloader, val_dataloader, config2)


if __name__ == '__main__':
    main()
    