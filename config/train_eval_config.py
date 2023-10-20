import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification


class train_eval_Config():
    def __init__(self):
        self.lr = 5e-5
        self.batch_size = 16
        self.num_epochs = 1
        self.Datpath = os.path.abspath(os.path.join(os.getcwd(), '..', 'load_dataset'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lblEncode = {'本地旅游': 0,
                             '通报查处': 1,
                             '基建民生': 2,
                             '社会热点': 3,
                             '暖新闻': 4,
                             '人事任免': 5,
                             '政策类型': 6,
                             '产业金融': 7,
                             '人文历史': 8,
                             '数据排名': 9}
        self.reverse_lblEncode= {0: '本地旅游',
                                 1: '通报查处',
                                 2: '基建民生',
                                 3: '社会热点',
                                 4: '暖新闻',
                                 5: '人事任免',
                                 6: '政策类型',
                                 7: '产业金融',
                                 8: '人文历史',
                                 9: '数据排名'}
        self.criterion = torch.nn.CrossEntropyLoss()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=len(self.lblEncode))
        self.lr = 5e-5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')