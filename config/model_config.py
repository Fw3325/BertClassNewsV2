from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
class model_Config(object):

    """配置参数"""
    def __init__(self):
        self.path = os.path.abspath(os.path.join(os.getcwd(), '..'))
        self.datPath = '{}/data/train_data/'.format(self.path)
        self.model_dir = 'BertRetrainNewAllDat_v1.pt'
        self.wtPath = '{}/data/checkpoint/'.format(self.path)
        self.batch_size = 16                                           # mini-batch大小
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = 1
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=len(self.lblEncode))
        self.lossn = torch.nn.CrossEntropyLoss