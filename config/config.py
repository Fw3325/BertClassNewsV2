import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import subprocess

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value
        
        
class train_eval_Config():
    def __init__(self):
        self.lr = 5e-5
        self.batch_size = 16
        self.num_epochs = 1
        self.Datpath = os.path.abspath(os.path.join(os.getcwd(), '..', 'load_dataset'))
        # self.Datpath = os.path.abspath(os.path.join(os.getcwd(), 'load_dataset'))
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
        
        

class model_Config(object):

    """配置参数"""
    def __init__(self):
        self.path = os.path.abspath(os.path.join(os.getcwd(), '..'))
        # self.path = os.path.abspath(os.getcwd())
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
        
        
class aug_Config():
    def __init__(self):
        self.augRateRU = 2
        self.augRateCM = {0: 5331,1: 2477,2: 5072,3: 6663,4: 544 * 2,5: 1734,6: 2046,7: 908*2,8: 267*3,9: 639*2}
        self.rw_cn = 1
        self.sw_cn = 1
        self.hoe_cn = 1
        self.rd_cn = 1
        self.evc_cn = 1
        self.rw_cr = 0.3
        self.sw_cr = 0.3
        self.hoe_cr = 0.1
        self.rd_cr = 0.1
        self.evc_cr = 0.5   
        

class loadDat_Config():
    def __init__(self):
        self.path = os.path.abspath(os.path.join(os.getcwd(), '..'))
        # self.path = os.path.abspath(os.getcwd())
        self.Trpath = '{}/data/train_data/train_cls-sample.txt'.format(self.path)
        self.Testpath = '{}/data/train_data/dev_cls-sample.txt'.format(self.path)
        self.train_dir = '{}/data/train_data/train_cls.json'.format(self.path)
        self.val_dir = '{}/data/train_data/val_cls.json'.format(self.path)
        self.test_dir = '{}/data/train_data/test_cls.json'.format(self.path)
        self.batch_size = 16
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        


import abc

class AbstractConfigFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_train_eval_config(self):
        pass

    @abc.abstractmethod  
    def create_model_config(self):
        pass

    @abc.abstractmethod  
    def create_aug_Config(self):
        pass

    @abc.abstractmethod  
    def create_loadDat_Config(self):
        pass

class ConcreteConfigFactory(AbstractConfigFactory):

    def create_train_eval_config(self):
        return train_eval_Config()

    def create_model_config(self):
        return model_Config()

    def create_aug_Config(self):
        return aug_Config()

    def create_loadDat_Config(self):
        return loadDat_Config()

class Config(object):
    def __init__(self):
        self.train_eval_config = train_eval_Config()
        self.model_config = model_Config()
        self.aug_config = aug_Config()
        self.loadDat_config = loadDat_Config()
