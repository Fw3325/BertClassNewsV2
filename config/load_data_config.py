import os
from transformers import BertTokenizer

class loadDat_Config():
    def __init__(self):
        self.path = os.path.abspath(os.path.join(os.getcwd(), '..'))
        self.Trpath = '{}/data/train_data/train_cls-sample.txt'.format(self.path)
        self.Testpath = '{}/data/train_data/dev_cls-sample.txt'.format(self.path)
        self.train_dir = '{}/data/train_data/train_cls.json'.format(self.path)
        self.val_dir = '{}/data/train_data/val_cls.json'.format(self.path)
        self.test_dir = '{}/data/train_data/test_cls.json'.format(self.path)
        self.batch_size = 16
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')