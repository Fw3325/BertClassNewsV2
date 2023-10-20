import torch
from transformers import BertModel

from transformers import BertTokenizer, BertForSequenceClassification

import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value
        
n_labels = 2

def get_token_model(n_labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    model = BertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=n_labels)
    return tokenizer, model

if __name__ == '__main__':
    tokenizer, model = get_token_model(n_labels)
    print (model)
    
