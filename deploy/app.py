import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import sys
import os
import logging
import warnings
sys.path.append("/root/autodl-tmp/BertClassNews/")
from config.config import *
from load_dataset.load_data_cls import *
from train.utils import *
warnings.filterwarnings("ignore")

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

app = Flask(__name__)
config = Config().model_config
model = config.model
model = save_model(model, config)
model.to(config.device)
model.eval()


def get_prediction(text, config):
    with torch.no_grad():
        inputs = config.tokenizer(text, padding=True, truncation=True,return_tensors='pt')
        inp = torch.stack([t for t in inputs['input_ids']]).to(config.device)
        attention_mask =torch.stack([t for t in inputs['attention_mask']]).to(config.device)
        token_type_ids =torch.stack([t for t in inputs['token_type_ids']]).to(config.device)
        outputs = model(input_ids=inp, attention_mask=attention_mask,
                      token_type_ids=token_type_ids)
        predictions = outputs[0].argmax(dim=1)
        class_name = [config.reverse_lblEncode[i] for i in predictions.cpu().tolist()]
        class_id = [i for i in predictions.cpu().tolist()]
    return class_id, class_name




@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        # 转化为字节
        # text = file.read()
        config = Config().model_config
        config2 = Config().loadDat_config
        # test_df = read_process_cls_dat(config2.Testpath)
        # class_id, class_name = get_prediction(test_df.head()['content'], config)
        class_id, class_name = get_prediction(text, config)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006)