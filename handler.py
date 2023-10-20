import torch
from transformers import BertTokenizer

class BERTHandler(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def preprocess(self, requests):
        texts = [req.get('data') for req in requests]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        return inputs
    
    def inference(self, inputs):
        output = self.model(**inputs)
        # inp = torch.stack([t for t in inputs['input_ids']]).to(config.device)
        # attention_mask =torch.stack([t for t in inputs['attention_mask']]).to(config.device)
        # token_type_ids =torch.stack([t for t in inputs['token_type_ids']]).to(config.device)
        # outputs = model(input_ids=inp, attention_mask=attention_mask,
        #               token_type_ids=token_type_ids)
        return output.logits
        # return outputs[0]
    
    def postprocess(self, output):
        return torch.argmax(output, dim=1)