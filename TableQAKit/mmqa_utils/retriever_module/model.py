from transformers import AutoModel
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):

        inter = self.fc1(self.dropout_func(input))
        # inter_act = gelu(inter)
        inter_act = F.gelu(inter) # ?
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)


class RetrieverModel(nn.Module):
    """
        retrieve image and passages for each question
    """
    def __init__(self, bert_model = 'microsoft/deberta-v3-large', output_dim = 1, dropout = 0.2):
        if type(bert_model) == str:
            bert_model = AutoModel.from_pretrained(bert_model)
        
        super(RetrieverModel, self).__init__()
        self.bert_model = bert_model
        self.hidden_size = self.bert_model.embeddings.word_embeddings.embedding_dim
        self.projection = FFNLayer(self.hidden_size, self.hidden_size, output_dim, dropout=dropout)

    def forward(self, data):
        inputs = {"input_ids": data['input_ids'], "attention_mask": data['input_mask']}
        cls_output = self.bert_model(**inputs)[0][:,0,:]
        logits = self.projection(cls_output)
        # probs = logits.squeeze(-1).unsqueeze(0)

        probs = logits.squeeze(-1)
        probs = torch.softmax(probs, -1)
        return probs

if __name__ == '__main__':
    bert_model = AutoModel.from_pretrained("microsoft/deberta-v3-large")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = RetrieverModel(bert_model)
