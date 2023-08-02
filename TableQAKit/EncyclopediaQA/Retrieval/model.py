import torch
import torch.nn as nn
import math

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
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)    

class Retriever(nn.Module):
    def __init__(self, bert_model):
        super(Retriever, self).__init__()
        self.bert_model = bert_model
        self.hidden_size = self.bert_model.embeddings.word_embeddings.embedding_dim
        self.projection = FFNLayer(self.hidden_size, self.hidden_size, 1, 0.2)

    def forward(self, data):
        inputs = {"input_ids": data['input_ids'], "attention_mask": data['attention_mask']}
        cls_output = self.bert_model(**inputs)[0][:,0,:]
        logits = self.projection(cls_output)
        probs = torch.softmax(logits, 0)
        return probs
        