import math
import torch
import torch.nn as nn


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, inputs):
        inter = self.fc1(self.dropout_func(inputs))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)


class Retriever(nn.Module):
    def __init__(self, encoder=None, loss_fn_type="CE", input_dim=1024, intermediate_dim=2048, dropout=0.1, layer_norm=True):
        super(Retriever, self).__init__()
        self.loss_fn_type = loss_fn_type
        self.loss_fn = None
        if loss_fn_type == "CE":
            output_dim = 2
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_fn_type == "BCE":
            output_dim = 1
            self.loss_fn = nn.BCELoss()
            self.sigmoid = nn.Sigmoid()
        else:
            raise ValueError(f'Unsupported loss_fn: {loss_fn_type}')
        self.encoder = encoder
        self.classifier = FFNLayer(input_dim=input_dim, intermediate_dim=intermediate_dim,
                                   output_dim=output_dim, dropout=dropout, layer_norm=layer_norm)
        self.loss = None

    def forward(self, inputs):
        encoder_out = self.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        if self.loss_fn_type == "CE":
            output = self.classifier(encoder_out.last_hidden_state[:, 0, :])
        elif self.loss_fn_type == "BCE":
            output = self.sigmoid(self.classifier(encoder_out.last_hidden_state[:, 0, :])).flatten()
            inputs["labels"] = inputs["labels"].flatten().float()
        if inputs["labels"] is not None:
            self.loss = self.loss_fn(output, inputs["labels"])
        return output
