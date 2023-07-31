import math
import torch
import torch.nn as nn


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.layer_norm = nn.LayerNorm(intermediate_dim) if layer_norm else None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, inputs):
        x = self.fc1(self.dropout_func(inputs))
        x = gelu(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        return self.fc2(x)


def _get_loss_fn_and_output_dim(loss_fn_type):
    if loss_fn_type == "CE":
        return nn.CrossEntropyLoss(), 2
    elif loss_fn_type == "BCE":
        return nn.BCELoss(), 1
    else:
        raise ValueError(f'Unsupported loss_fn: {loss_fn_type}')


class Retriever(nn.Module):
    def __init__(self, encoder=None, loss_fn_type="CE", input_dim=1024, intermediate_dim=2048, dropout=0.1, layer_norm=True):
        super(Retriever, self).__init__()
        self.encoder = encoder
        self.loss_fn_type = loss_fn_type
        self.loss_fn, self.output_dim = _get_loss_fn_and_output_dim(loss_fn_type)
        self.classifier = FFNLayer(input_dim=input_dim, intermediate_dim=intermediate_dim,
                                   output_dim=self.output_dim, dropout=dropout, layer_norm=layer_norm)
        self.loss = None

    def forward(self, inputs):
        encoder_out = self.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        if self.loss_fn_type == "CE":
            output = self.classifier(encoder_out.last_hidden_state[:, 0, :])
        elif self.loss_fn_type == "BCE":
            output = self.classifier(encoder_out.last_hidden_state[:, 0, :])
            output = output.flatten().sigmoid()
            inputs["labels"] = inputs["labels"].flatten().float()
        else:
            raise NotImplementedError

        if inputs["labels"] is not None:
            self.loss = self.loss_fn(output, inputs["labels"])

        return output
