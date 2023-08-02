import torch
import torch.nn as nn
from tatqa_metric import TaTQAEmAndF1
from .tools.util import FFNLayer
from .tools import allennlp as util
from typing import Dict, List, Tuple
import numpy as np
from reg_hnt.data.file_utils import is_scatter_available
import math
from reg_hnt.reghnt.model_utils import rnn_wrapper, lens2mask, PoolingFunction, FFN
from reg_hnt.reghnt.new_model import Arithmetic, SubwordAggregation, InputRNNLayer, RGTAT, OtherModel
from reg_hnt.reghnt.modeling_tree import Prediction
from reg_hnt.reghnt.Vocabulary import *
from reg_hnt.data.tatqa_dataset import is_year, clean_year

np.set_printoptions(threshold=np.inf)
if is_scatter_available():
    from torch_scatter import scatter
    from torch_scatter import scatter_max


class RegHNTModel(nn.Module):
    def __init__(self,
                 encoder,
                 config,
                 bsz,
                 operator_classes: int,
                 scale_classes: int,
                 operator_criterion: nn.CrossEntropyLoss = None,
                 scale_criterion: nn.CrossEntropyLoss = None,
                 hidden_size: int = None,
                 dropout_prob: float = None,
                 arithmetic_op_index: List = None,
                 op_mode: int = None,
                 ablation_mode: int = None,
                 ):
        super(RegHNTModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.operator_classes = operator_classes
        self.scale_classes = scale_classes
        if hidden_size is None:
            hidden_size = self.config.hidden_size
            self.hidden_size = hidden_size
        if dropout_prob is None:
            dropout_prob = self.config.hidden_dropout_prob

        self.operator_predictor = FFNLayer(hidden_size, hidden_size, operator_classes, dropout_prob)
        
        self.scale_predictor = FFNLayer(3*hidden_size, hidden_size, scale_classes, dropout_prob)
        
        self.dropout_layer = nn.Dropout(p=0.2)

        self.subword_aggregation = SubwordAggregation(hidden_size, subword_aggregation="attentive-pooling")

        self.operator_criterion = operator_criterion
        self.scale_criterion = scale_criterion
        self.config = config
        self.rnn_layer = InputRNNLayer(hidden_size, hidden_size, cell='lstm', schema_aggregation='attentive-pooling')
        self.hidden_layer = RGTAT(gnn_hidden_size=hidden_size, gnn_num_layers=8, \
            num_heads=8, relation_num=len(Relation_vocabulary['word2id']))
        
        self.arithmetic_model = Arithmetic(hidden_size)
        self.other_model = OtherModel(hidden_size)
        
        self.arithmetic_op_index = arithmetic_op_index

        self._metrics = TaTQAEmAndF1()
        

    def forward(self,**batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        operator_class = batch['operator_class']
        question_ids = batch['question_id']
        position_ids = None
        device = input_ids.device
        
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        
        sequence_output = outputs[0]
        batch_size = sequence_output.size(0)

        
        question, table, paragraph = self.subword_aggregation(sequence_output, batch)
        input_dict = {
            "question": self.dropout_layer(question),
            "table": self.dropout_layer(table),
            "paragraph": self.dropout_layer(paragraph)
        }
        
        inputs = self.rnn_layer(input_dict, batch)

        inputs = self.hidden_layer(inputs, batch)
        
        
        hidden = inputs.new_zeros(batch_size, batch['mask']['b'].size(1), self.hidden_size)
        hidden_output = hidden.masked_scatter_(batch['mask']['b'].unsqueeze(-1), inputs)

        question_hidden = util.replace_masked_values(hidden_output, batch['mask']['question'].unsqueeze(-1).long(), 0)
        question_mean = torch.mean(question_hidden, dim=1)
        table_hidden = util.replace_masked_values(hidden_output, batch['mask']['table'].unsqueeze(-1).long(), 0)
        table_mean = torch.mean(table_hidden, dim=1)
        paragraph_hidden = util.replace_masked_values(hidden_output, batch['mask']['paragraph'].unsqueeze(-1).long(), 0)
        paragraph_mean = torch.mean(paragraph_hidden, dim=1)
        
        q_t_p_mean = torch.cat((question_mean, table_mean, paragraph_mean), dim=-1)
        scale_prediction = self.scale_predictor(q_t_p_mean)

        operator_prediction = self.operator_predictor(sequence_output[:, 0])
        
        outputs = batch['outputs']
        operator_class = batch['operator_class'].tolist()
        
        #####################################################################################################
        arithmetic_labels = list(map(lambda i: 1 if operator_class[i]==OPERATOR_CLASSES["ARITHMETIC"] else 0, range(len(operator_class))))
        arithmetic_num = arithmetic_labels.count(1)
        other_labels = list([1 for i in range(batch_size)])
        
        sequence_output_ari, hidden_output_ari = [], []
        
        for i in range(batch_size):
            if arithmetic_labels[i] == 1:
                sequence_output_ari.append(sequence_output[i])
                hidden_output_ari.append(hidden_output[i])
        

        if arithmetic_num > 0:
            sequence_output_ari = torch.stack(sequence_output_ari, dim=0)
            hidden_output_ari = torch.stack(hidden_output_ari, dim=0)
        
        ######################################################################################################
        arithmetic_loss, other_loss = 0, 0
        table_tag_prediction, paragraph_tag_prediction = None, None
        if arithmetic_num > 0:
            arithmetic_loss, all_node_outputs, target = self.arithmetic_model(sequence_output_ari, hidden_output_ari, batch, arithmetic_labels)
        
        other_loss, table_tag_prediction, paragraph_tag_prediction = self.other_model(sequence_output, hidden_output, batch, other_labels)

        operator_prediction_loss = self.operator_criterion(operator_prediction, torch.tensor(operator_class).cuda())
        
        scale_class = batch['scale_class']
        scale_prediction_loss = self.scale_criterion(scale_prediction, scale_class)

        output_dict = {}
        output_dict['loss'] = arithmetic_loss + other_loss + operator_prediction_loss + scale_prediction_loss
        output_dict["question_id"] = []
        output_dict["answer"] = []
        output_dict["scale"] = []

        LOss = output_dict["loss"].item()
        if np.isnan(LOss):
            import pdb; pdb.set_trace()

        with torch.no_grad():
 
            predicted_operator_class = torch.argmax(operator_prediction, dim=-1).tolist()
            predicted_scale_class = torch.argmax(scale_prediction, dim=-1).detach().cpu().numpy()
            
            for i in range(len(predicted_operator_class)):
                if predicted_operator_class[i] == OPERATOR_CLASSES["ARITHMETIC"]:
                    sequence_output_i = sequence_output[i].unsqueeze(0)
                    hidden_output_i = hidden_output[i].unsqueeze(0)
                    batch_label_i = list(map(lambda x: 1 if x==i else 0, range(batch_size)))
                    answer = self.arithmetic_model.predict(sequence_output_i, hidden_output_i, batch, batch_label_i, int(predicted_scale_class[i]), batch['answer_dict'][i])
                else:
                    sequence_output_i = sequence_output[i].unsqueeze(0)
                    hidden_output_i = hidden_output[i].unsqueeze(0)
                    batch_label_i = list(map(lambda x: 1 if x==i else 0, range(batch_size)))
                    answer = self.other_model.predict(hidden_output_i, batch, batch_label_i, predicted_operator_class[i], table_tag_prediction[i].unsqueeze(0), paragraph_tag_prediction[i].unsqueeze(0))

                if not answer:
                    answer = -1
                
                output_dict["answer"].append(answer)
                output_dict["scale"].append(SCALE[int(predicted_scale_class[i])])
                output_dict["question_id"].append(question_ids[i])
                
                self._metrics(batch['answer_dict'][i], answer, SCALE[int(predicted_scale_class[i])])

        return output_dict
    
    def predict(self, **batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        question_ids = batch['question_id']
        position_ids = None

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        
        sequence_output = outputs[0]
        batch_size = sequence_output.size(0)
        
        question, table, paragraph = self.subword_aggregation(sequence_output, batch)
        input_dict = {
            "question": self.dropout_layer(question),
            "table": self.dropout_layer(table),
            "paragraph": self.dropout_layer(paragraph)
        }
        
        inputs = self.rnn_layer(input_dict, batch)
        
        inputs = self.hidden_layer(inputs, batch)
        
        hidden = inputs.new_zeros(batch_size, batch['mask']['b'].size(1), self.hidden_size)
        hidden_output = hidden.masked_scatter_(batch['mask']['b'].unsqueeze(-1), inputs)

        question_hidden = util.replace_masked_values(hidden_output, batch['mask']['question'].unsqueeze(-1).long(), 0)
        question_mean = torch.mean(question_hidden, dim=1)
        table_hidden = util.replace_masked_values(hidden_output, batch['mask']['table'].unsqueeze(-1).long(), 0)
        table_mean = torch.mean(table_hidden, dim=1)
        paragraph_hidden = util.replace_masked_values(hidden_output, batch['mask']['paragraph'].unsqueeze(-1).long(), 0)
        paragraph_mean = torch.mean(paragraph_hidden, dim=1)
        
        q_t_p_mean = torch.cat((question_mean, table_mean, paragraph_mean), dim=-1)
        scale_prediction = self.scale_predictor(q_t_p_mean)
        operator_prediction = self.operator_predictor(sequence_output[:, 0])
        
        predicted_operator_class = torch.argmax(operator_prediction, dim=-1).tolist()
        predicted_scale_class = torch.argmax(scale_prediction, dim=-1).detach().cpu().numpy()

        output_dict = {}
        
        output_dict["question_id"] = []
        output_dict["answer"] = []
        output_dict["scale"] = []
        output_dict["gold_answers"] = []
        answer = -1
        if predicted_operator_class[0] == OPERATOR_CLASSES["ARITHMETIC"]:
            answer = self.arithmetic_model.predict(sequence_output, hidden_output, batch, [1], int(predicted_scale_class[0]), batch['answer_dict'][0])
        else:
            _, table_tag_prediction, paragraph_tag_prediction = self.other_model.forward2(sequence_output, hidden_output, batch, [1])
            answer = self.other_model.predict(hidden_output, batch, [1], predicted_operator_class[0], table_tag_prediction, paragraph_tag_prediction)


        if "which year " in batch['questions'][0] or "Which year " in batch['questions'][0] or "Which FY " in batch['questions'][0] or "which FY " in batch['questions'][0]:
            if isinstance(answer, list) and len(answer) == 1 and is_year(answer[0]):
                answer[0] = clean_year(answer[0])
        
        if "which years" in batch['questions'][0] or "Which years" in batch['questions'][0]:
            if isinstance(answer, list) and len(answer) > 1:
                for i in range(len(answer)):
                    if is_year(answer[i]):
                        answer[i] = clean_year(answer[i])

        
        output_dict["answer"].append(answer)
        output_dict["scale"].append(SCALE[int(predicted_scale_class[0])])
        output_dict["question_id"].append(question_ids[0])
        output_dict["gold_answers"].append(batch['answer_dict'][0])

        current_op = OPERATOR_CLASSES_R[predicted_operator_class[0]]
        gold_op = OPERATOR_CLASSES_R[batch['operator_class'].tolist()[0]]

        
        self._metrics({**batch['answer_dict'][0], "uid": question_ids[0]}, answer,
                SCALE[int(predicted_scale_class[0])], None, None, 
                pred_op=current_op, gold_op=gold_op)
        return output_dict
        
        
    def predict2(self, **batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        question_ids = batch['question_id']
        position_ids = None

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        
        sequence_output = outputs[0]
        batch_size = sequence_output.size(0)
        
        question, table, paragraph = self.subword_aggregation(sequence_output, batch)
        input_dict = {
            "question": self.dropout_layer(question),
            "table": self.dropout_layer(table),
            "paragraph": self.dropout_layer(paragraph)
        }
        
        inputs = self.rnn_layer(input_dict, batch)
        
        inputs = self.hidden_layer(inputs, batch)
        
        hidden = inputs.new_zeros(batch_size, batch['mask']['b'].size(1), self.hidden_size)
        hidden_output = hidden.masked_scatter_(batch['mask']['b'].unsqueeze(-1), inputs)

        question_hidden = util.replace_masked_values(hidden_output, batch['mask']['question'].unsqueeze(-1).long(), 0)
        question_mean = torch.mean(question_hidden, dim=1)
        table_hidden = util.replace_masked_values(hidden_output, batch['mask']['table'].unsqueeze(-1).long(), 0)
        table_mean = torch.mean(table_hidden, dim=1)
        paragraph_hidden = util.replace_masked_values(hidden_output, batch['mask']['paragraph'].unsqueeze(-1).long(), 0)
        paragraph_mean = torch.mean(paragraph_hidden, dim=1)
        
        q_t_p_mean = torch.cat((question_mean, table_mean, paragraph_mean), dim=-1)
        scale_prediction = self.scale_predictor(q_t_p_mean)
        operator_prediction = self.operator_predictor(sequence_output[:, 0])
        
        
        
        predicted_operator_class = torch.argmax(operator_prediction, dim=-1).tolist()
        predicted_scale_class = torch.argmax(scale_prediction, dim=-1).detach().cpu().numpy()

        output_dict = {}
        
        output_dict["question_id"] = []
        output_dict["answer"] = []
        output_dict["scale"] = []
        output_dict["operator"] = []

        if predicted_operator_class[0] == OPERATOR_CLASSES["ARITHMETIC"]:
            answer = self.arithmetic_model.predict(sequence_output, hidden_output, batch, [1], int(predicted_scale_class[0]), 0)
        else:
            _, table_tag_prediction, paragraph_tag_prediction = self.other_model.forward2(sequence_output, hidden_output, batch, [1])
            answer = self.other_model.predict(hidden_output, batch, [1], predicted_operator_class[0], table_tag_prediction, paragraph_tag_prediction)


        if "which year " in batch['questions'][0] or "Which year " in batch['questions'][0] or "Which FY " in batch['questions'][0] or "which FY " in batch['questions'][0]:
            if isinstance(answer, list) and len(answer) == 1 and is_year(answer[0]):
                answer[0] = clean_year(answer[0])
        
        if "which years" in batch['questions'][0] or "Which years" in batch['questions'][0]:
            if isinstance(answer, list) and len(answer) > 1:
                for i in range(len(answer)):
                    if is_year(answer[i]):
                        answer[i] = clean_year(answer[i])

        
        output_dict["answer"].append(answer)
        output_dict["scale"].append(SCALE[int(predicted_scale_class[0])])
        output_dict["question_id"].append(question_ids[0])
        output_dict["operator"].append(predicted_operator_class[0])

        

        return output_dict
        
        
    def reset(self):
        self._metrics.reset()

    def set_metrics_mdoe(self, mode):
        self._metrics = TaTQAEmAndF1(mode=mode)

    def get_metrics(self, logger=None, reset: bool = False) -> Dict[str, float]:
        detail_em, detail_f1 = self._metrics.get_detail_metric()

        raw_detail = self._metrics.get_raw_pivot_table()
        exact_match, f1_score, scale_score, op_score = self._metrics.get_overall_metric(reset)
        print(f"raw matrix:{raw_detail}\r\n")
        print(f"detail em:{detail_em}\r\n")
        print(f"detail f1:{detail_f1}\r\n")
        print(f"global em:{exact_match}\r\n")
        print(f"global f1:{f1_score}\r\n")
        print(f"global scale:{scale_score}\r\n")
        print(f"global op:{op_score}\r\n")
        if logger is not None:
            logger.info(f"raw matrix:{raw_detail}\r\n")
            logger.info(f"detail em:{detail_em}\r\n")
            logger.info(f"detail f1:{detail_f1}\r\n")
            logger.info(f"global em:{exact_match}\r\n")
            logger.info(f"global f1:{f1_score}\r\n")
            logger.info(f"global scale:{scale_score}\r\n")
            logger.info(f"global op:{op_score}\r\n")
        return {'em': exact_match, 'f1': f1_score, "scale": scale_score, "op": op_score}

    def get_df(self):
        raws = self._metrics.get_raw()
        detail_em, detail_f1 = self._metrics.get_detail_metric()
        raw_detail = self._metrics.get_raw_pivot_table()
        return detail_em, detail_f1, raws, raw_detail


    
