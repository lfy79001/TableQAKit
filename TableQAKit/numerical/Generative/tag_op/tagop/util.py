import sys
import random
import torch
import numpy
import logging
from time import gmtime, strftime
from typing import Union, List, Tuple
from tatqa_eval import get_metrics, extract_gold_answers

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tag_op.tagop.tools.allennlp as allennlp


def set_environment(seed, set_cuda=False):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and set_cuda:
        torch.cuda.manual_seed_all(seed)


def create_logger(name, silent=False, to_disk=True, log_file=None):
    """Logger wrapper
    """
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DropEmAndF1(object):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computes exact match and F1 score using the official DROP
    evaluator (which has special handling for numbers and for questions with multiple answer spans,
    among other things).
    """
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __call__(self, prediction: Union[str, List], ground_truth: List[str]):  # type: ignore
        """
        Parameters
        ----------
        prediction: ``Union[str, List]``
            The predicted answer from the model evaluated. This could be a string, or a list of string
            when multiple spans are predicted as answer.
        ground_truths: ``dict``
            All the ground truth answer annotations.
        """
        # If you wanted to split this out by answer type, you could look at [1] here and group by
        # that, instead of only keeping [0].
        # ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in ground_truths]
        #  gold_answers is a string or list of string
        # gold_type, gold_answers, gold_unit = extract_gold_answers(ground_truth)

        if not prediction:
            self._count += 1
        else:
            ground_truth_answer_strings = ground_truth # a list of list
            # print(ground_truth_answer_strings)
            # print(prediction)
            exact_match, f1_score = metric_max_over_ground_truths(
                    get_metrics,
                    [prediction],
                    [ground_truth_answer_strings]
            )
            self._total_em += exact_match
            self._total_f1 += f1_score
            self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official DROP script
        over all inputs.
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return exact_match, f1_score

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __str__(self):
        return f"DropEmAndF1(em={self._total_em}, f1={self._total_f1})"

    
    
class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        self.enc_layer.flatten_parameters()
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)


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
        inter_act = F.gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)    
    
class GCN(nn.Module):
    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(GCN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=False)
        
        ## row/col/cell
        self.row_data_cell_fc = torch.nn.Linear(node_dim, node_dim, bias=False)
        self.col_data_cell_fc = torch.nn.Linear(node_dim, node_dim, bias=False)
        self.data_cell_row_fc = torch.nn.Linear(node_dim, node_dim, bias=False)
        self.data_cell_col_fc = torch.nn.Linear(node_dim, node_dim, bias=False)
        self.same_row_data_cell_fc = torch.nn.Linear(node_dim, node_dim, bias=False)
        self.same_col_data_cell_fc = torch.nn.Linear(node_dim, node_dim, bias=False)
        
        ## row/col - question
        self.row_question_fc = torch.nn.Linear(node_dim, node_dim, bias=False)
        self.col_question_fc = torch.nn.Linear(node_dim, node_dim, bias=False)
        
            

    def forward(self, question_node, table_row_header_nodes,table_col_header_nodes, table_data_cell_nodes,
                     table_row_header_index_mask,table_col_header_index_mask, table_data_cell_index_mask,
                     same_row_data_cell_relation, same_col_data_cell_relation, data_cell_row_rel,
                      data_cell_col_rel):


#         question_node = question_node.unsqueeze(1) ## bsz, 1, h
        
        row_header_nodes_len = table_row_header_nodes.size(1)
        col_header_nodes_len = table_col_header_nodes.size(1)
        data_cell_nodes_len = table_data_cell_nodes.size(1)

        
        
        row_header_data_cell_mask = table_row_header_index_mask.unsqueeze(-1) * table_data_cell_index_mask.unsqueeze(1)
        row_header_data_cell_graph = data_cell_row_rel * row_header_data_cell_mask
        
        col_header_data_cell_mask = table_col_header_index_mask.unsqueeze(-1) * table_data_cell_index_mask.unsqueeze(1)
        col_header_data_cell_graph = data_cell_col_rel * col_header_data_cell_mask
        
        data_cell_data_cell_mask = table_data_cell_index_mask.unsqueeze(-1) * table_data_cell_index_mask.unsqueeze(1)
        same_row_data_cell_graph = same_row_data_cell_relation * data_cell_data_cell_mask
        
        data_cell_data_cell_mask = table_data_cell_index_mask.unsqueeze(-1) * table_data_cell_index_mask.unsqueeze(1)
        same_col_data_cell_graph = same_col_data_cell_relation * data_cell_data_cell_mask
        
#         import pdb
#         pdb.set_trace()
        
        row_header_neighbor_num = row_header_data_cell_graph.sum(-1)
        row_header_neighbor_num_mask = (row_header_neighbor_num >= 1).long()
        row_header_neighbor_num = allennlp.replace_masked_values(row_header_neighbor_num.float()+1, row_header_neighbor_num_mask, 1)
        
        col_header_neighbor_num = col_header_data_cell_graph.sum(-1)
        col_header_neighbor_num_mask = (col_header_neighbor_num >= 1).long()
        col_header_neighbor_num = allennlp.replace_masked_values(col_header_neighbor_num.float()+1, col_header_neighbor_num_mask, 1)
        
        
        data_cell_neighbor_num  = same_row_data_cell_graph.sum(-1) + same_col_data_cell_graph.sum(-1) + row_header_data_cell_graph.transpose(dim0=-1, dim1=1).sum(-1) + col_header_data_cell_graph.transpose(dim0=-1, dim1=1).sum(-1)
        data_cell_neighbor_num_mask = (data_cell_neighbor_num >= 1).long()
        data_cell_neighbor_num = allennlp.replace_masked_values(data_cell_neighbor_num.float(), data_cell_neighbor_num_mask, 1)
        
        import pdb
#         pdb.set_trace()
        
        for step in range(self.iteration_steps):
            row_header_nodes_weight = torch.sigmoid(self._node_weight_fc(table_row_header_nodes)).squeeze(-1)
            col_header_nodes_weight = torch.sigmoid(self._node_weight_fc(table_col_header_nodes)).squeeze(-1)
            data_cell_nodes_weight = torch.sigmoid(self._node_weight_fc(table_data_cell_nodes)).squeeze(-1)

            row_header_data_cell_info = self.row_data_cell_fc(table_data_cell_nodes)
            col_header_data_cell_info = self.col_data_cell_fc(table_data_cell_nodes)
            row_header_question_info = self.row_question_fc(question_node).unsqueeze(1).expand(-1,row_header_nodes_len,-1)
            col_header_question_info = self.col_question_fc(question_node).unsqueeze(1).expand(-1,col_header_nodes_len,-1)
            
            data_cell_same_row_info = self.same_row_data_cell_fc(table_data_cell_nodes)
            data_cell_same_col_info = self.same_col_data_cell_fc(table_data_cell_nodes)
            
            data_cell_row_header_info = self.data_cell_row_fc(table_row_header_nodes)
            data_cell_col_header_info = self.data_cell_col_fc(table_col_header_nodes)
            
            import pdb
#             pdb.set_trace()
            
            row_header_data_cell_weight = allennlp.replace_masked_values(
                data_cell_nodes_weight.unsqueeze(1).expand(-1, row_header_nodes_len, -1),
                row_header_data_cell_graph,
                0)
            
            col_header_data_cell_weight = allennlp.replace_masked_values(
                data_cell_nodes_weight.unsqueeze(1).expand(-1, col_header_nodes_len, -1),
                col_header_data_cell_graph,
                0)
            
            data_cell_data_cell_same_row_weight = allennlp.replace_masked_values(
                data_cell_nodes_weight.unsqueeze(1).expand(-1, data_cell_nodes_len, -1),
                same_row_data_cell_graph,
                0)
            
            data_cell_data_cell_same_col_weight = allennlp.replace_masked_values(
                data_cell_nodes_weight.unsqueeze(1).expand(-1, data_cell_nodes_len, -1),
                same_col_data_cell_graph,
                0)
            
            data_cell_row_header_weight = allennlp.replace_masked_values(
                row_header_nodes_weight.unsqueeze(1).expand(-1, data_cell_nodes_len, -1),
                row_header_data_cell_graph.transpose(dim0=-1, dim1=1),
                0)
            
            data_cell_col_header_weight = allennlp.replace_masked_values(
                col_header_nodes_weight.unsqueeze(1).expand(-1, data_cell_nodes_len, -1),
                col_header_data_cell_graph.transpose(dim0=-1, dim1=1),
                0)
            

            row_header_data_cell_info = torch.matmul(row_header_data_cell_weight, row_header_data_cell_info)
            col_header_data_cell_info = torch.matmul(col_header_data_cell_weight, col_header_data_cell_info)
            
#             import pdb
#             pdb.set_trace()
            
            row_header_question_info = allennlp.replace_masked_values(row_header_question_info, table_row_header_index_mask.unsqueeze(-1).expand(-1,-1,self.node_dim), 0)
            
            col_header_question_info = allennlp.replace_masked_values(col_header_question_info, table_col_header_index_mask.unsqueeze(-1).expand(-1,-1,self.node_dim), 0)
            
            
            data_cell_same_row_info = torch.matmul(data_cell_data_cell_same_row_weight, data_cell_same_row_info)
            data_cell_same_col_info = torch.matmul(data_cell_data_cell_same_col_weight, data_cell_same_col_info)
            
            data_cell_row_header_info = torch.matmul(data_cell_row_header_weight, data_cell_row_header_info)
            data_cell_col_header_info = torch.matmul(data_cell_col_header_weight, data_cell_col_header_info)
            
#             import pdb
#             pdb.set_trace()
            agg_row_header_info = (row_header_data_cell_info + row_header_question_info)/row_header_neighbor_num.unsqueeze(-1)
            agg_col_header_info = (col_header_data_cell_info + col_header_question_info)/col_header_neighbor_num.unsqueeze(-1)
            
            agg_data_cell_info = (data_cell_same_row_info+data_cell_col_header_info+data_cell_row_header_info+data_cell_col_header_info)/data_cell_neighbor_num.unsqueeze(-1)
            
            import pdb
#             pdb.set_trace()
            table_row_header_nodes = F.relu(agg_row_header_info)
            table_col_header_nodes = F.relu(agg_col_header_info)
            table_data_cell_nodes = F.relu(agg_data_cell_info)
        
        return table_row_header_nodes,table_col_header_nodes, table_data_cell_nodes
