import torch
import torch.nn as nn
from tatqa_metric import TaTQAEmAndF1
from .tools.util import FFNLayer
from .tools import allennlp as util
from typing import Dict, List, Tuple
import numpy as np
from tag_op.data.file_utils import is_scatter_available
from tag_op.data.data_util import get_op_1, get_op_2, get_op_3, SCALE, OPERATOR_CLASSES_
from tag_op.tagop.util import GCN, ResidualGRU
from torch.nn import MultiheadAttention
import torch.nn.functional as F

np.set_printoptions(threshold=np.inf)
# soft dependency
if is_scatter_available():
    from torch_scatter import scatter
    from torch_scatter import scatter_max


def get_continuous_tag_slots(paragraph_token_tag_prediction):
    tag_slots = []
    span_start = False
    for i in range(1, len(paragraph_token_tag_prediction)):
        if paragraph_token_tag_prediction[i] != 0 and not span_start:
            span_start = True
            start_index = i
        if paragraph_token_tag_prediction[i] == 0 and span_start:
            span_start = False
            tag_slots.append((start_index, i))
    if span_start:
        tag_slots.append((start_index, len(paragraph_token_tag_prediction)))
    return tag_slots


def get_span_tokens_from_paragraph(paragraph_token_tag_prediction, paragraph_tokens) -> List[str]:
    '''
    return all the spans.
    '''
    import pdb
#     pdb.set_trace()
    span_tokens = []
    span_start = False
    for i in range(1, min(len(paragraph_tokens) + 1, len(paragraph_token_tag_prediction))):
        if paragraph_token_tag_prediction[i] == 0:
            span_start = False
        if paragraph_token_tag_prediction[i] != 0:
            if not span_start:
                span_tokens.append([paragraph_tokens[i - 1]])
                span_start = True
            else:
                span_tokens[-1] += [paragraph_tokens[i - 1]]
    span_tokens = [" ".join(tokens) for tokens in span_tokens]
    return span_tokens


def get_span_tokens_from_table(table_cell_tag_prediction, table_cell_tokens) -> List[str]:
    span_tokens = []
    for i in range(1, len(table_cell_tag_prediction)):
        if table_cell_tag_prediction[i] != 0:
            span_tokens.append(str(table_cell_tokens[i-1]))
    return span_tokens


def get_single_span_tokens_from_paragraph(paragraph_token_tag_prediction,
                                          paragraph_token_tag_prediction_score,
                                          paragraph_tokens) -> List[str]:
    '''
    return a single span with max mean probability
    '''
    import pdb
#     pdb.set_trace()
    tag_slots = get_continuous_tag_slots(paragraph_token_tag_prediction)
    best_result = float("-inf")
    best_combine = []
    for tag_slot in tag_slots:
        current_result = np.mean(paragraph_token_tag_prediction_score[tag_slot[0]:tag_slot[1]])
        if current_result > best_result:
            best_result = current_result
            best_combine = tag_slot
    if not best_combine:
        return []
    else:
        return [" ".join(paragraph_tokens[best_combine[0] - 1: best_combine[1] - 1])]

def get_single_span_tokens_from_table(table_cell_tag_prediction,
                                      table_cell_tag_prediction_score,
                                      table_cell_tokens) -> List[str]:
    '''
    return a string of cell in table with the max probability.
    '''
    import pdb
#     pdb.set_trace()
    
    tagged_cell_index = [i for i in range(len(table_cell_tag_prediction)) if table_cell_tag_prediction[i] != 0]
    if not tagged_cell_index:
        return []
    tagged_cell_tag_prediction_score = \
        [table_cell_tag_prediction_score[i] for i in tagged_cell_index]
    best_result_index = tagged_cell_index[int(np.argmax(tagged_cell_tag_prediction_score))]
    return [str(table_cell_tokens[best_result_index-1])]

def get_numbers_from_reduce_sequence(sequence_reduce_tag_prediction, sequence_numbers):
    return [sequence_numbers[i - 1] for i in
            range(1, min(len(sequence_numbers) + 1, len(sequence_reduce_tag_prediction)))
            if sequence_reduce_tag_prediction[i] != 0 and np.isnan(sequence_numbers[i - 1]) is not True]


def get_numbers_from_table(cell_tag_prediction, table_numbers):
    return [table_numbers[i] for i in range(len(cell_tag_prediction)) if cell_tag_prediction[i] != 0 and \
            np.isnan(table_numbers[i]) is not True]




class RobertaSelfOutput(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class TagopModel(nn.Module):
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
        super(TagopModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.operator_classes = operator_classes
        self.scale_classes = scale_classes
        if hidden_size is None:
            hidden_size = self.config.hidden_size
        if dropout_prob is None:
            dropout_prob = self.config.hidden_dropout_prob
        # operator predictor
        self.operator_predictor = FFNLayer(hidden_size, hidden_size, operator_classes, dropout_prob)
        # scale predictor
        self.scale_predictor = FFNLayer(3 * hidden_size, hidden_size, scale_classes, dropout_prob)
        # tag predictor: two-class classification
        self.tag_predictor = FFNLayer(hidden_size, hidden_size, 2, dropout_prob)
        # order predictor: three-class classification
        self.order_predictor = FFNLayer(hidden_size, hidden_size, 2, dropout_prob)
        # criterion for operator/scale loss calculation
        self.operator_criterion = operator_criterion
        self.scale_criterion = scale_criterion
        # NLLLoss for tag_prediction
        self.NLLLoss = nn.NLLLoss(reduction="sum")
        # tapas config
        self.config = config

        self.arithmetic_op_index = arithmetic_op_index
        if ablation_mode == 0:
            self.OPERATOR_CLASSES = OPERATOR_CLASSES_
        elif ablation_mode == 1:
            self.OPERATOR_CLASSES = get_op_1(op_mode)
        elif ablation_mode == 2:
            self.OPERATOR_CLASSES = get_op_2(op_mode)
        else:
            self.OPERATOR_CLASSES = get_op_3(op_mode)
        self._metrics = TaTQAEmAndF1()
        
        self.add_structure=False
        if self.add_structure is True:
            self._gcn = GCN(node_dim=hidden_size, iteration_steps=3) 
            self._gcn_enc = ResidualGRU(hidden_size, dropout_prob, 2)
            self._proj_fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self._proj_sequence = nn.Linear(hidden_size, 1, bias=False)
            
        self.add_params=True
        if self.add_params is True:
            self.multihead_attn = nn.ModuleList([MultiheadAttention(hidden_size, 16) for i in range(2)])
#             self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for i in range(2)])
            self.self_output_layer = nn.ModuleList([RobertaSelfOutput(hidden_size, 1e-5, 0.1) for i in range(2)])
            self._gcn_enc = ResidualGRU(hidden_size, dropout_prob, 2)
            self._proj_fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self._proj_sequence = nn.Linear(hidden_size, 1, bias=False)
            
        self.add_row_col_tag = True
        if self.add_row_col_tag:
            self.row_col_tag_predictor = FFNLayer(hidden_size, hidden_size, 2, dropout_prob)
            
        
    """
    :parameter
    input_ids, shape:[bsz, 512] split_tokens' ids, 0 for padded token.
    attention_mask, shape:[bsz, 512] 0 for padded token and 1 for others
    token_type_ids, shape[bsz, 512, 3].
    # row_ids and column_ids are non-zero for table-contents and 0 for others, including headers.
        segment_ids[:, :, 0]: 1 for table and 0 for others
        column_ids[:, :, 1]: indicate to which column of the table a token belongs (starting from 1). Is 0 for all question
      tokens, special tokens and padding.
        row_ids[:, :, 2]: indicate to which row of the table a token belongs (starting from 1). Is 0 for all question tokens,
      special tokens and padding. Tokens of column headers are also 0.
    paragraph_mask, shape[bsz, 512] 1 for paragraph_tokens and 0 for others
    paragraph_index, shape[bsz, 512] 0 for non-paragraph tokens and index starting from 1 for paragraph tokens
    tag_labels: [bsz, 512] 1 for tokens in the answer and 0 for others
    operator_labels: [bsz, 8]
    scale_labels: [bsz, 8]
    number_order_labels: [bsz, 2]
    paragraph_tokens: [bsz, text_len], white-space split tokens
    tables: [bsz,] raw tables in DataFrame.
    paragraph_numbers: [bsz, text_len], corresponding number extracted from tokens, nan for non-number token
    table_numbers: [bsz, table_size], corresponding number extracted from table cells, nan for non-number cell. Shape is the same as flattened table.
    """
    def forward(self,
                input_ids: torch.LongTensor,  # bsz,512
                question_mask:torch.LongTensor,  # bsz,512
                attention_mask: torch.LongTensor,# bsz,512
                token_type_ids: torch.LongTensor,# bsz,512
                paragraph_mask: torch.LongTensor,# bsz,512
                paragraph_index: torch.LongTensor,# bsz,512
                tag_labels: torch.LongTensor,# bsz,512
                operator_labels: torch.LongTensor,# bsz
                scale_labels: torch.LongTensor,# bsz
                number_order_labels: torch.LongTensor, # bsz
                gold_answers: str,  # bsz
                paragraph_tokens: List[List[str]],
                paragraph_numbers: List[np.ndarray],
                table_cell_numbers: List[np.ndarray],
                question_ids: List[str],
                position_ids: torch.LongTensor = None,
                table_mask: torch.LongTensor = None, # bsz,512
                table_cell_index: torch.LongTensor = None, # bsz,512
                table_cell_tokens: List[List[str]] = None,
                
                ## add table structure
                table_row_header_index:torch.LongTensor = None,
                table_col_header_index :torch.LongTensor = None,
                table_data_cell_index :torch.LongTensor = None,
                table_row_header_index_mask :torch.LongTensor = None,
                table_col_header_index_mask :torch.LongTensor = None,
                table_data_cell_index_mask :torch.LongTensor = None,
                same_row_data_cell_relation :torch.LongTensor = None,
                same_col_data_cell_relation :torch.LongTensor = None,
                data_cell_row_rel :torch.LongTensor = None,
                data_cell_col_rel :torch.LongTensor = None,
                
                row_tags:torch.LongTensor = None,
                row_tag_mask:torch.LongTensor = None,
                col_tags:torch.LongTensor = None,
                col_tag_mask:torch.LongTensor = None,
                row_include_cells:torch.LongTensor = None,
                col_include_cells:torch.LongTensor = None,
                all_cell_index:torch.LongTensor = None,
                all_cell_index_mask:torch.LongTensor = None,
                
                mode=None,
                epoch=None, ) -> Dict[str, torch.Tensor]:
#         import pdb
#         pdb.set_trace()
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)

        sequence_output = outputs[0]
        batch_size = sequence_output.shape[0]

        cls_output = sequence_output[:, 0, :]
        operator_prediction = self.operator_predictor(cls_output)
        
        table_sequence_output = util.replace_masked_values(sequence_output, table_mask.unsqueeze(-1), 0)
        table_sequence_reduce_mean_output = reduce_mean_index_vector(table_sequence_output, table_cell_index)
        ## add structure of table
        ## get the representation of headers and some value node related to header
        if self.add_structure:
            ## bsz, row_num, h
            question_weight = self._proj_sequence(sequence_output).squeeze(-1)
            question_weight = util.masked_softmax(question_weight, question_mask)
            question_node = util.weighted_sum(sequence_output, question_weight)
            
            table_row_header_nodes = torch.gather(table_sequence_reduce_mean_output, dim=1, index = table_row_header_index.unsqueeze(-1).expand(-1, -1, table_sequence_reduce_mean_output.size(-1)))
            ## bsz, col_num, h
            table_col_header_nodes = torch.gather(table_sequence_reduce_mean_output, dim=1, index = table_col_header_index.unsqueeze(-1).expand(-1, -1, table_sequence_reduce_mean_output.size(-1)))
            ## bsz, data_cell_num, h
            table_data_cell_nodes = torch.gather(table_sequence_reduce_mean_output, dim=1, index = table_data_cell_index.unsqueeze(-1).expand(-1, -1, table_sequence_reduce_mean_output.size(-1)))
            
#             import pdb
#             pdb.set_trace()
            table_row_header_nodes,table_col_header_nodes, table_data_cell_nodes \
            =self._gcn(question_node, table_row_header_nodes,table_col_header_nodes, table_data_cell_nodes,
                     table_row_header_index_mask,table_col_header_index_mask, table_data_cell_index_mask,
                     same_row_data_cell_relation, same_col_data_cell_relation, data_cell_row_rel,
                      data_cell_col_rel)
            import pdb
#             pdb.set_trace()
            table_sequence_reduce_mean_output_gcn = torch.zeros((batch_size, table_sequence_reduce_mean_output.size(1) + 1, table_sequence_reduce_mean_output[-1].size(-1)), dtype=torch.float).cuda()
            
            table_cell_index_cat = torch.cat([table_row_header_index,table_col_header_index,table_data_cell_index], dim=1)
            table_cell_nodes_cat = torch.cat([table_row_header_nodes,table_col_header_nodes,table_data_cell_nodes], dim=1)
            table_sequence_reduce_mean_output_gcn.scatter_(1, table_cell_index_cat.unsqueeze(-1).expand(-1, -1,table_sequence_reduce_mean_output.size(-1)), table_cell_nodes_cat)
            table_sequence_reduce_mean_output_gcn = table_sequence_reduce_mean_output_gcn[:,:-1,:]

            table_sequence_output_gcn = torch.gather(table_sequence_reduce_mean_output_gcn, index=table_cell_index.unsqueeze(-1).expand(-1,-1,table_sequence_reduce_mean_output_gcn.size(-1)) , dim=1)
        
            table_sequence_output = self._gcn_enc(self._proj_fc(table_sequence_output+table_sequence_output_gcn))
            table_sequence_reduce_mean_output = reduce_mean_index_vector(table_sequence_output, table_cell_index)
            
        
#         if self.add_params:
#             question_weight = self._proj_sequence(sequence_output).squeeze(-1)
#             question_weight = util.masked_softmax(question_weight, question_mask)
#             question_node = util.weighted_sum(sequence_output, question_weight)
            
#             import pdb
#             pdb.set_trace()
            
#             table_row_header_nodes = torch.gather(table_sequence_reduce_mean_output, dim=1, index = table_row_header_index.unsqueeze(-1).expand(-1, -1, table_sequence_reduce_mean_output.size(-1)))
#             ## bsz, col_num, h
#             table_col_header_nodes = torch.gather(table_sequence_reduce_mean_output, dim=1, index = table_col_header_index.unsqueeze(-1).expand(-1, -1, table_sequence_reduce_mean_output.size(-1)))
#             ## bsz, data_cell_num, h
#             table_data_cell_nodes = torch.gather(table_sequence_reduce_mean_output, dim=1, index = table_data_cell_index.unsqueeze(-1).expand(-1, -1, table_sequence_reduce_mean_output.size(-1)))
            
#             nodes = torch.cat([table_row_header_nodes,table_col_header_nodes,table_data_cell_nodes], dim=1)
#             attention_mask = torch.cat([table_row_header_index_mask,table_col_header_index_mask, table_data_cell_index_mask], dim=1)
            
#             attention_mask_repeat = attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(1)
#             attention_mask_repeat = attention_mask_repeat.repeat(16,1,1).float()
#             import pdb
# #             pdb.set_trace()
            
#             nodes_tmp = nodes
#             for id, layer in enumerate(self.multihead_attn):
#                 query = nodes_tmp.transpose(dim0=1, dim1=0)
#                 key = nodes_tmp.transpose(dim0=1, dim1=0)
#                 value = nodes_tmp.transpose(dim0=1, dim1=0)
#                 attn_output, _ = layer(query=query, key=key, value=value, attn_mask=attention_mask_repeat)

# #                 nodes_tmp += attn_output.transpose(dim0=1, dim1=0)
# #                 nodes_tmp = self.layer_norm[id](nodes_tmp)
# #                 pdb.set_trace()
#                 nodes_tmp = self.self_output_layer[id](attn_output.transpose(dim0=1, dim1=0), nodes_tmp)
                
    
# #             pdb.set_trace()
#             nodes_tmp = util.replace_masked_values(nodes_tmp, attention_mask.unsqueeze(-1), 0)
            
#             table_sequence_reduce_mean_output_attention = torch.zeros((batch_size, table_sequence_reduce_mean_output.size(1) + 1, table_sequence_reduce_mean_output[-1].size(-1)), dtype=torch.float).cuda()
#             table_cell_index_cat = torch.cat([table_row_header_index,table_col_header_index,table_data_cell_index], dim=1)
            
#             table_sequence_reduce_mean_output_attention.scatter_(1, table_cell_index_cat.unsqueeze(-1).expand(-1, -1,table_sequence_reduce_mean_output.size(-1)), nodes_tmp)
#             table_sequence_reduce_mean_output_attention = table_sequence_reduce_mean_output_attention[:,:-1,:]

#             table_sequence_output_attention = torch.gather(table_sequence_reduce_mean_output_attention, index=table_cell_index.unsqueeze(-1).expand(-1,-1,table_sequence_reduce_mean_output_attention.size(-1)) , dim=1)
        
#             table_sequence_output = self._gcn_enc(self._proj_fc(table_sequence_output+table_sequence_output_attention))
#             table_sequence_reduce_mean_output = reduce_mean_index_vector(table_sequence_output, table_cell_index)
            
        
        if self.add_params:
            import pdb

            question_weight = self._proj_sequence(sequence_output).squeeze(-1)
            question_weight = util.masked_softmax(question_weight, question_mask)
            question_node = util.weighted_sum(sequence_output, question_weight)
            
            all_cell_nodes = torch.gather(table_sequence_reduce_mean_output, dim=1, index =all_cell_index.unsqueeze(-1).expand(-1, -1, table_sequence_reduce_mean_output.size(-1)))
            all_cell_self_attention_mask_repeat = all_cell_index_mask.unsqueeze(-1) * all_cell_index_mask.unsqueeze(1)
            all_cell_self_attention_mask_repeat = all_cell_self_attention_mask_repeat.repeat(16,1,1).float()
            
            nodes_tmp = all_cell_nodes
            for id, layer in enumerate(self.multihead_attn):
                query = nodes_tmp.transpose(dim0=1, dim1=0)
                key = nodes_tmp.transpose(dim0=1, dim1=0)
                value = nodes_tmp.transpose(dim0=1, dim1=0)
                attn_output, _ = layer(query=query, key=key, value=value, attn_mask=all_cell_self_attention_mask_repeat)
                nodes_tmp = self.self_output_layer[id](attn_output.transpose(dim0=1, dim1=0), nodes_tmp)
#             pdb.set_trace()
            nodes_tmp = util.replace_masked_values(nodes_tmp, all_cell_index_mask.unsqueeze(-1), 0)
            
            if self.add_row_col_tag:
                ## add row representation
                row_state = torch.matmul(row_include_cells.float(), nodes_tmp)
                col_state = torch.matmul(col_include_cells.float(), nodes_tmp)
                ## add col_representation
                row_tag_prediction = self.row_col_tag_predictor(row_state)
                row_tag_prediction_prob = util.masked_softmax(row_tag_prediction, mask=None)
                row_tag_prediction = util.masked_log_softmax(row_tag_prediction, mask=None)
                row_tag_prediction = util.replace_masked_values(row_tag_prediction, row_tag_mask.unsqueeze(-1), 0) 
                row_tag_prediction_prob = util.replace_masked_values(row_tag_prediction_prob, row_tag_mask.unsqueeze(-1), 0)
                row_tag_prediction_prob = row_tag_prediction_prob.transpose(1,2)
               
            
                row_tag_label = util.replace_masked_values(row_tags.float(), row_tag_mask, 0)
                row_tag_prediction = row_tag_prediction.transpose(1, 2)  # [bsz, 2, table_size]
                row_tag_prediction_loss = self.NLLLoss(row_tag_prediction, row_tag_label.long())

                col_tag_prediction = self.row_col_tag_predictor(col_state)
                col_tag_prediction_prob = util.masked_softmax(col_tag_prediction, mask=None)       
                col_tag_prediction = util.masked_log_softmax(col_tag_prediction, mask=None)
                col_tag_prediction = util.replace_masked_values(col_tag_prediction, col_tag_mask.unsqueeze(-1), 0) 
                col_tag_prediction_prob = util.replace_masked_values(col_tag_prediction_prob, col_tag_mask.unsqueeze(-1), 0)
                col_tag_prediction_prob = col_tag_prediction_prob.transpose(1,2)
                
                
                col_tag_label = util.replace_masked_values(col_tags.float(), col_tag_mask, 0)
                col_tag_prediction = col_tag_prediction.transpose(1, 2)  # [bsz, 2, table_size]
                col_tag_prediction_loss = self.NLLLoss(col_tag_prediction, col_tag_label.long())

                
            table_sequence_reduce_mean_output_attention = torch.zeros((batch_size, table_sequence_reduce_mean_output.size(1) + 1, table_sequence_reduce_mean_output[-1].size(-1)), dtype=torch.float).cuda()
            
            table_sequence_reduce_mean_output_attention.scatter_(1, all_cell_index.unsqueeze(-1).expand(-1, -1,table_sequence_reduce_mean_output.size(-1)), nodes_tmp)
            table_sequence_reduce_mean_output_attention = table_sequence_reduce_mean_output_attention[:,:-1,:]

            table_sequence_output_attention = torch.gather(table_sequence_reduce_mean_output_attention, index=table_cell_index.unsqueeze(-1).expand(-1,-1,table_sequence_reduce_mean_output_attention.size(-1)) , dim=1)
        
            table_sequence_output = self._gcn_enc(self._proj_fc(table_sequence_output+table_sequence_output_attention))
            table_sequence_reduce_mean_output = reduce_mean_index_vector(table_sequence_output, table_cell_index)
        
        
        table_tag_input = table_sequence_output
        table_tag_prediction = self.tag_predictor(table_tag_input) #bsz, 512,2
        
        table_tag_prediction_prob =  util.masked_softmax(table_tag_prediction, mask=None)
        table_tag_prediction_prob = util.replace_masked_values(table_tag_prediction_prob, table_mask.unsqueeze(-1), 0) 
        
        table_tag_prediction = util.masked_log_softmax(table_tag_prediction, mask=None)
        table_tag_prediction = util.replace_masked_values(table_tag_prediction, table_mask.unsqueeze(-1), 0) 
        table_tag_labels = util.replace_masked_values(tag_labels.float(), table_mask, 0)

        paragraph_sequence_output = util.replace_masked_values(sequence_output, paragraph_mask.unsqueeze(-1), 0)
        paragraph_tag_input = torch.cat([question_node.unsqueeze(1).expand(-1, paragraph_sequence_output.size(1),-1), paragraph_sequence_output], dim=-1)
        paragraph_tag_input = paragraph_sequence_output
        paragraph_tag_prediction = self.tag_predictor(paragraph_tag_input)
        paragraph_tag_prediction = util.masked_log_softmax(paragraph_tag_prediction, mask=None)
        paragraph_tag_prediction = util.replace_masked_values(paragraph_tag_prediction, paragraph_mask.unsqueeze(-1), 0)
        paragraph_tag_labels = util.replace_masked_values(tag_labels.float(), paragraph_mask, 0)

        paragraph_reduce_mean = torch.mean(paragraph_sequence_output, dim=1)
        table_reduce_mean = torch.mean(table_sequence_output, dim=1)
        cls_output = torch.cat((cls_output, table_reduce_mean, paragraph_reduce_mean), dim=-1)
        scale_prediction = self.scale_predictor(cls_output)
        
        device = input_ids.device
        device_id = device.index
        
        ## cell-level tag prediction and representation
        table_tag_reduce_max_prediction, _ = \
            reduce_max_index_get_vector(table_tag_prediction[:, :, 1], table_sequence_output, table_cell_index)
    
        ## word level tag prediction and prepresentation
        paragraph_tag_reduce_max_prediction, _ = \
            reduce_max_index_get_vector(paragraph_tag_prediction[:, :, 1], paragraph_sequence_output, paragraph_index)
        paragraph_sequence_reduce_mean_output = reduce_mean_index_vector(paragraph_sequence_output, paragraph_index)
        
        
        
        ## cell-level mask
        table_reduce_mask = reduce_mean_index(table_mask, table_cell_index)
        ## word-level mask
        paragraph_reduce_mask = reduce_mean_index(paragraph_mask, paragraph_index)
        
        ## cell-level masked tag prediction
        masked_table_tag_reduce_max_prediction = util.replace_masked_values(table_tag_reduce_max_prediction,
                                                                            table_reduce_mask,
                                                                            -1e+5)
        ## word-level masked tag prediction
        masked_paragraph_tag_reduce_max_prediction = util.replace_masked_values(paragraph_tag_reduce_max_prediction,
                                                                                paragraph_reduce_mask,
                                                                                -1e+5)
        ## cell sort
        sorted_table_tag_prediction, sorted_cell_index = torch.sort(masked_table_tag_reduce_max_prediction,
                                                                    dim=-1, descending=True)
        ## word sort
        sorted_paragraph_tag_prediction, sorted_paragraph_index = torch.sort(masked_paragraph_tag_reduce_max_prediction, dim=-1, descending=True)
        
        ## top2 table and passage
        sorted_table_tag_prediction = sorted_table_tag_prediction[:, :2]
        sorted_cell_index = sorted_cell_index[:, :2]
        sorted_paragraph_tag_prediction = sorted_paragraph_tag_prediction[:, :2]
        sorted_paragraph_index = sorted_paragraph_index[:, :2]
        
        ## top2 cell in table and word passage sort
        concat_tag_prediction = torch.cat((sorted_paragraph_tag_prediction, sorted_table_tag_prediction),
                                          dim=1)
        _, sorted_concat_tag_index = torch.sort(concat_tag_prediction, dim=-1, descending=True)
        
        
        predicted_operator_class = torch.argmax(operator_prediction, dim=-1)
        top_2_number = np.zeros((batch_size, 2))
        
        top_2_order_ground_truth = torch.zeros(batch_size).to(device)
        top_2_sequence_output = torch.zeros(batch_size, 2, sequence_output.shape[2]).to(device)
        top_2_sequence_output_bw = torch.zeros(batch_size, 2, sequence_output.shape[2]).to(device)

        number_index = 0
        ground_truth_index = 0

        for bsz in range(batch_size):
            if ("DIVIDE" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["DIVIDE"]) or \
                    ("DIFF" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["DIFF"]) or \
                    ("CHANGE_RATIO" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["CHANGE_RATIO"]):
                
                _index = sorted_concat_tag_index[bsz]
                ## top2 number and corresponding output 
                if _index[0] > 1:
                    top_2_number[number_index, 0] = table_cell_numbers[bsz+device_id*batch_size][sorted_cell_index[bsz, _index[0] - 2] - 1]
                    top_2_sequence_output[number_index, 0, :] = table_sequence_reduce_mean_output[bsz,
                                                                sorted_cell_index[bsz, _index[0] - 2], :]
                else:
                    top_2_number[number_index, 0] = paragraph_numbers[bsz+device_id*batch_size][sorted_paragraph_index[bsz, _index[0]] - 1]
                    top_2_sequence_output[number_index, 0, :] = paragraph_sequence_reduce_mean_output[bsz,
                                                                sorted_paragraph_index[bsz, _index[0]], :]
                if _index[1] > 1:
                    top_2_number[number_index, 1] = table_cell_numbers[bsz+device_id*batch_size][sorted_cell_index[bsz, _index[1] - 2] - 1]
                    top_2_sequence_output[number_index, 1, :] = table_sequence_reduce_mean_output[bsz,
                                                                sorted_cell_index[bsz, _index[1] - 2], :]
                else:
                    top_2_number[number_index, 1] = paragraph_numbers[bsz+device_id*batch_size][sorted_paragraph_index[bsz, _index[1]] - 1]
                    top_2_sequence_output[number_index, 1, :] = paragraph_sequence_reduce_mean_output[bsz,
                                                                sorted_paragraph_index[bsz, _index[1]], :] 
                
                number_index += 1

                
                if ("DIVIDE" not in self.OPERATOR_CLASSES or operator_labels[bsz] != self.OPERATOR_CLASSES["DIVIDE"]) and \
                        ("DIFF" not in self.OPERATOR_CLASSES or operator_labels[bsz] != self.OPERATOR_CLASSES["DIFF"]) and \
                        ("CHANGE_RATIO" not in self.OPERATOR_CLASSES or operator_labels[bsz] != self.OPERATOR_CLASSES["CHANGE_RATIO"]):
                    continue
                    
                top_2_order_ground_truth[ground_truth_index] = number_order_labels[bsz]
                if _index[0] > 1:
                    top_2_sequence_output_bw[ground_truth_index, 0, :] = table_sequence_reduce_mean_output[bsz,
                                                                         sorted_cell_index[bsz, _index[0] - 2], :]
                else:
                    top_2_sequence_output_bw[ground_truth_index, 0, :] = paragraph_sequence_reduce_mean_output[bsz,
                                                                         sorted_paragraph_index[bsz, _index[0]], :]
                if _index[1] > 1:
                    top_2_sequence_output_bw[ground_truth_index, 1, :] = table_sequence_reduce_mean_output[bsz,
                                                                         sorted_cell_index[bsz, _index[1] - 2], :]
                else:
                    top_2_sequence_output_bw[ground_truth_index, 1, :] = paragraph_sequence_reduce_mean_output[bsz,
                                                                         sorted_paragraph_index[bsz, _index[1]], :]
                ground_truth_index += 1
        
        import pdb
#         pdb.set_trace()
        top_2_order_prediction = self.order_predictor(torch.mean(top_2_sequence_output[:number_index], dim=1))
        top_2_number = top_2_number[:number_index]
        top_2_order_prediction_bw = self.order_predictor(
            torch.mean(top_2_sequence_output_bw[:ground_truth_index], dim=1))
        top_2_order_ground_truth = top_2_order_ground_truth[:ground_truth_index]

        # down, calculate the loss
        output_dict = {}

        operator_prediction_loss = self.operator_criterion(operator_prediction, operator_labels)
        scale_prediction_loss = self.scale_criterion(scale_prediction, scale_labels)

        table_tag_prediction = table_tag_prediction.transpose(1, 2)  # [bsz, 2, table_size]
        table_tag_prediction_loss = self.NLLLoss(table_tag_prediction, table_tag_labels.long())
        paragraph_tag_prediction = paragraph_tag_prediction.transpose(1, 2)
        paragraph_token_tag_prediction_loss = self.NLLLoss(paragraph_tag_prediction, paragraph_tag_labels.long())

        if ground_truth_index != 0:
            top_2_order_prediction_bw = util.masked_log_softmax(top_2_order_prediction_bw, mask=None)
            top_2_order_prediction_loss = self.NLLLoss(top_2_order_prediction_bw, top_2_order_ground_truth.long())
        else:
            top_2_order_prediction_loss = torch.tensor(0, dtype=torch.float).to(device)

        
        
        
        output_dict["loss"] = operator_prediction_loss + scale_prediction_loss + \
                              table_tag_prediction_loss + \
                              paragraph_token_tag_prediction_loss + \
                              top_2_order_prediction_loss
        
        if self.add_row_col_tag:
            output_dict["loss"]+=row_tag_prediction_loss + col_tag_prediction_loss
        
        with torch.no_grad():
            output_dict["question_id"] = []
            output_dict["answer"] = []
            output_dict["scale"] = []
            # paragraph_tag_prediction_ = np.argmax(paragraph_tag_prediction_, axis=1)
            
            paragraph_tag_prediction = paragraph_tag_prediction.transpose(1, 2)
            paragraph_tag_prediction_score = paragraph_tag_prediction[:, :, 1]
            paragraph_tag_prediction = torch.argmax(paragraph_tag_prediction, dim=-1).float()
            
            ## word-level tag prediction and score
            paragraph_token_tag_prediction = reduce_mean_index(paragraph_tag_prediction, paragraph_index)
            paragraph_token_tag_prediction_score = reduce_max_index(paragraph_tag_prediction_score, paragraph_index)
            paragraph_token_tag_prediction = paragraph_token_tag_prediction.detach().cpu().numpy()
            paragraph_token_tag_prediction_score = paragraph_token_tag_prediction_score.detach().cpu().numpy()
            
            predicted_scale_class = torch.argmax(scale_prediction, dim=-1).detach().cpu().numpy()
            predicted_operator_class = predicted_operator_class.detach().cpu().numpy()
            
            ## row/ cell
            if self.add_row_col_tag:
                            
                
                
                table_cell_tag_prediction_prob_with_row = torch.matmul(row_tag_prediction_prob, row_include_cells.float()) ## bsz, 2, 512
                table_cell_tag_prediction_prob_with_row = table_cell_tag_prediction_prob_with_row.transpose(1,2)
                table_cell_tag_prediction_prob_with_row_new = torch.zeros(batch_size, 512, 2).cuda()
                table_cell_tag_prediction_prob_with_row_new[:,1:,:] = table_cell_tag_prediction_prob_with_row[:,:-1,:]
                table_tag_prediction_prob_with_row = torch.gather(table_cell_tag_prediction_prob_with_row_new, index=table_cell_index.unsqueeze(-1).expand(-1,-1,table_cell_tag_prediction_prob_with_row.size(-1)), dim=1)


                table_cell_tag_prediction_prob_with_col = torch.matmul(col_tag_prediction_prob, col_include_cells.float()) ## bsz, 2, 512
                table_cell_tag_prediction_prob_with_col = table_cell_tag_prediction_prob_with_col.transpose(1,2)
                table_cell_tag_prediction_prob_with_col_new = torch.zeros(batch_size, 512, 2).cuda()
                table_cell_tag_prediction_prob_with_col_new[:,1:,:] = table_cell_tag_prediction_prob_with_col[:,:-1,:]
                table_tag_prediction_prob_with_col = torch.gather(table_cell_tag_prediction_prob_with_col_new, index=table_cell_index.unsqueeze(-1).expand(-1,-1,table_cell_tag_prediction_prob_with_col.size(-1)), dim=1)
               
                table_tag_prediction_prob += table_tag_prediction_prob_with_row + table_tag_prediction_prob_with_col
                table_tag_prediction_prob = table_tag_prediction_prob/3

                table_tag_prediction_score = table_tag_prediction_prob[:, :, 1]   ## bsz, 512
                table_tag_prediction_prob = torch.argmax(table_tag_prediction_prob, dim=-1).float() ## bsz, 512
    #             cell-level tag prediction and score
                table_cell_tag_prediction = reduce_mean_index(table_tag_prediction_prob, table_cell_index)
                table_cell_tag_prediction_score = reduce_max_index(table_tag_prediction_score, table_cell_index) ## bsz, 512
                table_cell_tag_prediction = table_cell_tag_prediction.detach().cpu().numpy()
                table_cell_tag_prediction_score = table_cell_tag_prediction_score.detach().cpu().numpy()

    #                 row_tag_prediction_score = row_tag_prediction[:, :, 1]
    #                 row_tag_prediction = torch.argmax(row_tag_prediction, dim=-1).float()
    #                 table_tag_prediction_score_with_row = torch.matmul(row_tag_prediction_score.unsqueeze(1), row_include_cells.float()).squeeze(1)
    #                 table_tag_prediction_score_with_row_new = torch.zeros(batch_size, 513)
    #                 table_tag_prediction_score_with_row_new[:,1:] = table_tag_prediction_score_with_row
    #                 table_tag_prediction_score_with_row_new = table_tag_prediction_score_with_row_new[:,:-1].detach().cpu().numpy()

    #                 col_tag_prediction = col_tag_prediction.transpose(1, 2)
    #                 col_tag_prediction_score = col_tag_prediction[:, :, 1]
    #                 col_tag_prediction = torch.argmax(col_tag_prediction, dim=-1).float()
    #                 table_tag_prediction_score_with_col= torch.matmul(col_tag_prediction_score.unsqueeze(1).float(), col_include_cells.float()).squeeze(1)
    #                 table_tag_prediction_score_with_col_new = torch.zeros(batch_size, 513)
    #                 table_tag_prediction_score_with_col_new[:,1:] = table_tag_prediction_score_with_col
    #                 table_tag_prediction_score_with_col_new = table_tag_prediction_score_with_col_new[:,:-1].detach().cpu().numpy()

    #                 table_cell_tag_prediction_score += table_tag_prediction_score_with_row_new + table_tag_prediction_score_with_col_new
            else:
                table_tag_prediction = table_tag_prediction.transpose(1, 2)
                table_tag_prediction_score = table_tag_prediction[:, :, 1]   ## bsz, 512
                table_tag_prediction = torch.argmax(table_tag_prediction, dim=-1).float() ## bsz, 512
    #             cell-level tag prediction and score
                table_cell_tag_prediction = reduce_mean_index(table_tag_prediction, table_cell_index)
                table_cell_tag_prediction_score = reduce_max_index(table_tag_prediction_score, table_cell_index) ## bsz, 512
                table_cell_tag_prediction = table_cell_tag_prediction.detach().cpu().numpy()
                table_cell_tag_prediction_score = table_cell_tag_prediction_score.detach().cpu().numpy()
                
            top_2_index = 0
            if number_index != 0:
                top_2_order_prediction = top_2_order_prediction.detach().cpu().numpy()
                top_2_order_prediction = np.argmax(top_2_order_prediction, axis=1)
                
            for bsz in range(batch_size):
                # test_reduce_mean(paragraph_tag_prediction[bsz], paragraph_index[bsz])
                if "SPAN-TEXT" in self.OPERATOR_CLASSES and \
                        predicted_operator_class[bsz] == self.OPERATOR_CLASSES["SPAN-TEXT"]:
                    paragraph_selected_span_tokens = get_single_span_tokens_from_paragraph(
                        paragraph_token_tag_prediction[bsz],
                        paragraph_token_tag_prediction_score[bsz],
                        paragraph_tokens[bsz+batch_size*device_id]
                    )
                    answer = paragraph_selected_span_tokens
                    answer = sorted(answer)
                elif "SPAN-TABLE" in self.OPERATOR_CLASSES and \
                        predicted_operator_class[bsz] == self.OPERATOR_CLASSES["SPAN-TABLE"]:
#                     pdb.set_trace()
                    table_selected_tokens = get_single_span_tokens_from_table(
                        table_cell_tag_prediction[bsz],
                        table_cell_tag_prediction_score[bsz],
                        table_cell_tokens[bsz+batch_size*device_id])
                    answer = table_selected_tokens
                    answer = sorted(answer)
                elif "MULTI_SPAN" in self.OPERATOR_CLASSES and \
                        predicted_operator_class[bsz] == self.OPERATOR_CLASSES["MULTI_SPAN"]:
                    paragraph_selected_span_tokens = \
                        get_span_tokens_from_paragraph(paragraph_token_tag_prediction[bsz], paragraph_tokens[bsz+batch_size*device_id])
                    table_selected_tokens = \
                        get_span_tokens_from_table(table_cell_tag_prediction[bsz], table_cell_tokens[bsz+batch_size*device_id])
                    answer = paragraph_selected_span_tokens + table_selected_tokens
                    answer = sorted(answer)
                elif "COUNT" in self.OPERATOR_CLASSES and \
                        predicted_operator_class[bsz] == self.OPERATOR_CLASSES["COUNT"]:
                    paragraph_selected_tokens = \
                        get_span_tokens_from_paragraph(paragraph_token_tag_prediction[bsz], paragraph_tokens[bsz+batch_size*device_id])
                    table_selected_tokens = \
                        get_span_tokens_from_table(table_cell_tag_prediction[bsz], table_cell_tokens[bsz+batch_size*device_id])
                    answer = len(paragraph_selected_tokens) + len(table_selected_tokens)
                else:
                    if ("SUM" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["SUM"]) \
                            or ("TIMES" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["TIMES"]) \
                            or ("AVERAGE" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["AVERAGE"]):
                        ## return all the number in paragraph labeled evidence
                        paragraph_selected_numbers = \
                            get_numbers_from_reduce_sequence(paragraph_token_tag_prediction[bsz],
                                                             paragraph_numbers[bsz+batch_size*device_id])
                        ## return all the number in table labeled evidence
                        table_selected_numbers = \
                            get_numbers_from_reduce_sequence(table_cell_tag_prediction[bsz], table_cell_numbers[bsz+batch_size*device_id])
                        selected_numbers = paragraph_selected_numbers + table_selected_numbers
                        if not selected_numbers:
                            answer = ""
                        elif "SUM" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["SUM"]:
                            answer = np.around(np.sum(selected_numbers), 4)
                        elif "TIMES" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["TIMES"]:
                            answer = np.around(np.prod(selected_numbers), 4)
                        elif "AVERAGE" in self.OPERATOR_CLASSES:
                            answer = np.around(np.mean(selected_numbers), 4)
                    else:
                        if top_2_number.size <= 0:
                            answer = ""
                        if top_2_number.size > 0:
                            operand_one = top_2_number[top_2_index, 0]
                            operand_two = top_2_number[top_2_index, 1]
                            predicted_order = top_2_order_prediction[top_2_index]
                            if np.isnan(operand_one) or np.isnan(operand_two):
                                answer = ""
                            else:
                                if "DIFF" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["DIFF"]:
                                    if predicted_order == 0:
                                        answer = np.around(operand_one - operand_two, 4)
                                    else:
                                        answer = np.around(operand_two - operand_one, 4)
                                elif "DIVIDE" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["DIVIDE"]:
                                    if predicted_order == 0:
                                        answer = np.around(operand_one / operand_two, 4)
                                    else:
                                        answer = np.around(operand_two / operand_one, 4)
                                    if SCALE[int(predicted_scale_class[bsz])] == "percent":
                                        answer = answer * 100
                                elif "CHANGE_RATIO" in self.OPERATOR_CLASSES:
                                    if predicted_order == 0:
                                        answer = np.around(operand_one / operand_two - 1, 4)
                                    else:
                                        answer = np.around(operand_two / operand_one - 1, 4)
                                    if SCALE[int(predicted_scale_class[bsz])] == "percent":
                                        answer = answer * 100
                                    
                                    
                            top_2_index += 1

#                 output_dict["answer"].append(answer)
                output_dict["scale"].append(SCALE[int(predicted_scale_class[bsz])])
                output_dict["question_id"].append(question_ids[bsz])
                if predicted_operator_class[bsz] in self.arithmetic_op_index:
                    predict_type = "arithmetic"
                else:
                    predict_type = ""
                self._metrics(gold_answers[bsz+batch_size*device_id], answer, SCALE[int(predicted_scale_class[bsz])])

        return output_dict

    def predict(self,
                input_ids,
                question_mask,                
                attention_mask,
                token_type_ids,
                paragraph_mask,
                paragraph_index,
                tag_labels,
                gold_answers,
                paragraph_tokens,
                paragraph_numbers,
                table_cell_numbers,
                question_ids,
                position_ids=None,
                mode=None,
                epoch=None,
                table_mask=None,
                table_cell_index=None,
                table_cell_tokens=None,
                table_mapping_content=None,
                paragraph_mapping_content=None,
                
                 ## add table structure
                table_row_header_index:torch.LongTensor = None,
                table_col_header_index :torch.LongTensor = None,
                table_data_cell_index :torch.LongTensor = None,
                table_row_header_index_mask :torch.LongTensor = None,
                table_col_header_index_mask :torch.LongTensor = None,
                table_data_cell_index_mask :torch.LongTensor = None,
                same_row_data_cell_relation :torch.LongTensor = None,
                same_col_data_cell_relation :torch.LongTensor = None,
                data_cell_row_rel :torch.LongTensor = None,
                data_cell_col_rel :torch.LongTensor = None,
                table_cell_number_value_with_time=None,
                
                row_tags:torch.LongTensor = None,
                row_tag_mask:torch.LongTensor = None,
                col_tags:torch.LongTensor = None,
                col_tag_mask:torch.LongTensor = None,
                row_include_cells:torch.LongTensor = None,
                col_include_cells:torch.LongTensor = None,
                all_cell_index:torch.LongTensor = None,
                all_cell_index_mask:torch.LongTensor = None,
                
                ):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        sequence_output = outputs[0]
        batch_size = sequence_output.shape[0]
        
        cls_output = sequence_output[:, 0, :]
        operator_prediction = self.operator_predictor(cls_output)

        ## add structure of table
        ## get the representation of headers and some value node related to header
        table_sequence_output = util.replace_masked_values(sequence_output, table_mask.unsqueeze(-1), 0)
        table_sequence_reduce_mean_output = reduce_mean_index_vector(table_sequence_output, table_cell_index)
        
        if self.add_structure:
            ## bsz, row_num, h
            question_weight = self._proj_sequence(sequence_output).squeeze(-1)
            question_weight = util.masked_softmax(question_weight, question_mask)
            question_node = util.weighted_sum(sequence_output, question_weight)
            
            table_row_header_nodes = torch.gather(table_sequence_reduce_mean_output, dim=1, index = table_row_header_index.unsqueeze(-1).expand(-1, -1, table_sequence_reduce_mean_output.size(-1)))
            ## bsz, col_num, h
            table_col_header_nodes = torch.gather(table_sequence_reduce_mean_output, dim=1, index = table_col_header_index.unsqueeze(-1).expand(-1, -1, table_sequence_reduce_mean_output.size(-1)))
            ## bsz, data_cell_num, h
            table_data_cell_nodes = torch.gather(table_sequence_reduce_mean_output, dim=1, index = table_data_cell_index.unsqueeze(-1).expand(-1, -1, table_sequence_reduce_mean_output.size(-1)))
            
#             import pdb
#             pdb.set_trace()
            table_row_header_nodes,table_col_header_nodes, table_data_cell_nodes \
            =self._gcn(question_node, table_row_header_nodes,table_col_header_nodes, table_data_cell_nodes,
                     table_row_header_index_mask,table_col_header_index_mask, table_data_cell_index_mask,
                     same_row_data_cell_relation, same_col_data_cell_relation, data_cell_row_rel,
                      data_cell_col_rel)
        
            table_sequence_reduce_mean_output_gcn = torch.zeros((batch_size, table_sequence_reduce_mean_output.size(1) + 1, table_sequence_reduce_mean_output[-1].size(-1)), dtype=torch.float).cuda()
            
            table_cell_index_cat = torch.cat([table_row_header_index,table_col_header_index,table_data_cell_index], dim=1)
            table_cell_nodes_cat = torch.cat([table_row_header_nodes,table_col_header_nodes,table_data_cell_nodes], dim=1)
            table_sequence_reduce_mean_output_gcn.scatter_(1, table_cell_index_cat.unsqueeze(-1).expand(-1, -1,table_sequence_reduce_mean_output.size(-1)), table_cell_nodes_cat)
            table_sequence_reduce_mean_output_gcn = table_sequence_reduce_mean_output_gcn[:,:-1,:]

            table_sequence_output_gcn = torch.gather(table_sequence_reduce_mean_output_gcn, index=table_cell_index.unsqueeze(-1).expand(-1,-1,table_sequence_reduce_mean_output_gcn.size(-1)) , dim=1)
        
            table_sequence_output = self._gcn_enc(self._proj_fc(table_sequence_output+table_sequence_output_gcn))
            table_sequence_reduce_mean_output = reduce_mean_index_vector(table_sequence_output, table_cell_index)
        

#         if self.add_params:
#             question_weight = self._proj_sequence(sequence_output).squeeze(-1)
#             question_weight = util.masked_softmax(question_weight, question_mask)
#             question_node = util.weighted_sum(sequence_output, question_weight)
            
#             table_row_header_nodes = torch.gather(table_sequence_reduce_mean_output, dim=1, index = table_row_header_index.unsqueeze(-1).expand(-1, -1, table_sequence_reduce_mean_output.size(-1)))
#             ## bsz, col_num, h
#             table_col_header_nodes = torch.gather(table_sequence_reduce_mean_output, dim=1, index = table_col_header_index.unsqueeze(-1).expand(-1, -1, table_sequence_reduce_mean_output.size(-1)))
#             ## bsz, data_cell_num, h
#             table_data_cell_nodes = torch.gather(table_sequence_reduce_mean_output, dim=1, index = table_data_cell_index.unsqueeze(-1).expand(-1, -1, table_sequence_reduce_mean_output.size(-1)))
            
#             nodes = torch.cat([table_row_header_nodes,table_col_header_nodes,table_data_cell_nodes], dim=1)
#             attention_mask = torch.cat([table_row_header_index_mask,table_col_header_index_mask, table_data_cell_index_mask], dim=1)
            
#             attention_mask_repeat = attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(1)
#             attention_mask_repeat = attention_mask_repeat.repeat(16,1,1).float()
#             import pdb
# #             pdb.set_trace()
            
#             nodes_tmp = nodes
#             for id, layer in enumerate(self.multihead_attn):
#                 query = nodes_tmp.transpose(dim0=1, dim1=0)
#                 key = nodes_tmp.transpose(dim0=1, dim1=0)
#                 value = nodes_tmp.transpose(dim0=1, dim1=0)
#                 attn_output, _ = layer(query=query, key=key, value=value, attn_mask=attention_mask_repeat)
# #                 import pdb
# #                 pdb.set_trace()
# #                 nodes_tmp += attn_output.transpose(dim0=1, dim1=0)
                
# # #                 pdb.set_trace()
# # #                 if id == 0:
# # #                     nodes_tmp = F.relu(nodes_tmp) + nodes
# # #                 else:
# # #                     nodes_tmp = F.relu(nodes_tmp)
# # #                 pdb.set_trace()
# #                 nodes_tmp = self.layer_norm[id](nodes_tmp)
#                 nodes_tmp = self.self_output_layer[id](attn_output.transpose(dim0=1, dim1=0), nodes_tmp)
                
    
# #             pdb.set_trace()
#             nodes_tmp = util.replace_masked_values(nodes_tmp, attention_mask.unsqueeze(-1), 0)
            
#             table_sequence_reduce_mean_output_attention = torch.zeros((batch_size, table_sequence_reduce_mean_output.size(1) + 1, table_sequence_reduce_mean_output[-1].size(-1)), dtype=torch.float).cuda()
#             table_cell_index_cat = torch.cat([table_row_header_index,table_col_header_index,table_data_cell_index], dim=1)
            
#             table_sequence_reduce_mean_output_attention.scatter_(1, table_cell_index_cat.unsqueeze(-1).expand(-1, -1,table_sequence_reduce_mean_output.size(-1)), nodes_tmp)
#             table_sequence_reduce_mean_output_attention = table_sequence_reduce_mean_output_attention[:,:-1,:]

#             table_sequence_output_attention = torch.gather(table_sequence_reduce_mean_output_attention, index=table_cell_index.unsqueeze(-1).expand(-1,-1,table_sequence_reduce_mean_output_attention.size(-1)) , dim=1)
        
#             table_sequence_output = self._gcn_enc(self._proj_fc(table_sequence_output+table_sequence_output_attention))
#             table_sequence_reduce_mean_output = reduce_mean_index_vector(table_sequence_output, table_cell_index)
            
        
#        table_tag_input=torch.cat([question_node.unsqueeze(1).expand(-1, table_sequence_output.size(1),-1),table_sequence_output], dim=-1)

        if self.add_params:
            import pdb

            question_weight = self._proj_sequence(sequence_output).squeeze(-1)
            question_weight = util.masked_softmax(question_weight, question_mask)
            question_node = util.weighted_sum(sequence_output, question_weight)
            
            all_cell_nodes = torch.gather(table_sequence_reduce_mean_output, dim=1, index =all_cell_index.unsqueeze(-1).expand(-1, -1, table_sequence_reduce_mean_output.size(-1)))
            all_cell_self_attention_mask_repeat = all_cell_index_mask.unsqueeze(-1) * all_cell_index_mask.unsqueeze(1)
            all_cell_self_attention_mask_repeat = all_cell_self_attention_mask_repeat.repeat(16,1,1).float()
            
            nodes_tmp = all_cell_nodes
            for id, layer in enumerate(self.multihead_attn):
                query = nodes_tmp.transpose(dim0=1, dim1=0)
                key = nodes_tmp.transpose(dim0=1, dim1=0)
                value = nodes_tmp.transpose(dim0=1, dim1=0)
                attn_output, _ = layer(query=query, key=key, value=value, attn_mask=all_cell_self_attention_mask_repeat)
                nodes_tmp = self.self_output_layer[id](attn_output.transpose(dim0=1, dim1=0), nodes_tmp)
#             pdb.set_trace()
            nodes_tmp = util.replace_masked_values(nodes_tmp, all_cell_index_mask.unsqueeze(-1), 0)
            
            
            if self.add_row_col_tag:
                ## add row representation
                row_state = torch.matmul(row_include_cells.float(), nodes_tmp)
                col_state = torch.matmul(col_include_cells.float(), nodes_tmp)
                ## add col_representation
                row_tag_prediction = self.row_col_tag_predictor(row_state)
                row_tag_prediction_prob = util.masked_softmax(row_tag_prediction, mask=None)
                row_tag_prediction = util.masked_log_softmax(row_tag_prediction, mask=None)
                row_tag_prediction = util.replace_masked_values(row_tag_prediction, row_tag_mask.unsqueeze(-1), 0) 
                row_tag_prediction_prob = util.replace_masked_values(row_tag_prediction_prob, row_tag_mask.unsqueeze(-1), 0) 
                row_tag_prediction_prob = row_tag_prediction_prob.transpose(1,2)
                row_tag_label = util.replace_masked_values(row_tags.float(), row_tag_mask, 0)
                row_tag_prediction = row_tag_prediction.transpose(1, 2)  # [bsz, 2, table_size]
                row_tag_prediction_loss = self.NLLLoss(row_tag_prediction, row_tag_label.long())

                col_tag_prediction = self.row_col_tag_predictor(col_state)
                col_tag_prediction_prob = util.masked_softmax(col_tag_prediction, mask=None)
                col_tag_prediction = util.masked_log_softmax(col_tag_prediction, mask=None)
                col_tag_prediction = util.replace_masked_values(col_tag_prediction, col_tag_mask.unsqueeze(-1), 0)
                col_tag_prediction_prob = util.replace_masked_values(col_tag_prediction_prob, col_tag_mask.unsqueeze(-1), 0)
                col_tag_prediction_prob = col_tag_prediction_prob.transpose(1,2)
                col_tag_label = util.replace_masked_values(col_tags.float(), col_tag_mask, 0)
                col_tag_prediction = col_tag_prediction.transpose(1, 2)  # [bsz, 2, table_size]
                col_tag_prediction_loss = self.NLLLoss(col_tag_prediction, col_tag_label.long())
            
        
            table_sequence_reduce_mean_output_attention = torch.zeros((batch_size, table_sequence_reduce_mean_output.size(1) + 1, table_sequence_reduce_mean_output[-1].size(-1)), dtype=torch.float).cuda()
            
            table_sequence_reduce_mean_output_attention.scatter_(1, all_cell_index.unsqueeze(-1).expand(-1, -1,table_sequence_reduce_mean_output.size(-1)), nodes_tmp)
            table_sequence_reduce_mean_output_attention = table_sequence_reduce_mean_output_attention[:,:-1,:]

            table_sequence_output_attention = torch.gather(table_sequence_reduce_mean_output_attention, index=table_cell_index.unsqueeze(-1).expand(-1,-1,table_sequence_reduce_mean_output_attention.size(-1)) , dim=1)
        
            table_sequence_output = self._gcn_enc(self._proj_fc(table_sequence_output+table_sequence_output_attention))
            table_sequence_reduce_mean_output = reduce_mean_index_vector(table_sequence_output, table_cell_index)

        
        table_tag_input = table_sequence_output
        table_tag_prediction = self.tag_predictor(table_tag_input) #bsz, 512,2
        table_tag_prediction_prob = util.masked_softmax(table_tag_prediction, mask=None)
        table_tag_prediction = util.masked_log_softmax(table_tag_prediction, mask=None)
        table_tag_prediction = util.replace_masked_values(table_tag_prediction, table_mask.unsqueeze(-1), 0)
        table_tag_prediction_prob = util.replace_masked_values(table_tag_prediction_prob, table_mask.unsqueeze(-1), 0)
        
#         import pdb
        paragraph_sequence_output = util.replace_masked_values(sequence_output, paragraph_mask.unsqueeze(-1), 0)
#         paragraph_tag_input = torch.cat([question_node.unsqueeze(1).expand(-1, paragraph_sequence_output.size(1),-1), paragraph_sequence_output], dim=-1)
        paragraph_tag_input = paragraph_sequence_output
        paragraph_tag_prediction = self.tag_predictor(paragraph_tag_input)
        paragraph_tag_prediction = util.masked_log_softmax(paragraph_tag_prediction, mask=None)
        paragraph_tag_prediction = util.replace_masked_values(paragraph_tag_prediction, paragraph_mask.unsqueeze(-1), 0)

        paragraph_reduce_mean = torch.mean(paragraph_sequence_output, dim=1)
        table_reduce_mean = torch.mean(table_sequence_output, dim=1)
        cls_output = torch.cat((cls_output, table_reduce_mean, paragraph_reduce_mean), dim=-1)
        scale_prediction = self.scale_predictor(cls_output)

        device = input_ids.device
        ## select the max probability of subtokens of a cell
        table_tag_reduce_max_prediction, _ = \
            reduce_max_index_get_vector(table_tag_prediction[:, :, 1], table_sequence_output, table_cell_index)
        ## (bsz, 512), (bsz, 512, h), (bsz, 512)
        table_sequence_reduce_mean_output = reduce_mean_index_vector(table_sequence_output, table_cell_index) 
        # bsz, 512,1024
        
        ## select the max probability of subtokens of a cell
        paragraph_tag_reduce_max_prediction, _ = \
            reduce_max_index_get_vector(paragraph_tag_prediction[:, :, 1], paragraph_sequence_output, paragraph_index)
        paragraph_sequence_reduce_mean_output = reduce_mean_index_vector(paragraph_sequence_output, paragraph_index)
        
#         pdb.set_trace()
        table_reduce_mask = reduce_mean_index(table_mask, table_cell_index)
        paragraph_reduce_mask = reduce_mean_index(paragraph_mask, paragraph_index)
        
        masked_table_tag_reduce_max_prediction = util.replace_masked_values(table_tag_reduce_max_prediction,
                                                                            table_reduce_mask,
                                                                            -1e+5)
        masked_paragraph_tag_reduce_max_prediction = util.replace_masked_values(paragraph_tag_reduce_max_prediction,
                                                                                paragraph_reduce_mask,
                                                                                -1e+5)
        sorted_table_tag_prediction, sorted_cell_index = torch.sort(masked_table_tag_reduce_max_prediction,
                                                                    dim=-1, descending=True)
        sorted_paragraph_tag_prediction, sorted_paragraph_index = torch.sort(masked_paragraph_tag_reduce_max_prediction,
                                                                             dim=-1, descending=True)
        sorted_table_tag_prediction = sorted_table_tag_prediction[:, :2]
        sorted_cell_index = sorted_cell_index[:, :2]   ## select max probability of cell
        sorted_paragraph_tag_prediction = sorted_paragraph_tag_prediction[:, :2]
        sorted_paragraph_index = sorted_paragraph_index[:, :2]
        concat_tag_prediction = torch.cat((sorted_paragraph_tag_prediction, sorted_table_tag_prediction),
                                          dim=1)
        _, sorted_concat_tag_index = torch.sort(concat_tag_prediction, dim=-1, descending=True)
        top_2_sequence_output = torch.zeros(batch_size, 2, sequence_output.shape[2]).to(device)
        
        predicted_operator_class = torch.argmax(operator_prediction, dim=-1)
        top_2_number = np.zeros((batch_size, 2))
        number_index = 0
        
        
        ## top_2_number_with_timetamp
        top_2_number_timetamp = np.zeros((batch_size, 2))
        
        
#         ## for order
#         table_reduce_mean_for_order =  torch.zeros(batch_size, sequence_output.shape[2]).to(device)
#         paragraph_reduce_mean_for_order = torch.zeros(batch_size, sequence_output.shape[2]).to(device)
#         question_reduce_mean_for_order = torch.zeros(batch_size, sequence_output.shape[2]).to(device)
        
        for bsz in range(batch_size):
            if ("DIVIDE" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES[
                "DIVIDE"]) or \
                    ("DIFF" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES[
                        "DIFF"]) or \
                    ("CHANGE_RATIO" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES[
                        "CHANGE_RATIO"]):
                _index = sorted_concat_tag_index[bsz]
                if _index[0] > 1:
                    top_2_number[number_index, 0] = table_cell_numbers[bsz][sorted_cell_index[bsz, _index[0] - 2] - 1]
                    top_2_sequence_output[number_index, 0, :] = table_sequence_reduce_mean_output[bsz,
                                                                sorted_cell_index[bsz, _index[0] - 2], :]
                    ## timetamp
                    top_2_number_timetamp[number_index, 0] = table_cell_number_value_with_time[bsz][sorted_cell_index[bsz, _index[0] - 2] - 1]
                else:
                    top_2_number[number_index, 0] = paragraph_numbers[bsz][sorted_paragraph_index[bsz, _index[0]] - 1]
                    top_2_sequence_output[number_index, 0, :] = paragraph_sequence_reduce_mean_output[bsz,
                                                                sorted_paragraph_index[bsz, _index[0]], :]
                    
                    
                if _index[1] > 1:
                    top_2_number[number_index, 1] = table_cell_numbers[bsz][sorted_cell_index[bsz, _index[1] - 2] - 1]
                    top_2_sequence_output[number_index, 1, :] = table_sequence_reduce_mean_output[bsz,
                                                                sorted_cell_index[bsz, _index[1] - 2], :]
                    ## timetamp                
                    top_2_number_timetamp[number_index, 1] = table_cell_number_value_with_time[bsz][sorted_cell_index[bsz, _index[1] - 2] - 1]
                else:
                    top_2_number[number_index, 1] = paragraph_numbers[bsz][sorted_paragraph_index[bsz, _index[1]] - 1]
                    top_2_sequence_output[number_index, 1, :] = paragraph_sequence_reduce_mean_output[bsz,
                                                                sorted_paragraph_index[bsz, _index[1]], :]
#                 ## for order
#                 table_reduce_mean_for_order[number_index,:] = table_reduce_mean[bsz,:]
#                 paragraph_reduce_mean_for_order[number_index,:] = paragraph_reduce_mean_for_order[bsz,:]
#                 question_reduce_mean_for_order[number_index,:] = question_node[bsz,:]    
                    
                    
                number_index += 1
        top_2_order_prediction = self.order_predictor( torch.mean(top_2_sequence_output[:number_index], dim=1))
        ## for order error
#         order_input = torch.mean(top_2_sequence_output[:number_index], dim=1) + table_reduce_mean_for_order[:number_index]+paragraph_reduce_mean_for_order[:number_index] + question_reduce_mean_for_order[:number_index]
#         top_2_order_prediction = self.order_predictor(order_input)


        top_2_number = top_2_number[:number_index]
        top_2_number_timetamp = top_2_number_timetamp[:number_index]

        output_dict = {}

        predicted_scale_class = torch.argmax(scale_prediction, dim=-1).detach().cpu().numpy()
        predicted_operator_class = predicted_operator_class.detach().cpu().numpy()

#         import pdb
#         pdb.set_trace()
        paragraph_tag_prediction_score = paragraph_tag_prediction[:, :, 1]
        paragraph_tag_prediction = torch.argmax(paragraph_tag_prediction, dim=-1).float()
        paragraph_token_tag_prediction = reduce_mean_index(paragraph_tag_prediction, paragraph_index)
        paragraph_token_tag_prediction_score = reduce_max_index(paragraph_tag_prediction_score, paragraph_index)
        paragraph_token_tag_prediction = paragraph_token_tag_prediction.detach().cpu().numpy()
        paragraph_token_tag_prediction_score = paragraph_token_tag_prediction_score.detach().cpu().numpy()

#         table_tag_prediction_score = table_tag_prediction[:, :, 1]
#         table_tag_prediction = torch.argmax(table_tag_prediction, dim=-1).float()
#         table_cell_tag_prediction = reduce_mean_index(table_tag_prediction, table_cell_index)
#         table_cell_tag_prediction_score = reduce_max_index(table_tag_prediction_score, table_cell_index)
#         table_cell_tag_prediction = table_cell_tag_prediction.detach().cpu().numpy()
#         table_cell_tag_prediction_score = table_cell_tag_prediction_score.detach().cpu().numpy()
        
        ## row/ cell
        if self.add_row_col_tag:

            table_cell_tag_prediction_prob_with_row = torch.matmul(row_tag_prediction_prob, row_include_cells.float()) ## bsz, 2, 512
            table_cell_tag_prediction_prob_with_row = table_cell_tag_prediction_prob_with_row.transpose(1,2)
            table_cell_tag_prediction_prob_with_row_new = torch.zeros(batch_size, 512, 2).cuda()
            table_cell_tag_prediction_prob_with_row_new[:,1:,:] = table_cell_tag_prediction_prob_with_row[:,:-1,:]
            table_tag_prediction_prob_with_row = torch.gather(table_cell_tag_prediction_prob_with_row_new, index=table_cell_index.unsqueeze(-1).expand(-1,-1,table_cell_tag_prediction_prob_with_row.size(-1)), dim=1)


            table_cell_tag_prediction_prob_with_col = torch.matmul(col_tag_prediction_prob, col_include_cells.float()) ## bsz, 2, 512
            table_cell_tag_prediction_prob_with_col = table_cell_tag_prediction_prob_with_col.transpose(1,2)
            table_cell_tag_prediction_prob_with_col_new = torch.zeros(batch_size, 512, 2).cuda()
            table_cell_tag_prediction_prob_with_col_new[:,1:,:] = table_cell_tag_prediction_prob_with_col[:,:-1,:]
            table_tag_prediction_prob_with_col = torch.gather(table_cell_tag_prediction_prob_with_col_new, index=table_cell_index.unsqueeze(-1).expand(-1,-1,table_cell_tag_prediction_prob_with_col.size(-1)), dim=1)

            table_tag_prediction_prob += table_tag_prediction_prob_with_row + table_tag_prediction_prob_with_col
            table_tag_prediction_prob = table_tag_prediction_prob/3

            table_tag_prediction_score = table_tag_prediction_prob[:, :, 1]   ## bsz, 512
            table_tag_prediction_prob = torch.argmax(table_tag_prediction_prob, dim=-1).float() ## bsz, 512
#             cell-level tag prediction and score
            table_cell_tag_prediction = reduce_mean_index(table_tag_prediction_prob, table_cell_index)
            table_cell_tag_prediction_score = reduce_max_index(table_tag_prediction_score, table_cell_index) ## bsz, 512
            table_cell_tag_prediction = table_cell_tag_prediction.detach().cpu().numpy()
            table_cell_tag_prediction_score = table_cell_tag_prediction_score.detach().cpu().numpy()

#                 row_tag_prediction_score = row_tag_prediction[:, :, 1]
#                 row_tag_prediction = torch.argmax(row_tag_prediction, dim=-1).float()
#                 table_tag_prediction_score_with_row = torch.matmul(row_tag_prediction_score.unsqueeze(1), row_include_cells.float()).squeeze(1)
#                 table_tag_prediction_score_with_row_new = torch.zeros(batch_size, 513)
#                 table_tag_prediction_score_with_row_new[:,1:] = table_tag_prediction_score_with_row
#                 table_tag_prediction_score_with_row_new = table_tag_prediction_score_with_row_new[:,:-1].detach().cpu().numpy()

#                 col_tag_prediction = col_tag_prediction.transpose(1, 2)
#                 col_tag_prediction_score = col_tag_prediction[:, :, 1]
#                 col_tag_prediction = torch.argmax(col_tag_prediction, dim=-1).float()
#                 table_tag_prediction_score_with_col= torch.matmul(col_tag_prediction_score.unsqueeze(1).float(), col_include_cells.float()).squeeze(1)
#                 table_tag_prediction_score_with_col_new = torch.zeros(batch_size, 513)
#                 table_tag_prediction_score_with_col_new[:,1:] = table_tag_prediction_score_with_col
#                 table_tag_prediction_score_with_col_new = table_tag_prediction_score_with_col_new[:,:-1].detach().cpu().numpy()

#                 table_cell_tag_prediction_score += table_tag_prediction_score_with_row_new + table_tag_prediction_score_with_col_new
        else:
#                 table_tag_prediction = table_tag_prediction.transpose(1, 2)
            table_tag_prediction_score = table_tag_prediction[:, :, 1]   ## bsz, 512
            table_tag_prediction = torch.argmax(table_tag_prediction, dim=-1).float() ## bsz, 512
#             cell-level tag prediction and score
            table_cell_tag_prediction = reduce_mean_index(table_tag_prediction, table_cell_index)
            table_cell_tag_prediction_score = reduce_max_index(table_tag_prediction_score, table_cell_index) ## bsz, 512
            table_cell_tag_prediction = table_cell_tag_prediction.detach().cpu().numpy()
            table_cell_tag_prediction_score = table_cell_tag_prediction_score.detach().cpu().numpy()
                
        output_dict["question_id"] = []
        output_dict["answer"] = []
        output_dict["scale"] = []
        output_dict["gold_answers"] = []
        output_dict["pred_span"] = []
        output_dict["gold_span"] = []
        
        ## add for error analysis
        output_dict["aux_info"] = []

        top_2_index = 0
        
        selected_evidence = None
        
        if number_index != 0:
            top_2_order_prediction = top_2_order_prediction.detach().cpu().numpy()
            top_2_order_prediction = np.argmax(top_2_order_prediction, axis=1)
            
        for bsz in range(batch_size):
            pred_span = []
            current_op = "ignore"
            if "SPAN-TEXT" in self.OPERATOR_CLASSES and \
                    predicted_operator_class[bsz] == self.OPERATOR_CLASSES["SPAN-TEXT"]:
                paragraph_selected_span_tokens = get_single_span_tokens_from_paragraph(
                    paragraph_token_tag_prediction[bsz],
                    paragraph_token_tag_prediction_score[bsz],
                    paragraph_tokens[bsz]
                )
                answer = paragraph_selected_span_tokens
                answer = sorted(answer)
                output_dict["pred_span"].append(answer)
                pred_span += answer
                current_op = "Span-in-text"
                
                selected_evidence = paragraph_selected_span_tokens
                
                
            elif "SPAN-TABLE" in self.OPERATOR_CLASSES and \
                 predicted_operator_class[bsz] == self.OPERATOR_CLASSES["SPAN-TABLE"]:
                table_selected_tokens = get_single_span_tokens_from_table(
                    table_cell_tag_prediction[bsz],
                    table_cell_tag_prediction_score[bsz],
                    table_cell_tokens[bsz])
                answer = table_selected_tokens
                answer = sorted(answer)
                output_dict["pred_span"].append(answer)
                pred_span += answer
                current_op = "Cell-in-table"
                
                selected_evidence = table_selected_tokens
                
                
            elif "MULTI_SPAN" in self.OPERATOR_CLASSES and \
                    predicted_operator_class[bsz] == self.OPERATOR_CLASSES["MULTI_SPAN"]:
                paragraph_selected_span_tokens = \
                    get_span_tokens_from_paragraph(paragraph_token_tag_prediction[bsz], paragraph_tokens[bsz])
                table_selected_tokens = \
                    get_span_tokens_from_table(table_cell_tag_prediction[bsz], table_cell_tokens[bsz])
                answer = paragraph_selected_span_tokens + table_selected_tokens
                answer = sorted(answer)
                output_dict["pred_span"].append(answer)
                pred_span += answer
                current_op = "Spans"
                
                selected_evidence = table_selected_tokens
                
            elif "COUNT" in self.OPERATOR_CLASSES and \
                    predicted_operator_class[bsz] == self.OPERATOR_CLASSES["COUNT"]:
                paragraph_selected_tokens = \
                    get_span_tokens_from_paragraph(paragraph_token_tag_prediction[bsz], paragraph_tokens[bsz])
                table_selected_tokens = \
                    get_span_tokens_from_table(table_cell_tag_prediction[bsz], table_cell_tokens[bsz])
                answer = len(paragraph_selected_tokens) + len(table_selected_tokens)
                output_dict["pred_span"].append(answer)
                pred_span += sorted(paragraph_selected_tokens + table_selected_tokens)
                current_op = "Count"
                
                selected_evidence = pred_span
                
            else:
                if ("SUM" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["SUM"]) \
                        or ("TIMES" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["TIMES"]) \
                        or ("AVERAGE" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES["AVERAGE"]):
                    paragraph_selected_numbers = \
                        get_numbers_from_reduce_sequence(paragraph_token_tag_prediction[bsz],
                                                         paragraph_numbers[bsz])
                    table_selected_numbers = \
                        get_numbers_from_reduce_sequence(table_cell_tag_prediction[bsz], table_cell_numbers[bsz])
                    selected_numbers = paragraph_selected_numbers + table_selected_numbers
                    output_dict["pred_span"].append(selected_numbers)
                    pred_span += sorted(selected_numbers)
                    if not selected_numbers:
                        answer = ""
                    elif "SUM" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES[
                        "SUM"]:
                        answer = np.around(np.sum(selected_numbers), 4)
                        current_op = "Sum"
                    elif "TIMES" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == self.OPERATOR_CLASSES[
                        "TIMES"]:
                        answer = np.around(np.prod(selected_numbers), 4)
                        current_op = "Multiplication"
                    elif "AVERAGE" in self.OPERATOR_CLASSES:
                        answer = np.around(np.mean(selected_numbers), 4)
                        current_op = "Average"
                        
                    selected_evidence = selected_numbers
                else:
                    
                    if top_2_number.size <= 0:
                        answer = ""
                    if top_2_number.size > 0:
                        operand_one = top_2_number[top_2_index, 0]
                        operand_two = top_2_number[top_2_index, 1]
                        predicted_order = top_2_order_prediction[top_2_index]
                        pred_span += [operand_one, operand_two]
                        
                        ## add tamp
                        operand_one_timetamp = top_2_number_timetamp[top_2_index, 0]
                        operand_two_timetamp = top_2_number_timetamp[top_2_index, 1]
                        
#                         import pdb
#                         pdb.set_trace()
                        if np.isnan(operand_one) or np.isnan(operand_two):
                            answer = ""
                        else:
                            if "DIFF" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == \
                                    self.OPERATOR_CLASSES["DIFF"]:
                                current_op = "Difference"
                                
                                ## add our rule
                                if np.isnan(operand_one_timetamp) or operand_one_timetamp==0 or np.isnan(operand_two_timetamp) or operand_two_timetamp==0:
                                    if predicted_order == 0:
                                        answer = np.around(operand_one - operand_two, 4)
                                    else:
                                        answer = np.around(operand_two - operand_one, 4)
                                else:
                                    
                                    if operand_one_timetamp>operand_two_timetamp:
                                        answer = np.around(operand_one - operand_two, 4)
                                    elif operand_one_timetamp<operand_two_timetamp:
                                        answer = np.around(operand_two - operand_one, 4)
                                    else:
                                        if predicted_order == 0:
                                            answer = np.around(operand_one - operand_two, 4)
                                        else:
                                            answer = np.around(operand_two - operand_one, 4)
                                        
                                    
                                    
                            elif "DIVIDE" in self.OPERATOR_CLASSES and predicted_operator_class[bsz] == \
                                    self.OPERATOR_CLASSES["DIVIDE"]:
                                current_op = "Division"
                                
                                if np.isnan(operand_one_timetamp) or operand_one_timetamp==0 or np.isnan(operand_two_timetamp) or operand_two_timetamp==0:
                                    if predicted_order == 0:
                                        answer = np.around(operand_one / operand_two, 4)
                                    else:
                                        answer = np.around(operand_two / operand_one, 4)
                                else:
                                    if operand_one_timetamp>operand_two_timetamp:
                                        answer = np.around(operand_one / operand_two, 4)
                                    elif operand_one_timetamp<operand_two_timetamp:
                                        answer = np.around(operand_two / operand_one, 4)
                                    else:
                                        if predicted_order == 0:
                                            answer = np.around(operand_one / operand_two, 4)
                                        else:
                                            answer = np.around(operand_two / operand_one, 4)
                                        
                                    
                                if SCALE[int(predicted_scale_class[bsz])] == "percent":
                                    answer = answer * 100
                            elif "CHANGE_RATIO" in self.OPERATOR_CLASSES:
                                current_op = "Change ratio"
                                
                                if np.isnan(operand_one_timetamp) or operand_one_timetamp==0 or np.isnan(operand_two_timetamp) or operand_two_timetamp==0:
                                    if predicted_order == 0:
                                        answer = np.around(operand_one / operand_two - 1, 4)
                                    else:
                                        answer = np.around(operand_two / operand_one - 1, 4)
                                else:
                                    if operand_one_timetamp>operand_two_timetamp:
                                        answer = np.around(operand_one / operand_two - 1, 4)
                                    elif operand_one_timetamp<operand_two_timetamp:
                                         answer = np.around(operand_two / operand_one - 1, 4)
                                    else:
                                        if predicted_order == 0:
                                            answer = np.around(operand_one / operand_two - 1, 4)
                                        else:
                                            answer = np.around(operand_two / operand_one - 1, 4)
                                    
                                if SCALE[int(predicted_scale_class[bsz])] == "percent":
                                    answer = answer * 100
                        top_2_index += 1
                        
                        selected_evidence = pred_span

            output_dict["answer"].append(answer)
            output_dict["scale"].append(SCALE[int(predicted_scale_class[bsz])])
            output_dict["question_id"].append(question_ids[bsz])
            output_dict["gold_answers"].append(gold_answers[bsz])
            
            ## for error analysis
            import pdb
            
            aux_info = {}
            
            ## paragraph
            import pdb
#             pdb.set_trace()
            
            current_paragraph_tokens = paragraph_tokens[bsz]
            current_paragraph_token_tag_prediction = paragraph_token_tag_prediction[bsz]
            
            current_paragraph_tag_result = [(current_paragraph_tokens[i],current_paragraph_token_tag_prediction[i+1])  for i in range(min(512-4- sum(question_mask[bsz])-sum(table_mask[bsz]), len(current_paragraph_tokens)))]
    
            ## table cell
            current_table_tokens = table_cell_tokens[bsz]
            current_table_cell_tokens = table_cell_tokens[bsz]
            current_table_cell_tag_prediction = table_cell_tag_prediction[bsz]
#             assert len(current_table_cell_tokens) <=512
            current_table_cell_tag_result = [(current_table_cell_tokens[i],current_table_cell_tag_prediction[i+1])  for i in range(min(len(current_table_cell_tokens), 512-3-sum(question_mask[bsz])))]

            ## 
            ID_OP_MAP = dict(zip(self.OPERATOR_CLASSES.values(), self.OPERATOR_CLASSES.keys()))
            op_predict = ID_OP_MAP[predicted_operator_class[bsz]]
            

            aux_info.update({
                "question_id":question_ids[bsz],
                "current_paragraph_tag_result":str(current_paragraph_tag_result),
                "current_table_cell_tag_result":str(current_table_cell_tag_result),
                "op_predict":op_predict,
                "selected_evidence": str(selected_evidence),
                "scale_predict":SCALE[int(predicted_scale_class[bsz])],
                "gold_answers":gold_answers[bsz],
            })
            
            import pdb
#             pdb.set_trace()
            
            output_dict["aux_info"].append(aux_info)
            # output_dict["gold_span"].append(sorted(table_mapping_content[bsz] + paragraph_mapping_content[bsz]))
            if predicted_operator_class[bsz] in self.arithmetic_op_index:
                predict_type = "arithmetic"
            else:
                predict_type = ""
            self._metrics({**gold_answers[bsz], "uid": question_ids[bsz]}, answer,
                          SCALE[int(predicted_scale_class[bsz])], None, None,
                          pred_op=current_op, gold_op=gold_answers[bsz]["gold_op"])
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
        return {'em': exact_match, 'f1': f1_score, "scale": scale_score}

    def get_df(self):
        raws = self._metrics.get_raw()
        detail_em, detail_f1 = self._metrics.get_detail_metric()
        raw_detail = self._metrics.get_raw_pivot_table()
        return detail_em, detail_f1, raws, raw_detail


### Beginning of everything related to segmented tensors ###


class IndexMap(object):
    """Index grouping entries within a tensor."""

    def __init__(self, indices, num_segments, batch_dims=0):
        """
        Creates an index
        Args:
            indices (:obj:`torch.LongTensor`, same shape as a `values` Tensor to which the indices refer):
                Tensor containing the indices.
            num_segments (:obj:`torch.LongTensor`):
                Scalar tensor, the number of segments. All elements in a batched segmented tensor must have the same
                number of segments (although many segments can be empty).
            batch_dims (:obj:`int`, `optional`, defaults to 0):
                The number of batch dimensions. The first `batch_dims` dimensions of a SegmentedTensor are treated as
                batch dimensions. Segments in different batch elements are always distinct even if they have the same
                index.
        """
        self.indices = torch.as_tensor(indices)
        self.num_segments = torch.as_tensor(num_segments, device=indices.device)
        self.batch_dims = batch_dims

    def batch_shape(self):
        return self.indices.size()[: self.batch_dims]  # returns a torch.Size object


class ProductIndexMap(IndexMap):
    """The product of two indices."""

    def __init__(self, outer_index, inner_index):
        """
        Combines indices i and j into pairs (i, j). The result is an index where each segment (i, j) is the
        intersection of segments i and j. For example if the inputs represent table cells indexed by respectively rows
        and columns the output will be a table indexed by (row, column) pairs, i.e. by cell. The implementation
        combines indices {0, .., n - 1} and {0, .., m - 1} into {0, .., nm - 1}. The output has `num_segments` equal to
        `outer_index.num_segments` * `inner_index.num_segments`
        Args:
            outer_index (:obj:`IndexMap`):
                IndexMap.
            inner_index (:obj:`IndexMap`):
                IndexMap, must have the same shape as `outer_index`.
        """
        if outer_index.batch_dims != inner_index.batch_dims:
            raise ValueError("outer_index.batch_dims and inner_index.batch_dims must be the same.")

        super(ProductIndexMap, self).__init__(
            indices=(inner_index.indices + outer_index.indices * inner_index.num_segments),
            num_segments=inner_index.num_segments * outer_index.num_segments,
            batch_dims=inner_index.batch_dims,
        )
        self.outer_index = outer_index
        self.inner_index = inner_index

    def project_outer(self, index):
        """Projects an index with the same index set onto the outer components."""
        return IndexMap(
            indices=(index.indices // self.inner_index.num_segments).type(torch.float).floor().type(torch.long),
            num_segments=self.outer_index.num_segments,
            batch_dims=index.batch_dims,
        )

    def project_inner(self, index):
        """Projects an index with the same index set onto the inner components."""
        return IndexMap(
            indices=torch.fmod(index.indices, self.inner_index.num_segments)
                .type(torch.float)
                .floor()
                .type(torch.long),
            num_segments=self.inner_index.num_segments,
            batch_dims=index.batch_dims,
        )


def reduce_mean_vector(values, index, name="segmented_reduce_vector_mean"):
    return _segment_reduce_vector(values, index, "mean", name)


def reduce_mean(values, index, name="segmented_reduce_mean"):
    """
    Averages a tensor over its segments.
    Outputs 0 for empty segments.
    This operations computes the mean over segments, with support for:
        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present, the output will be a mean of
          vectors rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.
    Args:
        values (:obj:`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the mean must be taken segment-wise.
        index (:obj:`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (:obj:`str`, `optional`, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used
    Returns:
        output_values (:obj:`torch.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]): Tensor containing the
        output values. output_index (:obj:`IndexMap`): IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, "mean", name)


def reduce_mean_index_vector(values, index, max_length=512, name="index_reduce_mean"):
    return _index_reduce_vector(values, index, max_length, "mean", name)


def reduce_mean_index(values, index, max_length=512, name="index_reduce_mean"):
    '''
    属于同一个单词的token 表示取平均值
    '''
    return _index_reduce(values, index, max_length, "mean", name)


def reduce_max_index(values, index, max_length=512, name="index_reduce_max"):
    '''
    属于同一个单词的token 表示取最大值
    '''
    return _index_reduce_max(values, index, max_length, name)


def reduce_max_index_get_vector(values_for_reduce, values_for_reference, index,
                                max_length=512, name="index_reduce_get_vector"):
    return _index_reduce_max_get_vector(values_for_reduce, values_for_reference, index, max_length, name)


def flatten(index, name="segmented_flatten"):
    """
    Flattens a batched index map (which is typically of shape batch_size, seq_length) to a 1d index map. This operation
    relabels the segments to keep batch elements distinct. The k-th batch element will have indices shifted by
    `num_segments` * (k - 1). The result is a tensor with `num_segments` multiplied by the number of elements in the
    batch.
    Args:
        index (:obj:`IndexMap`):
            IndexMap to flatten.
        name (:obj:`str`, `optional`, defaults to 'segmented_flatten'):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): The flattened IndexMap.
    """
    # first, get batch_size as scalar tensor
    batch_size = torch.prod(torch.tensor(list(index.batch_shape())))
    # next, create offset as 1-D tensor of length batch_size,
    # and multiply element-wise by num segments (to offset different elements in the batch) e.g. if batch size is 2: [0, 64]
    offset = torch.arange(start=0, end=batch_size, device=index.num_segments.device) * index.num_segments
    offset = offset.view(index.batch_shape())
    for _ in range(index.batch_dims, len(index.indices.size())):  # typically range(1,2)
        offset = offset.unsqueeze(-1)

    indices = offset + index.indices
    return IndexMap(indices=indices.view(-1), num_segments=index.num_segments * batch_size, batch_dims=0)


def flatten_index(index, max_length=512, name="index_flatten"):
    batch_size = index.shape[0]
    offset = torch.arange(start=0, end=batch_size, device=index.device) * max_length
    offset = offset.view(batch_size, 1)
    return (index + offset).view(-1), batch_size * max_length


def range_index_map(batch_shape, num_segments, name="range_index_map"):
    """
    Constructs an index map equal to range(num_segments).
    Args:
        batch_shape (:obj:`torch.Size`):
            Batch shape
        num_segments (:obj:`int`):
            Number of segments
        name (:obj:`str`, `optional`, defaults to 'range_index_map'):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    batch_shape = torch.as_tensor(
        batch_shape, dtype=torch.long
    )  # create a rank 1 tensor vector containing batch_shape (e.g. [2])
    assert len(batch_shape.size()) == 1
    num_segments = torch.as_tensor(num_segments)  # create a rank 0 tensor (scalar) containing num_segments (e.g. 64)
    assert len(num_segments.size()) == 0

    indices = torch.arange(
        start=0, end=num_segments, device=num_segments.device
    )  # create a rank 1 vector with num_segments elements
    new_tensor = torch.cat(
        [torch.ones_like(batch_shape, dtype=torch.long, device=num_segments.device), num_segments.unsqueeze(dim=0)],
        dim=0,
    )
    # new_tensor is just a vector of [1 64] for example (assuming only 1 batch dimension)
    new_shape = [int(x) for x in new_tensor.tolist()]
    indices = indices.view(new_shape)

    multiples = torch.cat([batch_shape, torch.as_tensor([1])], dim=0)
    indices = indices.repeat(multiples.tolist())
    # equivalent (in Numpy:)
    # indices = torch.as_tensor(np.tile(indices.numpy(), multiples.tolist()))

    return IndexMap(indices=indices, num_segments=num_segments, batch_dims=list(batch_shape.size())[0])


def _segment_reduce(values, index, segment_reduce_fn, name):
    """
    Applies a segment reduction segment-wise.
    Args:
        values (:obj:`torch.Tensor`):
            Tensor with segment values.
        index (:obj:`IndexMap`):
            IndexMap.
        segment_reduce_fn (:obj:`str`):
            Name for the reduce operation. One of "sum", "mean", "max" or "min".
        name (:obj:`str`):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    # Flatten the batch dimensions, as segments ops (scatter) do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    flat_index = flatten(index)
    vector_shape = values.size()[len(index.indices.size()):]  # torch.Size object
    flattened_shape = torch.cat(
        [torch.as_tensor([-1], dtype=torch.long), torch.as_tensor(vector_shape, dtype=torch.long)], dim=0
    )
    # changed "view" by "reshape" in the following line
    flat_values = values.reshape(flattened_shape.tolist())

    segment_means = scatter(
        src=flat_values,
        index=flat_index.indices.type(torch.long),
        dim=0,
        dim_size=flat_index.num_segments,
        reduce=segment_reduce_fn,
    )

    # Unflatten the values.
    new_shape = torch.cat(
        [
            torch.as_tensor(index.batch_shape(), dtype=torch.long),
            torch.as_tensor([index.num_segments], dtype=torch.long),
            torch.as_tensor(vector_shape, dtype=torch.long),
        ],
        dim=0,
    )

    output_values = segment_means.view(new_shape.tolist())
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return output_values, output_index


def _segment_reduce_vector(values, index, segment_reduce_fn, name):
    """
    Applies a segment reduction segment-wise.
    Args:
        values (:obj:`torch.Tensor`):
            Tensor with segment values.
        index (:obj:`IndexMap`):
            IndexMap.
        segment_reduce_fn (:obj:`str`):
            Name for the reduce operation. One of "sum", "mean", "max" or "min".
        name (:obj:`str`):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    # Flatten the batch dimensions, as segments ops (scatter) do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    flat_index = flatten(index)
    vector_shape = values.size()[len(index.indices.size()):]  # torch.Size object
    bsz = values.shape[0]
    seq_len = values.shape[1]
    hidden_size = values.shape[2]
    flat_values = values.reshape(bsz * seq_len, hidden_size)
    segment_means = scatter(
        src=flat_values,
        index=flat_index.indices.type(torch.long),
        dim=0,
        dim_size=flat_index.num_segments,
        reduce=segment_reduce_fn,
    )
    output_values = segment_means.view(bsz, -1, hidden_size)
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return output_values, output_index


def _index_reduce(values, index, max_length, index_reduce_fn, name):
    flat_index, num_index = flatten_index(index, max_length)
    bsz = values.shape[0]
    seq_len = values.shape[1]
    flat_values = values.reshape(bsz * seq_len)
    index_means = scatter(
        src=flat_values,
        index=flat_index.type(torch.long),
        dim=0,
        dim_size=num_index,
        reduce=index_reduce_fn,
    )
    output_values = index_means.view(bsz, -1)
    return output_values


def _index_reduce_max(values, index, max_length, name):
    flat_index, num_index = flatten_index(index, max_length)
    bsz = values.shape[0]
    seq_len = values.shape[1]
    flat_values = values.reshape(bsz * seq_len)
    index_max, _ = scatter_max(
        src=flat_values,
        index=flat_index.type(torch.long),
        dim=0,
        dim_size=num_index,
    )
    output_values = index_max.view(bsz, -1)
    return output_values


def _index_reduce_max_get_vector(values_for_reduce, values_for_reference, index, max_length, name):
    import pdb
#     pdb.set_trace()
    flat_index, num_index = flatten_index(index, max_length)
    bsz = values_for_reduce.shape[0]
    seq_len = values_for_reference.shape[1]
    flat_values_for_reduce = values_for_reduce.reshape(bsz * seq_len)
    flat_values_for_reference = values_for_reference.reshape(bsz * seq_len, -1)
    
    reduce_values, reduce_index = scatter_max(
        src=flat_values_for_reduce,
        index=flat_index.type(torch.long),
        dim=0,
        dim_size=num_index,
    )
    ## reduce_values: select the max log_p for every sub_token in cell i.
    ## reduce_index: the sub_token index selected 
    
#     pdb.set_trace()
    reduce_index[reduce_index == -1] = flat_values_for_reference.shape[0]
    reduce_values = reduce_values.view(bsz, -1)
    flat_values_for_reference = torch.cat(
        (flat_values_for_reference, torch.zeros(1, flat_values_for_reference.shape[1]).to(values_for_reduce.device)),
        dim=0)
    flat_values_for_reference = torch.index_select(flat_values_for_reference, dim=0, index=reduce_index)
    flat_values_for_reference = flat_values_for_reference.view(bsz, reduce_values.shape[1], -1)
    return reduce_values, flat_values_for_reference


def _index_reduce_vector(values, index, max_length, index_reduce_fn, name):
    import pdb
#     pdb.set_trace()
    flat_index, num_index = flatten_index(index, max_length)
    bsz = values.shape[0]
    seq_len = values.shape[1]
    hidden_size = values.shape[2]
    flat_values = values.reshape(bsz * seq_len, hidden_size)
    index_means = scatter(
        src=flat_values,
        index=flat_index.type(torch.long),
        dim=0,
        dim_size=num_index,
        reduce=index_reduce_fn,
    )
    
    output_values = index_means.view(bsz, -1, hidden_size)
    return output_values
