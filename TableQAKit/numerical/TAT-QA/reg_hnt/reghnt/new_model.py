import copy, math
from hashlib import new
import enum
from turtle import forward
import torch, dgl
import dgl.function as fn
import torch
from torch import dtype
from torch import embedding
import torch.nn as nn
from .tools import allennlp as util
from tatqa_metric import TaTQAEmAndF1
from .tools.util import FFNLayer
from .tools import allennlp as util
from typing import Dict, List, Tuple
import numpy as np
from reg_hnt.data.file_utils import is_scatter_available
import math
import torch.nn.functional as F
from reg_hnt.reghnt.model_utils import rnn_wrapper, lens2mask, PoolingFunction, FFN
from reg_hnt.reghnt.functions import *
from reg_hnt.reghnt.modeling_tree import TreeNode, Score, TreeAttn, Prediction, TreeEmbedding, TreeBeam, generate_tree_input, GenerateNode, Merge, copy_list
from reg_hnt.reghnt.Vocabulary import *
from reg_hnt.reghnt.masked_cross_entropy import *
from reg_hnt.reghnt.model_functions import *
from reg_hnt.data.pre_order import prefix_eval
from reg_hnt.data.pre_order_ext import pretomid


class RGATLayer(nn.Module):
    def __init__(self, ndim, edim, num_heads=8, feat_drop=0.2):
        super(RGATLayer, self).__init__()
        self.ndim, self.edim = ndim, edim
        self.num_heads = num_heads
        dim = max([ndim, edim])
        self.d_k = dim // self.num_heads
        self.affine_q, self.affine_k, self.affine_v = nn.Linear(self.ndim, dim),\
            nn.Linear(self.ndim, dim, bias=False), nn.Linear(self.ndim, dim, bias=False)
        self.affine_o = nn.Linear(dim, self.ndim)
        self.layernorm = nn.LayerNorm(self.ndim)
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.ffn = FFN(self.ndim)

    def forward(self, x, lgx, g):
        """ @Params:
                x: node feats, num_nodes x ndim
                lgx: edge feats, num_edges x edim
                g: dgl.graph
        """
        # pre-mapping q/k/v affine
        q, k, v = self.affine_q(self.feat_dropout(x)), self.affine_k(self.feat_dropout(x)), self.affine_v(self.feat_dropout(x))
        e = lgx.view(-1, self.num_heads, self.d_k) if lgx.size(-1) == q.size(-1) else \
            lgx.unsqueeze(1).expand(-1, self.num_heads, -1)
        
        with g.local_scope():
            g.ndata['q'], g.ndata['k'] = q.view(-1, self.num_heads, self.d_k), k.view(-1, self.num_heads, self.d_k)
            g.ndata['v'] = v.view(-1, self.num_heads, self.d_k)
            g.edata['e'] = e
            out_x = self.propagate_attention(g)

        out_x = self.layernorm(x + self.affine_o(out_x.view(-1, self.num_heads * self.d_k)))
        out_x = self.ffn(out_x)
        return out_x, lgx

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_sum_edge_mul_dst('k', 'q', 'e', 'score'))
        g.apply_edges(scaled_exp('score', math.sqrt(self.d_k)))
        # Update node state
        g.update_all(src_sum_edge_mul_edge('v', 'e', 'score', 'v'), fn.sum('v', 'wv'))
        g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'), div_by_z('wv', 'z', 'o'))
        
        out_x = g.ndata['o']
        return out_x



class RGTAT(nn.Module):
    def __init__(self, gnn_hidden_size, gnn_num_layers, num_heads, relation_num):
        super(RGTAT, self).__init__()
        self.num_layers, self.num_heads = gnn_num_layers, num_heads
        self.ndim = gnn_hidden_size
        self.edim = self.ndim // self.num_heads
        self.relation_num = relation_num
        self.relation_embed = nn.Embedding(self.relation_num, self.edim)
        self.dropout = 0.2
        self.gnn_layers = nn.ModuleList([
            RGATLayer(self.ndim, self.edim, num_heads=num_heads, feat_drop=self.dropout)
            for _ in range(self.num_layers)])

    def forward(self, x, batch):
        local_lgx = self.relation_embed(batch['b_relation'])
        local_g = batch['b_graph']
        for i in range(self.num_layers):
            x, local_lgx = self.gnn_layers[i](x, local_lgx, local_g) 
        return x


class SubwordAggregation(nn.Module):
    """ Map subword or wordpieces into one fixed size vector based on aggregation method
    """
    def __init__(self, hidden_size, subword_aggregation='mean-pooling'):
        super(SubwordAggregation, self).__init__()
        self.hidden_size = hidden_size
        self.word_aggregation = PoolingFunction(self.hidden_size, self.hidden_size, method=subword_aggregation)
        self.number_aggregation = PoolingFunction(self.hidden_size, self.hidden_size, method=subword_aggregation)

    def forward(self, inputs, batch):
        bs = inputs.size(0)
        device = inputs.device

        word_mask = batch['word_mask'].bool()
        number_mask = batch['number_mask'].bool()
        word_subword_lens = batch['word_subword_lens']
        number_subword_lens = batch['number_subword_lens']
        word_subword_mask = batch['word_subword_mask']
        number_subword_mask = batch['number_subword_mask']

        old_word, old_number = inputs.masked_select(word_mask.unsqueeze(-1)), inputs.masked_select(number_mask.unsqueeze(-1))
        words = old_word.new_zeros(word_subword_lens.size(0), max(batch['word_subword_lens']).item(), self.hidden_size)
        try:
            words = words.masked_scatter_(word_subword_mask.unsqueeze(-1), old_word)
        except:
            import pdb; pdb.set_trace()
        numbers = old_number.new_zeros(number_subword_lens.size(0), max(batch['number_subword_lens']).item(), self.hidden_size)
        try:
            numbers = numbers.masked_scatter_(number_subword_mask.unsqueeze(-1), old_number)
        except:
            import pdb; pdb.set_trace()

        words = self.word_aggregation(words, mask=word_subword_mask)
        numbers = self.word_aggregation(numbers, mask=number_subword_mask)
        
        b_word_word_mask = batch['b_word_word_mask']
        b_number_word_mask = batch['b_number_word_mask']
        new_inputs = torch.zeros([b_word_word_mask.size(0), self.hidden_size]).to(device)
        new_inputs = new_inputs.masked_scatter_(b_word_word_mask.unsqueeze(-1), words)
        new_inputs = new_inputs.masked_scatter_(b_number_word_mask.unsqueeze(-1), numbers)
        
        
        inputs = torch.zeros([bs, batch['pl_mask']['b'].size(1), self.hidden_size]).to(device)
        inputs = inputs.masked_scatter_(batch['pl_mask']['b'].unsqueeze(-1), new_inputs)
        
        old_questions, old_tables, old_paragraphs = inputs.masked_select(batch['pl_mask']['q'].unsqueeze(-1)), \
            inputs.masked_select(batch['pl_mask']['t'].unsqueeze(-1)), inputs.masked_select(batch['pl_mask']['p'].unsqueeze(-1))
        questions = old_questions.new_zeros(bs, batch['max_len']['question'], self.hidden_size)
        questions = questions.masked_scatter_(batch['pl_mask']['question'].unsqueeze(-1), old_questions)
        tables = old_tables.new_zeros(bs, batch['max_len']['table_word'], self.hidden_size)
        tables = tables.masked_scatter_(batch['pl_mask']['table'].unsqueeze(-1), old_tables)
        paragraphs = old_paragraphs.new_zeros(bs, batch['max_len']['paragraph'], self.hidden_size)
        paragraphs = paragraphs.masked_scatter_(batch['pl_mask']['paragraph'].unsqueeze(-1), old_paragraphs)
        
        return questions, tables, paragraphs


class InputRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, cell='lstm', schema_aggregation='attentive-pooling', share_lstm=False):
        super(InputRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = cell.upper()
        self.question_lstm = getattr(nn, self.cell)(self.input_size, self.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.table_lstm = getattr(nn, self.cell)(self.input_size, self.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.paragraph_lstm = getattr(nn, self.cell)(self.input_size, self.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.schema_aggregation = schema_aggregation
        if self.schema_aggregation != 'head+tail':
            self.aggregation = PoolingFunction(self.hidden_size, self.hidden_size, method=schema_aggregation)

    def forward(self, input_dict, batch):
        """
            for question sentence, forward into a bidirectional LSTM to get contextual info and sequential dependence
            for schema phrase, extract representation for each phrase by concatenating head+tail vectors,
            batch.question_lens, batch.table_word_lens, batch.column_word_lens are used
        """

        device = input_dict['question'].device

        questions, _ = rnn_wrapper(self.question_lstm, input_dict['question'], batch['lens']['question'], cell=self.cell)
        questions = questions.contiguous().view(-1, self.hidden_size)[lens2mask(batch['lens']['question']).contiguous().view(-1)]

        paragraphs, _ = rnn_wrapper(self.paragraph_lstm, input_dict['paragraph'], batch['lens']['paragraph'], cell=self.cell)
        paragraphs = paragraphs.contiguous().view(-1, self.hidden_size)[lens2mask(batch['lens']['paragraph']).contiguous().view(-1)]
        
        
        old_tables, _ = rnn_wrapper(self.table_lstm, input_dict['table'], batch['lens']['table_word'], cell=self.cell)
        old_tables = old_tables.contiguous().view(-1, self.hidden_size)[lens2mask(batch['lens']['table_word']).contiguous().view(-1)]
        t_w_t_n_mask = batch['t_w_t_n_mask']
        
        new_tables = torch.zeros([t_w_t_n_mask.size(0), t_w_t_n_mask.size(1), self.hidden_size]).to(device)
        new_tables = new_tables.masked_scatter_(t_w_t_n_mask.unsqueeze(-1), old_tables)
        tables = self.aggregation(new_tables, mask=t_w_t_n_mask)

        
        questions = questions.split(batch['lens']['question'].tolist(), dim=0)
        tables = tables.split(batch['lens']['table'].tolist(), dim=0)
        paragraphs = paragraphs.split(batch['lens']['paragraph'].tolist(), dim=0)

        outputs = [th for q_t_c in zip(questions, tables, paragraphs) for th in q_t_c]
        outputs = torch.cat(outputs, dim=0)

        return outputs



def generate_question_3split(max_seq,max_len,split=3):
    punc_list=[",","：","；","？","！","，","“","”",",",".","?","，","。","？","．","；","｡"]
    question_mask = []
    split=2
    for i in range(max_seq):
        question_mask.append([0 for _ in range(split)])
    for i in range(max_seq,max_len):
        question_mask.append([1 for _ in range(split)])
    
    split_size1 = int(max_seq * 0.5)

    for i in range(0,split_size1):
        question_mask[i][1]=1

    for i in range(split_size1,max_seq):
        question_mask[i][0]=1
    return question_mask



    
class Arithmetic(nn.Module):
    def __init__(self, hidden_size):
        super(Arithmetic, self).__init__()
        self.hidden_size = hidden_size
        embedding_size = hidden_size
        self.embedding_size = embedding_size
        self.prediction = Prediction(hidden_size, op_nums=len(OPERATOR_LIST),input_size=len(CONST_LIST2))
        self.generate = GenerateNode(hidden_size, op_nums=len(OPERATOR_LIST), embedding_size=embedding_size)
        self.merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
        
        
    def get_all_number_encoder_outputs(self, encoder_outputs, num_pos, batch_size, num_size, hidden_size):
        indices = list()
        sen_len = encoder_outputs.size(0)
        masked_index = []
        temp_1 = [1 for _ in range(hidden_size)]
        temp_0 = [0 for _ in range(hidden_size)]
        for b in range(batch_size):
            for i in num_pos[b]:
                indices.append(i + b * sen_len)
                masked_index.append(temp_0)
            indices += [0 for _ in range(len(num_pos[b]), num_size)]
            masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
        indices = torch.LongTensor(indices).cuda()
        masked_index = torch.ByteTensor(masked_index).cuda()
        masked_index = masked_index.view(batch_size, num_size, hidden_size)
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()
        all_embedding = all_outputs.view(-1, encoder_outputs.size(2))
        all_num = all_embedding.index_select(0, indices)
        all_num = all_num.view(batch_size, num_size, hidden_size)
        masked_index=masked_index.bool()
        
        return all_num.masked_fill_(masked_index, 0.0),indices,masked_index
    
    def forward(self, sequence_output, hidden, batch, arithmetic_labels):
        target_batch = [output for i, output in enumerate(batch['outputs']) if arithmetic_labels[i]==1]

        cls_output = sequence_output[:, 0, :]
        batch_size = cls_output.size(0)

        target_length = [len(item) for item in target_batch]
        max_target_length = max([len(item) for item in target_batch])
        
        all_node_outputs = []
        
        question_lens = batch['lens']['question'].masked_select(torch.tensor(arithmetic_labels).bool())
        table_lens = batch['lens']['table'].masked_select(torch.tensor(arithmetic_labels).bool())
        paragraph_lens = batch['lens']['paragraph'].masked_select(torch.tensor(arithmetic_labels).bool())
        b_lens = torch.zeros([batch_size]).cuda()

        table_number_index = [output for i, output in enumerate(batch['table_number_index']) if arithmetic_labels[i]==1]
        paragraph_number_index = [output for i, output in enumerate(batch['paragraph_number_index']) if arithmetic_labels[i]==1]

        num_pos = []
        for i in range(batch_size):
            numbers = []
            numbers += [batch_number.item() + question_lens[i].item() for batch_number in table_number_index[i]]
            numbers += [batch_number.item() +question_lens[i].item()+table_lens[i].item() for batch_number in paragraph_number_index[i]]
            num_pos.append(numbers)

        copy_num_len = [len(_) for _ in num_pos]
        max_num_size = max(copy_num_len)
        
        for i in range(batch_size):
            b_lens[i] = question_lens[i] + table_lens[i] + paragraph_lens[i]
        
        b_lens = list(map(int, b_lens.tolist()))
        max_len = max(b_lens)
        
        change_size = False
        if max_len != hidden.size(1):
            change_size = True
            
            
        if change_size:
            hidden = hidden[:, :max_len, :]
        encoder_outputs = hidden.transpose(0,1).contiguous()
        

        all_nums_encoder_outputs, indices, masked_index = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, max_num_size,self.hidden_size)
        
        
        padding_hidden = torch.zeros([1, self.hidden_size]).cuda()
        
        b_mask = batch['mask']['b'].masked_select(torch.tensor(arithmetic_labels).unsqueeze(-1).bool()).view(batch_size, -1)
        if change_size:
            b_mask = b_mask[:, :max_len]
        
        seq_mask = (b_mask != False).to(dtype=torch.uint8)
        
        num_mask = []
        max_num_size = max_num_size + len(CONST_LIST2)
        for i in copy_num_len:
            d = i + len(CONST_LIST2)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.ByteTensor(num_mask).cuda()
        
        
        question_mask = []

        for b in range(batch_size):
            ques_mask_this = generate_question_3split(b_lens[b], max_len)
            question_mask.append(ques_mask_this)
        question_mask = torch.ByteTensor(question_mask).cuda()

        
        nums_stack_batch = [[] for _ in range(batch_size)]
        
        for target_batch_i in target_batch:
            target_batch_i.extend([0 for _ in range(max_target_length-len(target_batch_i))])

        target = torch.LongTensor(target_batch).cuda().transpose(0, 1)

        node_stacks = [[TreeNode(_)] for _ in cls_output.split(1, dim=0)]
        num_start = len(OPERATOR_LIST)
        
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]

        for t in range(max_target_length):

            num_score, op, current_embeddings, current_context, current_nums_embeddings,current_attn = self.prediction(
                node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask,indices,masked_index,question_mask)

            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start)
            target[t] = target_t
            left_child, right_child, node_label = self.generate(current_embeddings, generate_input, current_context)
            left_childs = []
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue
                # 是运算符
                if i < num_start:
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                # 是数字
                else:
                    current_num = current_nums_embeddings[idx, i- num_start].unsqueeze(0)
                    while len(o) > 0 and o[-1].terminal:
                        sub_stree = o.pop()
                        op = o.pop()
                        current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)
        
        try:
            all_node_outputs = torch.stack(all_node_outputs, dim=1)
        except:
            import pdb; pdb.set_trace()

        target = target.transpose(0, 1).contiguous()

        loss = masked_cross_entropy(all_node_outputs, target, target_length)
        return loss, all_node_outputs, target

    def predict(self, sequence_output, hidden, batch, arithmetic_labels, predicted_scale_class, gold_answer):
        MAX_LENGTH = 28
        beam_size = 3
        

        cls_output = sequence_output[:, 0, :]
        batch_size = cls_output.size(0)
        
        question_lens = batch['lens']['question'].masked_select(torch.tensor(arithmetic_labels).bool())
        table_lens = batch['lens']['table'].masked_select(torch.tensor(arithmetic_labels).bool())
        paragraph_lens = batch['lens']['paragraph'].masked_select(torch.tensor(arithmetic_labels).bool())
        b_lens = torch.zeros([batch_size]).cuda()

        table_number_index = [output for i, output in enumerate(batch['table_number_index']) if arithmetic_labels[i]==1]
        paragraph_number_index = [output for i, output in enumerate(batch['paragraph_number_index']) if arithmetic_labels[i]==1]
        b_tokens = [output for i, output in enumerate(batch['b_tokens']) if arithmetic_labels[i]==1]
        b_question_lens = [output for i, output in enumerate(batch['lens']['question']) if arithmetic_labels[i]==1]
        num_pos = []
        for i in range(batch_size):
            numbers = []
            numbers += [batch_number.item() + question_lens[i].item() for batch_number in table_number_index[i]]
            numbers += [batch_number.item() +question_lens[i].item()+table_lens[i].item() for batch_number in paragraph_number_index[i]]
            num_pos.append(numbers)
        
        copy_num_len = [len(_) for _ in num_pos]
        max_num_size = max(copy_num_len)
        
        for i in range(batch_size):
            b_lens[i] = question_lens[i] + table_lens[i] + paragraph_lens[i]
        
        b_lens = list(map(int, b_lens.tolist()))
        max_len = max(b_lens)
        
        change_size = False
        if max_len != hidden.size(1):
            change_size = True
            
            
        if change_size:
            hidden = hidden[:, :max_len, :]
        encoder_outputs = hidden.transpose(0,1).contiguous()
        

        all_nums_encoder_outputs, indices, masked_index = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, max_num_size,self.hidden_size)
        
        
        padding_hidden = torch.zeros([1, self.hidden_size]).cuda()
        
        b_mask = batch['mask']['b'].masked_select(torch.tensor(arithmetic_labels).unsqueeze(-1).bool()).view(batch_size, -1)
        if change_size:
            b_mask = b_mask[:, :max_len]
        
        seq_mask = (b_mask != False).to(dtype=torch.uint8)
        
        num_mask = []
        max_num_size = max_num_size + len(CONST_LIST2)
        for i in copy_num_len:
            d = i + len(CONST_LIST2)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.ByteTensor(num_mask).cuda()
        
        
        question_mask = []

        for b in range(batch_size):
            ques_mask_this = generate_question_3split(b_lens[b], max_len)
            question_mask.append(ques_mask_this)
        question_mask = torch.ByteTensor(question_mask).cuda()



        node_stacks = [[TreeNode(_)] for _ in cls_output.split(1, dim=0)]
        num_start = len(OPERATOR_LIST)
        
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        
        beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]
        
        for t in range(MAX_LENGTH):
            current_beams = []

            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                
                left_childs = b.left_childs
                            # num_score [B, 2 + num_num]  op [B, op_num]  current_embeddings [B,1,H]  current_context [B,1,H] current_nums_embeddings [B, 2+num_num,H]
                num_score, op, current_embeddings, current_context, current_nums_embeddings,current_attn = self.prediction(
                    b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                    seq_mask, num_mask,indices, masked_index, question_mask)
                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
                # topv 分数    topi 下标
                topv, topi = out_score.topk(beam_size)
                
                
                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)

                    out_token = int(ti)
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    if out_token < num_start:
                        generate_input = torch.LongTensor([out_token]).cuda()
                        left_child, right_child, node_label = self.generate(current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                                current_left_childs, current_out))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break
        del seq_mask,padding_hidden,num_mask
        torch.cuda.empty_cache()
        predict_answer_id = beams[0].out
        numbers = OPERATOR_LIST + CONST_LIST2 + batch['question_number'][arithmetic_labels.index(1)]  + batch['table_number'][arithmetic_labels.index(1)] + batch['paragraph_number'][arithmetic_labels.index(1)]
        question_id = batch['question_id'][arithmetic_labels.index(1)]
        
        numbers_scale = [-1 for _ in range(len(OPERATOR_LIST)+len(CONST_LIST2))] + batch['t_scale'][0].tolist() + batch['p_scale'][0].tolist()
        zero_count = predict_answer_id.count(0)
        
        if len(predict_answer_id) == 0 or predict_answer_id is None or zero_count == len(predict_answer_id):
            answer = -1
            return answer
        else:
            while predict_answer_id[-1] == 0:
                predict_answer_id.pop(-1)
            if len(predict_answer_id) == 1 and predict_answer_id[0] >= len(OPERATOR_LIST) + len(CONST_LIST2)+ len(batch['question_number'][arithmetic_labels.index(1)]) and predict_answer_id[0] < len(OPERATOR_LIST) + len(CONST_LIST2) + len(batch['question_number'][arithmetic_labels.index(1)]) + len(batch['table_number'][arithmetic_labels.index(1)]):
                answer_id = predict_answer_id[0]-len(OPERATOR_LIST) - len(CONST_LIST2) - len(batch['question_number'][arithmetic_labels.index(1)])
                table_token_id = table_number_index[0].tolist()[answer_id]
                answer = b_tokens[0][question_lens[0].item() + table_token_id]
                return answer
            else:
                pre_answer = list(map(lambda x: numbers[x], predict_answer_id))
                pre_answer = list(map(lambda x: str(x) if not isinstance(x, str) else x, pre_answer))

                prefix_expr = ' '.join(pre_answer)
                
                try:
                    answer = prefix_eval(prefix_expr)
                except:
                    answer = 0
            
                
                if predicted_scale_class == 4:
                    answer = np.around(answer*100.0, 2)
                else:
                    answer = np.around(answer, 2)
                # answer, gold_answer['answer'], batch['question_id'], predict_answer_id, predict_answer_id, prefix_expr
                # predict_answer_id, prefix_expr
                # try:
                #     if gold_answer['answer'] + answer == 0:
                #         import pdb; pdb.set_trace()
                # except:
                #     answer = answer
                return answer
                        


def scale_judge(predict_answer_id, numbers, numbers_scale, batch):
    predicted_scale_class = 0
    question = batch['questions'][0]
    if 'percentage' in question:
        predicted_scale_class = 4
        return predicted_scale_class
    predict_number = [id for id in predict_answer_id if id >= len(OPERATOR_LIST)+len(CONST_LIST2)]
    predict_number_scale = list(map(lambda x: numbers_scale[x], predict_number))

    predict_number_scale_set = list(set(predict_number_scale))
    if len(predict_number_scale_set) == 0:
        return predicted_scale_class
    if len(predict_number_scale_set) == 2:
        shuzhi_big = max(predict_number_scale_set)
        maxscale = max(predict_number_scale,key=predict_number_scale.count)
        if maxscale != shuzhi_big:
            predicted_scale_class = maxscale
        else:
            predicted_scale_class = shuzhi_big
        return predicted_scale_class
    if len(predict_number_scale_set) == 1:
        return predict_number_scale_set[0]
    
    return -1

class OtherModel(nn.Module):
    def __init__(self, hidden_size):
        super(OtherModel, self).__init__()
        self.hidden_size = hidden_size
        self.tag_predictor = FFNLayer(hidden_size, hidden_size, 2, 0.2)
        self.NLLLoss = nn.NLLLoss(reduction="sum")

    def forward(self, sequence_output, hidden, batch, other_labels):
        batch_size = hidden.size(0)
        question_lens = batch['lens']['question'].masked_select(torch.tensor(other_labels).bool())
        table_lens = batch['lens']['table'].masked_select(torch.tensor(other_labels).bool())
        paragraph_lens = batch['lens']['paragraph'].masked_select(torch.tensor(other_labels).bool())
        
        b_lens = torch.zeros([batch_size]).cuda()
        for i in range(batch_size):
            b_lens[i] = question_lens[i] + table_lens[i] + paragraph_lens[i]
        
        b_lens = list(map(int, b_lens.tolist()))
        max_len = max(b_lens)
        
        change_size = False
        if max_len != hidden.size(1):
            change_size = True
        
        if change_size:
            hidden = hidden[:, :max_len, :]
        
        b_mask = batch['mask']['b'].masked_select(torch.tensor(other_labels).unsqueeze(-1).bool()).view(batch_size, -1)
        question_mask = batch['mask']['question'].masked_select(torch.tensor(other_labels).unsqueeze(-1).bool()).view(batch_size, -1).long()
        table_mask = batch['mask']['table'].masked_select(torch.tensor(other_labels).unsqueeze(-1).bool()).view(batch_size, -1).long()
        paragraph_mask = batch['mask']['paragraph'].masked_select(torch.tensor(other_labels).unsqueeze(-1).bool()).view(batch_size, -1).long()
        tags = batch['b_tags'].masked_select(torch.tensor(other_labels).unsqueeze(-1).bool()).view(batch_size, -1).float()
        
        if change_size:
            b_mask = b_mask[:, :max_len]
            question_mask = question_mask[:, :max_len]
            table_mask = table_mask[:, :max_len]
            paragraph_mask = paragraph_mask[:, :max_len]
            tags = tags[:, :max_len]


        table_sequence_output = util.replace_masked_values(hidden, table_mask.unsqueeze(-1), 0)
        table_tag_prediction = self.tag_predictor(table_sequence_output)
        table_tag_prediction = util.masked_log_softmax(table_tag_prediction, mask=None)
        table_tag_prediction = util.replace_masked_values(table_tag_prediction, table_mask.unsqueeze(-1), 0)
        table_tag_labels = util.replace_masked_values(tags, table_mask, 0)
        
        paragraph_sequence_output = util.replace_masked_values(hidden, paragraph_mask.unsqueeze(-1), 0)
        paragraph_tag_prediction = self.tag_predictor(paragraph_sequence_output)
        paragraph_tag_prediction = util.masked_log_softmax(paragraph_tag_prediction, mask=None)
        paragraph_tag_prediction = util.replace_masked_values(paragraph_tag_prediction, paragraph_mask.unsqueeze(-1), 0)
        paragraph_tag_labels = util.replace_masked_values(tags, paragraph_mask, 0)


        table_tag_prediction = table_tag_prediction.transpose(1, 2)
        table_tag_prediction_loss = self.NLLLoss(table_tag_prediction, table_tag_labels.long())
        
        paragraph_tag_prediction = paragraph_tag_prediction.transpose(1, 2)
        paragraph_token_tag_prediction_loss = self.NLLLoss(paragraph_tag_prediction, paragraph_tag_labels.long())
        
        other_loss = table_tag_prediction_loss + paragraph_token_tag_prediction_loss
        
        return other_loss, table_tag_prediction, paragraph_tag_prediction


    def forward2(self, sequence_output, hidden, batch, other_labels):
        batch_size = hidden.size(0)
        question_lens = batch['lens']['question'].masked_select(torch.tensor(other_labels).bool())
        table_lens = batch['lens']['table'].masked_select(torch.tensor(other_labels).bool())
        paragraph_lens = batch['lens']['paragraph'].masked_select(torch.tensor(other_labels).bool())
        
        b_lens = torch.zeros([batch_size]).cuda()
        for i in range(batch_size):
            b_lens[i] = question_lens[i] + table_lens[i] + paragraph_lens[i]
        
        b_lens = list(map(int, b_lens.tolist()))
        max_len = max(b_lens)
        
        change_size = False
        if max_len != hidden.size(1):
            change_size = True
        
        if change_size:
            hidden = hidden[:, :max_len, :]
        
        b_mask = batch['mask']['b'].masked_select(torch.tensor(other_labels).unsqueeze(-1).bool()).view(batch_size, -1)
        question_mask = batch['mask']['question'].masked_select(torch.tensor(other_labels).unsqueeze(-1).bool()).view(batch_size, -1).long()
        table_mask = batch['mask']['table'].masked_select(torch.tensor(other_labels).unsqueeze(-1).bool()).view(batch_size, -1).long()
        paragraph_mask = batch['mask']['paragraph'].masked_select(torch.tensor(other_labels).unsqueeze(-1).bool()).view(batch_size, -1).long()
        
        if change_size:
            b_mask = b_mask[:, :max_len]
            question_mask = question_mask[:, :max_len]
            table_mask = table_mask[:, :max_len]
            paragraph_mask = paragraph_mask[:, :max_len]

        table_sequence_output = util.replace_masked_values(hidden, table_mask.unsqueeze(-1), 0)
        table_tag_prediction = self.tag_predictor(table_sequence_output)
        table_tag_prediction = util.masked_log_softmax(table_tag_prediction, mask=None)
        table_tag_prediction = util.replace_masked_values(table_tag_prediction, table_mask.unsqueeze(-1), 0)

        
        paragraph_sequence_output = util.replace_masked_values(hidden, paragraph_mask.unsqueeze(-1), 0)
        paragraph_tag_prediction = self.tag_predictor(paragraph_sequence_output)
        paragraph_tag_prediction = util.masked_log_softmax(paragraph_tag_prediction, mask=None)
        paragraph_tag_prediction = util.replace_masked_values(paragraph_tag_prediction, paragraph_mask.unsqueeze(-1), 0)

    
        table_tag_prediction = table_tag_prediction.transpose(1, 2)
        
        paragraph_tag_prediction = paragraph_tag_prediction.transpose(1, 2)
        
        other_loss = 0
        
        return other_loss, table_tag_prediction, paragraph_tag_prediction



    def predict(self, hidden, batch, other_labels, operator_class, table_tag_prediction, paragraph_tag_prediction):

        table_tag_prediction = table_tag_prediction.transpose(1, 2)
        table_tag_prediction_score = table_tag_prediction[:, :, 1]
        table_tag_prediction = torch.argmax(table_tag_prediction, dim=-1).float()
        table_tag_prediction = table_tag_prediction.detach().cpu().numpy()
        table_tag_prediction_score = table_tag_prediction_score.detach().cpu().numpy()
        
        paragraph_tag_prediction = paragraph_tag_prediction.transpose(1, 2)
        paragraph_tag_prediction_score = paragraph_tag_prediction[:, :, 1]
        paragraph_tag_prediction = torch.argmax(paragraph_tag_prediction, dim=-1).float() 
        paragraph_tag_prediction = paragraph_tag_prediction.detach().cpu().numpy()
        paragraph_tag_prediction_score = paragraph_tag_prediction_score.detach().cpu().numpy()

        global_tokens = batch['b_tokens'][other_labels.index(1)]
        answer = None
        if operator_class == OPERATOR_CLASSES["SPAN"]:
            answer = get_single_span_tokens_from_table_and_paragraph(
                table_tag_prediction,
                table_tag_prediction_score,
                paragraph_tag_prediction,
                paragraph_tag_prediction_score,
                global_tokens
            )
            answer = sorted(answer)
        elif operator_class == OPERATOR_CLASSES['MULTI_SPAN']:
            paragraph_selected_span_tokens = \
                get_span_tokens_from_paragraph(paragraph_tag_prediction, global_tokens)
            table_selected_tokens = \
                get_span_tokens_from_table(table_tag_prediction, global_tokens)
            answer =  paragraph_selected_span_tokens + table_selected_tokens
            answer = sorted(answer)
        elif operator_class == OPERATOR_CLASSES['COUNT']:
            paragraph_selected_tokens = \
                get_span_tokens_from_paragraph(paragraph_tag_prediction, global_tokens)
            table_selected_tokens = \
                get_span_tokens_from_table(table_tag_prediction, global_tokens)
            answer = len(paragraph_selected_tokens) + len(table_selected_tokens)
        
        if answer is None or answer == []:
            answer = -1
        if answer == 0 and operator_class == OPERATOR_CLASSES['COUNT']:
            answer = 1
        return answer




