r"""Modify from fastNLP"""

import torch
from torch import nn
from fastNLP.models.seq2seq_model import Seq2SeqModel
from fastNLP.modules.decoder.seq2seq_decoder import Seq2SeqDecoder, State
import torch.nn.functional as F
from fastNLP.core.utils import _get_model_device
from functools import partial

from tag_op.tagop.generator_util import get_program_mask, update_op_stack
from tag_op.data.operation_util import GRAMMER_CLASS, AUX_NUM
from tag_op.tagop.tools import allennlp as allennlp
from tag_op.tagop.tools.util import FFNLayer

class SequenceGeneratorModel(nn.Module):
    """
    用于封装Seq2SeqModel使其可以做生成任务

    """

    def __init__(self, seq2seq_model: Seq2SeqModel, bos_token_id, eos_token_id=None, max_length=30, max_len_a=0.0,
                 num_beams=1, do_sample=True,
                 repetition_penalty=1, length_penalty=1.0, pad_token_id=0,
                 restricter=None):
        """

        :param Seq2SeqModel seq2seq_model: 序列到序列模型. 会使用seq2seq_model的decoder进行生成
        :param int,None bos_token_id: 句子开头的token id
        :param int,None eos_token_id: 句子结束的token id
        :param int max_length: 生成句子的最大长度, 每句话的decode长度为max_length + max_len_a*src_len
        :param float max_len_a: 每句话的decode长度为max_length + max_len_a*src_len。 如果不为0，需要保证State中包含encoder_mask
        :param int num_beams: beam search的大小
        :param bool do_sample: 是否通过采样的方式生成
        :param float temperature: 只有在do_sample为True才有意义
        :param int top_k: 只从top_k中采样
        :param float top_p: 只从top_p的token中采样，nucles sample
        :param float repetition_penalty: 多大程度上惩罚重复的token
        :param float length_penalty: 对长度的惩罚，小于1鼓励长句，大于1鼓励短剧
        :param int pad_token_id: 当某句话生成结束之后，之后生成的内容用pad_token_id补充
        """
        super().__init__()
        self.seq2seq_model = seq2seq_model
        self.restricter = restricter
        self.generator = SequenceGenerator(seq2seq_model.decoder, max_length=max_length, max_len_a=max_len_a,
                                           num_beams=num_beams,
                                           do_sample=do_sample,
                                           bos_token_id=bos_token_id,
                                           eos_token_id=eos_token_id,
                                           repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                                           pad_token_id=pad_token_id,
                                           restricter=restricter)
        ## scale predictor
        hidden_size = seq2seq_model.encoder.bart_encoder.embed_tokens.weight.size(-1)
        self.scale_predictor = FFNLayer(1 * hidden_size, hidden_size, 5, 0.1)
        
        
        
#     def build_vm(self, src_tokens, question_mask, paragraph_mask, table_mask, all_cell_token_index, row_include_cells, col_include_cells):
#         import pdb
        
#         bsz = src_tokens.size(0)
#         seq_len = src_tokens.size(-1)
#         att_mask_vm_lower = torch.zeros([bsz, 512, 512])
#         att_mask_vm_upper = torch.zeros([bsz, 512, 512])
#         for k in range(bsz):
#             table_start = int(torch.sum(question_mask[k], dim=-1).item()) + 2
#             paragraph_start = table_start + int(torch.sum(table_mask[k], dim=-1).item())
#             att_mask_vm_lower[k][0:table_start, 0:seq_len].fill_(1)
#             att_mask_vm_upper[k][0:table_start, 0:seq_len].fill_(1)
#             att_mask_vm_lower[k][paragraph_start:seq_len, 0:seq_len].fill_(1)
#             att_mask_vm_upper[k][paragraph_start:seq_len, 0:seq_len].fill_(1)
#             att_mask_vm_lower[k][table_start:paragraph_start, 0:table_start].fill_(1)
#             att_mask_vm_lower[k][table_start:paragraph_start, paragraph_start:seq_len].fill_(1)
#             att_mask_vm_upper[k][table_start:paragraph_start, 0:table_start].fill_(1)
#             att_mask_vm_upper[k][table_start:paragraph_start, paragraph_start:seq_len].fill_(1)

#             ## lower mask
#             row = row_include_cells[k].size(0)
#             for r in range(row):
#                 tmp =[item for item in row_include_cells[k][r].tolist() if item!=0]
#                 tokens_idx_offset = [all_cell_token_index[k][cell-1].tolist() for cell in tmp]
#                 tokens_idx = []
#                 for item in tokens_idx_offset:
#                     tokens_idx.extend([i+table_start for i in range(item[0], item[1])])

#                 for i in tokens_idx:
#                     for j in tokens_idx:
#                         att_mask_vm_lower[k][i][j] = 1
#                         att_mask_vm_upper[k][i][j] = 1
#             ## upper mask     
#             import pdb
# #             pdb.set_trace()
#             col = col_include_cells[k].size(0)
#             for c in range(col):
#                 tmp =[item for item in col_include_cells[k][c].tolist() if item!=0]
#                 tokens_idx_offset = [all_cell_token_index[k][cell-1].tolist() for cell in tmp]
#                 tokens_idx = []
#                 for item in tokens_idx_offset:
#                     tokens_idx.extend([i+table_start for i in range(item[0], item[1])])

#                 for i in tokens_idx:
#                     for j in tokens_idx:
#                         att_mask_vm_upper[k][i][j] = 1
                    
#         return att_mask_vm_lower, att_mask_vm_upper
    
    
    def forward(self, src_tokens, tgt_tokens, src_seq_len=None, tgt_seq_len=None, first=None, question_mask=None, table_mask=None, table_cell_index=None, all_cell_index=None, all_cell_index_mask=None, paragraph_mask=None, row_include_cells=None, col_include_cells=None, all_cell_token_index=None,att_mask_vm_lower=None, att_mask_vm_upper=None):
        """
        透传调用seq2seq_model的forward

        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor tgt_tokens: bsz x max_len'
        :param torch.LongTensor src_seq_len: bsz
        :param torch.LongTensor tgt_seq_len: bsz
        :return:
        """
        '''
        scale forward
        '''
        import pdb
#         pdb.set_trace()
#         att_mask_vm_lower, att_mask_vm_upper = self.build_vm(src_tokens, question_mask, paragraph_mask, table_mask, all_cell_token_index, row_include_cells, col_include_cells)
        
        encoder_outputs, mask, hidden_states = self.seq2seq_model.encoder(src_tokens, src_seq_len, question_mask, table_mask, paragraph_mask, table_cell_index, all_cell_index, all_cell_index_mask, row_include_cells, col_include_cells, all_cell_token_index, att_mask_vm_lower, att_mask_vm_upper)
        cls_output = encoder_outputs[:, 0,:]
        '''
        table_sequence_output = allennlp.replace_masked_values(encoder_outputs, table_mask.unsqueeze(-1)[:, :encoder_outputs.size(1), :], 0)
        table_reduce_mean = torch.mean(table_sequence_output , dim=1)
        paragraph_sequence_output = allennlp.replace_masked_values(encoder_outputs, paragraph_mask.unsqueeze(-1)[:, :encoder_outputs.size(1), :], 0)
        paragraph_reduce_mean = torch.mean(paragraph_sequence_output, dim=1)
        cls_output = torch.cat((cls_output, table_reduce_mean, paragraph_reduce_mean), dim=-1)
        '''
        scale_prediction_logits = self.scale_predictor(cls_output)
        decode_logits = self.seq2seq_model(src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first,  question_mask, table_mask,paragraph_mask, table_cell_index, all_cell_index, all_cell_index_mask,row_include_cells, col_include_cells, all_cell_token_index, att_mask_vm_lower, att_mask_vm_upper)
        logits = {'pred':(scale_prediction_logits, decode_logits['pred'])}
        
        return logits

    def predict(self, src_tokens, src_seq_len=None, first=None, question_id = None, table_mask=None,paragraph_mask=None, table_cell_number_value=None, paragraph_number_value=None, table_cell_index=None, paragraph_index=None, row_include_cells=None, col_include_cells=None, question_mask=None,  all_cell_index=None, all_cell_index_mask=None,  all_cell_token_index=None, att_mask_vm_lower=None,att_mask_vm_upper=None):
        """
        给定source的内容，输出generate的内容

        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor src_seq_len: bsz
        :return:
        """
#         att_mask_vm_lower, att_mask_vm_upper = self.build_vm(src_tokens, question_mask, paragraph_mask, table_mask, all_cell_token_index, row_include_cells, col_include_cells)
        state = self.seq2seq_model.prepare_state(src_tokens, src_seq_len, first, None, question_mask, table_mask, paragraph_mask, table_cell_index, all_cell_index, all_cell_index_mask,row_include_cells, col_include_cells, all_cell_token_index, att_mask_vm_lower, att_mask_vm_upper)
        assert table_cell_index.size(1)==512
        result, scores = self.generator.generate(state, None, question_id, table_mask, paragraph_mask, table_cell_number_value, paragraph_number_value, table_cell_index, paragraph_index, row_include_cells, col_include_cells)
        
        import pdb
        encoder_outputs, mask, hidden_states = self.seq2seq_model.encoder(src_tokens, src_seq_len, question_mask, table_mask, paragraph_mask, table_cell_index, all_cell_index, all_cell_index_mask,row_include_cells, col_include_cells,  all_cell_token_index, att_mask_vm_lower, att_mask_vm_upper)
        cls_output= encoder_outputs[:,0,:]
        '''
        table_sequence_output = allennlp.replace_masked_values(encoder_outputs, table_mask.unsqueeze(-1)[:, :encoder_outputs.size(1), :], 0)
        table_reduce_mean = torch.mean(table_sequence_output , dim=1)
        paragraph_sequence_output = allennlp.replace_masked_values(encoder_outputs, paragraph_mask.unsqueeze(-1)[:, :encoder_outputs.size(1), :], 0)
        paragraph_reduce_mean = torch.mean(paragraph_sequence_output, dim=1)
        cls_output = torch.cat((cls_output, table_reduce_mean, paragraph_reduce_mean), dim=-1)
        '''
        scale_prediction_logits = self.scale_predictor(cls_output)
        scale_prediction = F.softmax(scale_prediction_logits)
        
        _, pred_scale = torch.max(scale_prediction, 1)
        return {'pred': (result, pred_scale, scores)}

r"""

"""

__all__ = [
    'SequenceGenerator'
]



class SequenceGenerator:
    """
    给定一个Seq2SeqDecoder，decode出句子

    """
    def __init__(self, decoder: Seq2SeqDecoder, max_length=20, max_len_a=0.0, num_beams=1,
                 do_sample=False, bos_token_id=None, eos_token_id=None,
                 repetition_penalty=1, length_penalty=1.0, pad_token_id=0, restricter=None):
        """

        :param Seq2SeqDecoder decoder: Decoder对象
        :param int max_length: 生成句子的最大长度, 每句话的decode长度为max_length + max_len_a*src_len
        :param float max_len_a: 每句话的decode长度为max_length + max_len_a*src_len。 如果不为0，需要保证State中包含encoder_mask
        :param int num_beams: beam search的大小
        :param bool do_sample: 是否通过采样的方式生成
        :param float temperature: 只有在do_sample为True才有意义
        :param int top_k: 只从top_k中采样
        :param float top_p: 只从top_p的token中采样，nucles sample
        :param int,None bos_token_id: 句子开头的token id
        :param int,None eos_token_id: 句子结束的token id
        :param float repetition_penalty: 多大程度上惩罚重复的token
        :param float length_penalty: 对长度的惩罚，小于1鼓励长句，大于1鼓励短剧
        :param int pad_token_id: 当某句话生成结束之后，之后生成的内容用pad_token_id补充
        """
        self.generate_func = partial(greedy_generate, decoder=decoder, max_length=max_length, max_len_a=max_len_a,
                                     num_beams=num_beams,
                                     bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                                     repetition_penalty=repetition_penalty,
                                     length_penalty=length_penalty, pad_token_id=pad_token_id,
                                     restricter=restricter)
        self.do_sample = do_sample
        self.max_length = max_length
        self.num_beams = num_beams
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.decoder = decoder
        self.pad_token_id = pad_token_id
        self.restricter = restricter
        self.max_len_a = max_len_a

    def set_new_generator(self, max_length=-1, max_len_a=-1, num_beams=-1,
                          repetition_penalty=-1, length_penalty=-1, restricter=-1):
        if max_length == -1:
            max_length = self.max_length
        if max_len_a == -1:
            max_len_a = self.max_len_a
        if num_beams == -1:
            num_beams = self.num_beams
        if repetition_penalty == -1:
            repetition_penalty = self.repetition_penalty
        if length_penalty == -1:
            length_penalty = self.length_penalty
        if restricter == -1:
            restricter = self.restricter
        self.generate_func = partial(greedy_generate, decoder=self.decoder, max_length=max_length, max_len_a=max_len_a,
                                     num_beams=num_beams,
                                     bos_token_id=self.bos_token_id, eos_token_id=self.eos_token_id,
                                     repetition_penalty=repetition_penalty,
                                     length_penalty=length_penalty, pad_token_id=self.pad_token_id,
                                     restricter=restricter)

    @torch.no_grad()
    def generate(self, state, tokens=None, question_id=None, table_mask=None, paragraph_mask=None, table_cell_number_value=None, paragraph_number_value=None, table_cell_index=None, paragraph_index=None, row_include_cells=None, col_include_cells=None):
        """

        :param State state: encoder结果的State, 是与Decoder配套是用的
        :param torch.LongTensor,None tokens: batch_size x length, 开始的token
        :return: bsz x max_length' 生成的token序列。如果eos_token_id不为None, 每个sequence的结尾一定是eos_token_id
        """
        return self.generate_func(tokens=tokens, state=state, question_id=question_id, table_mask=table_mask, paragraph_mask=paragraph_mask, table_cell_number_value=table_cell_number_value, paragraph_number_value=paragraph_number_value, table_cell_index=table_cell_index, paragraph_index=paragraph_index, row_include_cells=row_include_cells, col_include_cells=col_include_cells)

@torch.no_grad()
def greedy_generate(decoder, tokens=None, state=None, question_id=None, table_mask=None, paragraph_mask=None, table_cell_number_value=None, paragraph_number_value=None,table_cell_index=None, paragraph_index=None, row_include_cells=None, col_include_cells=None, max_length=20, max_len_a=0.0, num_beams=1,
                    bos_token_id=None, eos_token_id=None, pad_token_id=0,
                    repetition_penalty=1, length_penalty=1.0, restricter=None):
    """
    贪婪地搜索句子

    :param Decoder decoder: Decoder对象
    :param torch.LongTensor tokens: batch_size x len, decode的输入值，如果为None，则自动从bos_token_id开始生成
    :param State state: 应该包含encoder的一些输出。
    :param int max_length: 生成句子的最大长度, 每句话的decode长度为max_length + max_len_a*src_len
    :param float max_len_a: 每句话的decode长度为max_length + max_len_a*src_len。 如果不为0，需要保证State中包含encoder_mask
    :param int num_beams: 使用多大的beam进行解码。
    :param int bos_token_id: 如果tokens传入为None，则使用bos_token_id开始往后解码。
    :param int eos_token_id: 结束的token，如果为None，则一定会解码到max_length这么长。
    :param int pad_token_id: pad的token id
    :param float repetition_penalty: 对重复出现的token多大的惩罚。
    :param float length_penalty: 对每个token（除了eos）按照长度进行一定的惩罚。
    :return:
    """
    
    import pdb
#     pdb.set_trace()
    if num_beams == 1:
        token_ids = _no_beam_search_generate(decoder, tokens=tokens, state=state, max_length=max_length, max_len_a=max_len_a,
                                             bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                                             repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                                             pad_token_id=pad_token_id, restricter=restricter)
    else:
        token_ids, scores = _beam_search_generate(decoder, tokens=tokens, state=state, question_id=question_id, table_mask=table_mask, paragraph_mask=paragraph_mask, table_cell_number_value=table_cell_number_value, paragraph_number_value=paragraph_number_value, table_cell_index=table_cell_index, paragraph_index = paragraph_index, row_include_cells=row_include_cells, col_include_cells=col_include_cells, max_length=max_length, max_len_a=max_len_a,num_beams=num_beams, bos_token_id=bos_token_id, eos_token_id=eos_token_id, do_sample=False, repetition_penalty=repetition_penalty, length_penalty=length_penalty,pad_token_id=pad_token_id, restricter=restricter)

    return token_ids, scores


def _no_beam_search_generate(decoder: Seq2SeqDecoder, state, tokens=None, max_length=20, max_len_a=0.0, bos_token_id=None,
                             eos_token_id=None,
                             repetition_penalty=1.0, length_penalty=1.0, pad_token_id=0,
                             restricter=None):
    device = _get_model_device(decoder)
    if tokens is None:
        if bos_token_id is None:
            raise RuntimeError("You have to specify either `tokens` or `bos_token_id`.")
        batch_size = state.num_samples
        if batch_size is None:
            raise RuntimeError("Cannot infer the number of samples from `state`.")
        tokens = torch.full([batch_size, 1], fill_value=bos_token_id, dtype=torch.long).to(device)
    batch_size = tokens.size(0)
    if state.num_samples:
        assert state.num_samples == batch_size, "The number of samples in `tokens` and `state` should match."

    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id

    scores = decoder.decode(tokens=tokens, state=state)  # 主要是为了update state
    # 这里需要考虑如果在第一个位置就结束的情况
    # if _eos_token_id!=-1:
    #     scores[:, _eos_token_id] = -1e12

    if restricter is not None:
        _, next_tokens = restricter(state, tokens, scores, num_beams=1)
    else:
        next_tokens = scores.argmax(dim=-1, keepdim=True)
    token_ids = torch.cat([tokens, next_tokens], dim=1)
    cur_len = token_ids.size(1)
    dones = token_ids.new_zeros(batch_size).eq(1).__or__(next_tokens.squeeze(1).eq(eos_token_id))
    # tokens = tokens[:, -1:]

    if max_len_a!=0:
        # (bsz x num_beams, )
        if state.encoder_mask is not None:
            max_lengths = (state.encoder_mask.sum(dim=1).float()*max_len_a).long() + max_length
        else:
            max_lengths = tokens.new_full((tokens.size(0), ), fill_value=max_length, dtype=torch.long)
        real_max_length = max_lengths.max().item()
    else:
        real_max_length = max_length
        if state.encoder_mask is not None:
            max_lengths = state.encoder_mask.new_ones(state.encoder_mask.size(0)).long()*max_length
        else:
            max_lengths = tokens.new_full((tokens.size(0),), fill_value=max_length, dtype=torch.long)

    while cur_len < real_max_length:
        scores = decoder.decode(tokens=token_ids, state=state)  # batch_size x vocab_size

        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            scores.scatter_(dim=1, index=token_ids, src=token_scores)

        if eos_token_id is not None and length_penalty != 1.0:
            token_scores = scores / cur_len ** length_penalty  # batch_size x vocab_size
            eos_mask = scores.new_ones(scores.size(1))
            eos_mask[eos_token_id] = 0
            eos_mask = eos_mask.unsqueeze(0).eq(1)
            scores = scores.masked_scatter(eos_mask, token_scores)  # 也即除了eos，其他词的分数经过了放大/缩小

        if restricter is not None:
            _, next_tokens = restricter(state, token_ids, scores, 1)
        else:
            next_tokens = scores.argmax(dim=-1, keepdim=True)
        next_tokens = next_tokens.squeeze(-1)

        # 如果已经达到对应的sequence长度了，就直接填为eos了
        if _eos_token_id!=-1:
            next_tokens = next_tokens.masked_fill(max_lengths.eq(cur_len+1), _eos_token_id)
        next_tokens = next_tokens.masked_fill(dones, pad_token_id)  # 对已经搜索完成的sample做padding
        tokens = next_tokens.unsqueeze(1)

        token_ids = torch.cat([token_ids, tokens], dim=-1)  # batch_size x max_len

        end_mask = next_tokens.eq(_eos_token_id)
        dones = dones.__or__(end_mask)
        cur_len += 1

        if dones.min() == 1:
            break

    # if eos_token_id is not None:
    #     tokens.scatter(index=max_lengths[:, None], dim=1, value=eos_token_id)  # 将最大长度位置设置为eos
    # if cur_len == max_length:
    #     token_ids[:, -1].masked_fill_(~dones, eos_token_id)  # 若到最长长度仍未到EOS，则强制将最后一个词替换成eos
    return token_ids


def _beam_search_generate(decoder: Seq2SeqDecoder, tokens=None, state=None, question_id=None, table_mask=None, paragraph_mask=None, table_cell_number_value=None, paragraph_number_value=None, table_cell_index=None, paragraph_index=None, row_include_cells=None, col_include_cells=None,  max_length=20, max_len_a=0.0, num_beams=4,
                          bos_token_id=None, eos_token_id=None, do_sample=True,
                          repetition_penalty=1.0, length_penalty=None, pad_token_id=0,
                          restricter=None) -> torch.LongTensor:
    assert do_sample is False
    import pdb
    
    # 进行beam search
    device = _get_model_device(decoder)
    if tokens is None:
        if bos_token_id is None:
            raise RuntimeError("You have to specify either `tokens` or `bos_token_id`.")
        batch_size = state.num_samples
        if batch_size is None:
            raise RuntimeError("Cannot infer the number of samples from `state`.")
        tokens = torch.full([batch_size, 1], fill_value=bos_token_id, dtype=torch.long).to(device)
    batch_size = tokens.size(0)
    if state.num_samples:
        assert state.num_samples == batch_size, "The number of samples in `tokens` and `state` should match."

    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id

    scores = decoder.decode(tokens=tokens, state=state)  # 这里要传入的是整个句子的长度
    # 这里需要考虑如果在第一个位置就结束的情况
    # if _eos_token_id!=-1:
    #     scores[:, _eos_token_id] = -1e12
    vocab_size = scores.size(1)
    assert vocab_size >= num_beams, "num_beams should be smaller than the number of vocabulary size."

    
    ## ==========================================compute a decoder mask ==========================================
    import pdb
    scores = F.log_softmax(scores, dim=-1)  # (batch_size, vocab_size)
    
    with_contraints=True
    if with_contraints:
        decoded_mask= []
        for i in range(batch_size):
            current_tokens = []
            current_op_stack = []
            id = i
            current_table_mask = table_mask[id].tolist()
            current_paragraph_mask = paragraph_mask[id].tolist()
    #         if not torch.is_tensor(table_cell_number_value[id]):
            current_table_cell_number_value = table_cell_number_value[id].tolist()
            current_paragraph_number_value = paragraph_number_value[id].tolist()
            current_table_cell_index = table_cell_index[id].tolist()
            current_paragraph_index = paragraph_index[id].tolist()
            current_row_include_cells = row_include_cells[id].tolist()
            current_col_include_cells = col_include_cells[id].tolist()

            current_question_id = question_id[id]
            current_decoded_mask = get_program_mask(current_tokens, current_op_stack, GRAMMER_CLASS, AUX_NUM, vocab_size, current_question_id, current_table_mask, current_paragraph_mask, current_table_cell_number_value, current_paragraph_number_value,current_table_cell_index, current_paragraph_index, current_row_include_cells, current_col_include_cells, target_shift=2) 
            current_decoded_mask = torch.tensor(current_decoded_mask, dtype=torch.long)
            decoded_mask.append(current_decoded_mask)
        decoded_mask = torch.stack(decoded_mask, 0).to(device)
        scores = allennlp.replace_masked_values(scores, decoded_mask, -1e24)
    ## ==========================================compute a decoder mask ==========================================
    
    # 得到(batch_size, num_beams), (batch_size, num_beams)
    # TODO 把限制写到这个位置, 加1是因为需要考虑输出就是eos的情况
    if restricter is not None:
        _next_scores, _next_tokens = restricter(state, tokens, scores, num_beams+1)
    else:
        # 是bsz x (num_beams+1)大小的东西
        _next_scores, _next_tokens = torch.topk(scores, num_beams+1, dim=1, largest=True, sorted=True)

    # 根据index来做顺序的调转
    indices = torch.arange(batch_size, dtype=torch.long).to(device)
    indices = indices.repeat_interleave(num_beams)
    
    state.reorder_state(indices) ## ?
    tokens = tokens.index_select(dim=0, index=indices)  # batch_size * num_beams x length

    # if hasattr(state, 'tgt_seq_len'):  # TODO 应该需要删除
    #     max_lengths = state.tgt_seq_len
    #     real_max_length = max_lengths.max().item()
    if max_len_a!=0:
        # (bsz x num_beams, )
        if state.encoder_mask is not None:
            max_lengths = (state.encoder_mask.sum(dim=1).float()*max_len_a).long() + max_length
        else:
            max_lengths = tokens.new_full((batch_size*num_beams, ), fill_value=max_length, dtype=torch.long)
        real_max_length = max_lengths.max().item()
    else:
        real_max_length = max_length
        if state.encoder_mask is not None:
            max_lengths = state.encoder_mask.new_ones(state.encoder_mask.size(0)).long()*max_length
        else:
            max_lengths = tokens.new_full((batch_size*num_beams,), fill_value=max_length, dtype=torch.long)
    hypos = [
        BeamHypotheses(num_beams, real_max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
    ]

#     pdb.set_trace()
    not_eos_mask = _next_tokens.ne(_eos_token_id)  # 为1的地方不是eos
    keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)  # 为1的地方需要保留
    keep_mask = not_eos_mask.__and__(keep_mask)  # 为1的地方是需要进行下一步search的

    next_tokens = _next_tokens.masked_select(keep_mask).view(batch_size, num_beams)  # 这是真的接下来要继续的
    next_scores = _next_scores.masked_select(keep_mask).view(batch_size, num_beams)

    rows, cols = not_eos_mask.eq(0)[:, :num_beams].nonzero(as_tuple=True)

    if len(rows)>0:  # 说明有的开头就结束了
        for row, col in zip(rows.tolist(), cols.tolist()):
            _token = torch.cat([tokens[row*num_beams], _next_tokens[row, col:col+1]], dim=0)
            hypos[row].add(_token.clone(), _next_scores[row, col].item())

    # 记录生成好的token (batch_size', cur_len)
    token_ids = torch.cat([tokens, next_tokens.view(-1, 1)], dim=-1)
    dones = [False] * batch_size
    
    ### ================================ update op_stack ================================ ##
    if with_contraints:
        import pdb
        all_op_stack = []
        for i in range(batch_size*num_beams):
            current_token_ids = token_ids[i][1:].tolist()
            current_op_stack = update_op_stack(current_token_ids, GRAMMER_CLASS, AUX_NUM, vocab_size, target_shift=2)
            all_op_stack.append(current_op_stack)
    ### ================================ update op_stack ================================ ##
    
    beam_scores = next_scores.view(-1)  # batch_size * num_beams
    #  用来记录已经生成好的token的长度
    cur_len = token_ids.size(1)
    # 0, num_beams, 2*num_beams, ...
    batch_inds_with_numbeams_interval = (torch.arange(batch_size) * num_beams).view(-1, 1).to(token_ids)

    while cur_len < real_max_length:
        scores = decoder.decode(token_ids, state)  # (bsz x num_beams, vocab_size)
        import pdb
       
        
        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            scores.scatter_(dim=1, index=token_ids, src=token_scores)

        if _eos_token_id!=-1:
            max_len_eos_mask = max_lengths.eq(cur_len+1)
            eos_scores = scores[:, _eos_token_id]
            # 如果已经达到最大长度，就把eos的分数加大
            scores[:, _eos_token_id] = torch.where(max_len_eos_mask, eos_scores+1e32, eos_scores)
        
        scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
        
        ## ============================compute next token score mask  ============================##
        _scores = []
        if with_contraints:
            for i in range(batch_size*num_beams):
                current_tokens = token_ids[i][1:].tolist()
                current_op_stack = all_op_stack[i]
                id = i//num_beams
                current_table_mask = table_mask[id].tolist()
                current_paragraph_mask = paragraph_mask[id].tolist()
                current_table_cell_number_value = table_cell_number_value[id].tolist()
                current_paragraph_number_value = paragraph_number_value[id].tolist()
                current_table_cell_index = table_cell_index[id].tolist()
                current_paragraph_index = paragraph_index[id].tolist()
                current_row_include_cells = row_include_cells[id].tolist()
                current_col_include_cells = col_include_cells[id].tolist()
                current_question_id = question_id[id]

                current_decoded_mask = get_program_mask(current_tokens, current_op_stack, GRAMMER_CLASS, AUX_NUM, vocab_size, current_question_id, current_table_mask, current_paragraph_mask, current_table_cell_number_value, current_paragraph_number_value, current_table_cell_index, current_paragraph_index, current_row_include_cells, current_col_include_cells, target_shift=2) 
                current_decoded_mask = torch.tensor(current_decoded_mask, dtype=torch.long)

                current_scores = allennlp.replace_masked_values(scores[i], current_decoded_mask.to(device), -1e24 )
                _scores.append(current_scores)

    #         
            scores = torch.stack(_scores, dim=0)
    ## ============================compute next token score mask  ============================##

        _scores = scores + beam_scores[:, None]  # (batch_size * num_beams, vocab_size)
        _scores = _scores.view(batch_size, -1)  # (batch_size, num_beams*vocab_size)
        

        
        # TODO 把限制加到这个位置
        if restricter is not None:
            next_scores, ids = restricter(state, token_ids, _scores, 2 * num_beams)
        else:
            next_scores, ids = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)  # (bsz, 2*num_beams)
            
        from_which_beam = ids // vocab_size  # (batch_size, 2*num_beams)
        next_tokens = ids % vocab_size  # (batch_size, 2*num_beams)

        #  接下来需要组装下一个batch的结果。
        #  需要选定哪些留下来
        # next_scores, sorted_inds = next_scores.sort(dim=-1, descending=True)
        # next_tokens = next_tokens.gather(dim=1, index=sorted_inds)
        # from_which_beam = from_which_beam.gather(dim=1, index=sorted_inds)
        

        
        import pdb
        not_eos_mask = next_tokens.ne(_eos_token_id)  # 为1的地方不是eos
        keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)  # 为1的地方需要保留
        keep_mask = not_eos_mask.__and__(keep_mask)  # 为1的地方是需要进行下一步search的
        
        _next_tokens = next_tokens.masked_select(keep_mask).view(-1, 1)
        _from_which_beam = from_which_beam.masked_select(keep_mask).view(batch_size, num_beams)  # 上面的token是来自哪个beam
        _next_scores = next_scores.masked_select(keep_mask).view(batch_size, num_beams)
        beam_scores = _next_scores.view(-1)
        
        
        '''
        if generate the illegal operation
        '''
        ## =========================== if generate the illegal operation =====================
        illegal = False
        for i in range(batch_size):
            for j in range(num_beams):
                if _next_scores[i, j]<=-1e24:
                    _next_tokens[i*num_beams+j] = 1
                    illegal = True
        ## =========================== if generate the illegal operation =====================
        
        if illegal:
            import pdb
#             pdb.set_trace()
        
        
        
        flag = True
        if cur_len+1 == real_max_length:
            eos_batch_idx = torch.arange(batch_size).to(next_tokens).repeat_interleave(repeats=num_beams, dim=0)
            eos_beam_ind = torch.arange(num_beams).to(token_ids).repeat(batch_size)  # 表示的是indice
            eos_beam_idx = from_which_beam[:, :num_beams].reshape(-1)  # 表示的是从哪个beam获取得到的
        else:
            # 将每个batch中在num_beam内的序列添加到结束中, 为1的地方需要结束了

            effective_eos_mask = next_tokens[:, :num_beams].eq(_eos_token_id)  # batch_size x num_beams
            if effective_eos_mask.sum().gt(0):
                eos_batch_idx, eos_beam_ind = effective_eos_mask.nonzero(as_tuple=True)
                # 是由于from_which_beam是 (batch_size, 2*num_beams)的，所以需要2*num_beams
                eos_beam_idx = eos_batch_idx * num_beams * 2 + eos_beam_ind
                eos_beam_idx = from_which_beam.view(-1)[eos_beam_idx]  # 获取真实的从哪个beam获取的eos
            else:
                flag = False

        if flag:
            _token_ids = torch.cat([token_ids, _next_tokens], dim=-1)
            for batch_idx, beam_ind, beam_idx in zip(eos_batch_idx.tolist(), eos_beam_ind.tolist(),
                                                     eos_beam_idx.tolist()):
                if not dones[batch_idx]:
                    score = next_scores[batch_idx, beam_ind].item()
                    # 之后需要在结尾新增一个eos
                    if _eos_token_id!=-1:
                        hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx, :cur_len].clone(), score)
                    else:
                        hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx].clone(), score)

        # 更改state状态, 重组token_ids
        reorder_inds = (batch_inds_with_numbeams_interval + _from_which_beam).view(-1)  # flatten成一维
        state.reorder_state(reorder_inds)
        # 重新组织token_ids的状态
        import pdb
        #pdb.set_trace()
        token_ids = torch.cat([token_ids.index_select(index=reorder_inds, dim=0), _next_tokens], dim=-1)
        
            ### ================================ update op_stack ================================ ##
        if with_contraints:
            import pdb
            #pdb.set_trace()
            all_op_stack = []
            for i in range(batch_size*num_beams):
                current_token_ids = token_ids[i][1:].tolist()
                current_op_stack = update_op_stack(current_token_ids, GRAMMER_CLASS, AUX_NUM, vocab_size, target_shift=2)
                all_op_stack.append(current_op_stack)
        #pdb.set_trace()
            ### ================================ update op_stack ================================ ##
        
        
        for batch_idx in range(batch_size):
            dones[batch_idx] = dones[batch_idx] or hypos[batch_idx].is_done(next_scores[batch_idx, 0].item()) or \
                               max_lengths[batch_idx*num_beams]==cur_len+1

        cur_len += 1

        if all(dones):
            break
        
    # select the best hypotheses
    tgt_len = token_ids.new_zeros(batch_size)
    best = []
    best_score = []
    for i, hypotheses in enumerate(hypos):

        best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
        best_score_hyp = max(hypotheses.hyp, key=lambda x: x[0])[0]
        # 把上面替换为非eos的词替换回eos
        if _eos_token_id!=-1:
            best_hyp = torch.cat([best_hyp, best_hyp.new_ones(1)*_eos_token_id])
        tgt_len[i] = len(best_hyp)
        best.append(best_hyp)
        best_score.append(best_score_hyp)
#     pdb.set_trace()
    # generate target batch
    decoded = token_ids.new_zeros(batch_size, tgt_len.max().item()).fill_(pad_token_id)
    best_score = torch.tensor(best_score, dtype=torch.float)
    for i, hypo in enumerate(best):
        decoded[i, :tgt_len[i]] = hypo
    
    return decoded, best_score


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty


