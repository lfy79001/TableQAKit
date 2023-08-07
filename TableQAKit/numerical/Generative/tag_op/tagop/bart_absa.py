import torch
from .modeling_bart import BartEncoder, BartDecoder, BartModel
from transformers import BartTokenizer
from fastNLP import seq_len_to_mask
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch.nn.functional as F
from fastNLP.models import Seq2SeqModel
from torch import nn
import math
from tag_op.tagop.losses import Seq2SeqLoss
from tag_op.tagop.util import FFNLayer

from tag_op.tagop.util import GCN, ResidualGRU
from torch.nn import MultiheadAttention
from tag_op.tagop.modeling_tagop import reduce_mean_index_vector
from tag_op.tagop.tools import allennlp as util
from transformers.modeling_bart import EncoderLayer

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


    
# def invert_mask(attention_mask):
#     """Turns 1->0, 0->1, False->True, True-> False"""
#     assert attention_mask.dim() == 2
#     return attention_mask.eq(0)

class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder, add_structure, config):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder
        import pdb
        #pdb.set_trace()
        ## add cell_level attention
        hidden_size = self.bart_encoder.embed_tokens.weight.size(1)
        self.add_structure=add_structure>0
        self.config = config
#         if self.add_structure>0:
# #             dropout_prob = 0.1
#             self.multihead_attn = nn.ModuleList([MultiheadAttention(hidden_size, 16) for i in range(self.add_structure)])
# # #             self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for i in range(2)])
# #             self.self_output_layer = nn.ModuleList([RobertaSelfOutput(hidden_size, 1e-5, 0.1) for i in range(self.add_structure)])
# #             self._gcn_enc = ResidualGRU(hidden_size, dropout_prob, 2)
# #             self._proj_ln = nn.LayerNorm(hidden_size)
# #             self._proj_sequence = nn.Linear(hidden_size, 1, bias=False)
# #             self.cell_level_layer = EncoderLayer(config)
        
    

    
    
    
    def forward(self, src_tokens, src_seq_len, question_mask, table_mask, paragraph_mask, table_cell_index, all_cell_index, all_cell_index_mask, row_include_cells, col_include_cells, all_cell_token_index, att_mask_vm_lower, att_mask_vm_upper):
        
        if self.add_structure:
            mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
            output_hidden_states=True
            att_mask_vm_lower = att_mask_vm_lower[:, :src_tokens.size(-1), :src_tokens.size(-1)]
            att_mask_vm_upper = att_mask_vm_upper[:, :src_tokens.size(-1), :src_tokens.size(-1)]
            att_mask_vm_lower = (1.0-att_mask_vm_lower)*(-1e24)
            att_mask_vm_upper = (1.0-att_mask_vm_upper)*(-1e24)
            
            att_mask_vm_lower = att_mask_vm_lower.cuda()
            att_mask_vm_upper = att_mask_vm_upper.cuda()

#             if attention_mask is not None:
#             attention_mask = mask.eq(0)
#             attention_mask = torch.mul(mask.unsqueeze(-1), mask.unsqueeze(1))
#             attention_mask = (1.0-attention_mask.float())*(-1e24)
            
            inputs_embeds = self.bart_encoder.embed_tokens(src_tokens) * self.bart_encoder.embed_scale
            embed_pos = self.bart_encoder.embed_positions(src_tokens)
            x = inputs_embeds + embed_pos
            x = self.bart_encoder.layernorm_embedding(x)
            x = F.dropout(x, p=self.bart_encoder.dropout, training=self.bart_encoder.training)

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

            encoder_states = [] if output_hidden_states else None
            all_attentions = None
            layers_num = len(self.bart_encoder.layers)
            if layers_num<12:
                low = 3
            else:
                low = 4
            
            for layer_id, encoder_layer in enumerate(self.bart_encoder.layers):
                if output_hidden_states:
                    encoder_states.append(x)
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                import random
                dropout_probability = random.uniform(0, 1)
                if self.bart_encoder.training and (dropout_probability < self.bart_encoder.layerdrop):  # skip the layer
                    attn = None
                else:
                    if layer_id<int(low):
                        x, attn = encoder_layer(x, attn_mask=att_mask_vm_lower.unsqueeze(1).expand(-1, self.config.encoder_attention_heads,-1,-1))
                    else:
                        x, attn = encoder_layer(x, attn_mask=att_mask_vm_upper.unsqueeze(1).expand(-1, self.config.encoder_attention_heads,-1,-1))
                        
#             pdb.set_trace()
            if self.bart_encoder.layer_norm:
                x = self.bart_encoder.layer_norm(x)
            if output_hidden_states:
                encoder_states.append(x)
                # T x B x C -> B x T x C
                encoder_states = tuple(hidden_state.transpose(0, 1) for hidden_state in encoder_states)

            # T x B x C -> B x T x C
            x = x.transpose(0, 1)

            return x, mask, encoder_states
            
        else:
            mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
            dict = self.bart_encoder(input_ids=src_tokens, key_padding_mask=mask, return_dict=True,
                                     output_hidden_states=True)
            encoder_outputs = dict.last_hidden_state ##bsz, l, h
            hidden_states = dict.hidden_states ## tuple
        
        

            return encoder_outputs, mask, hidden_states


class FBartDecoder(Seq2SeqDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, use_encoder_mlp=True):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        self.label_start_id = label_ids[0]
        self.label_end_id = label_ids[-1]+1
        # 0th position is <s>, 1st position is </s>
        mapping = torch.LongTensor([0, 2]+sorted(label_ids, reverse=False))
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping) 
        hidden_size = decoder.embed_tokens.weight.size(1)
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, hidden_size))

    def forward(self, tokens, state):
        # bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # eos is 1
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # mapping to the BART token index
        mapping_token_mask = tokens.lt(self.src_start_index)  #
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        # first get the

        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[2:3])  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id])  # bsz x max_len x num_class

        # bsz x max_word_len x hidden_size
        src_outputs = state.encoder_output

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)

        mask = mask.unsqueeze(1).__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        
        
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits

    def decode(self, tokens, state):
        return self(tokens, state)[:, -1]


class CaGFBartDecoder(FBartDecoder):
    # Copy and generate,
    def __init__(self, decoder, pad_token_id, label_ids, use_encoder_mlp=False):
        super().__init__(decoder, pad_token_id, label_ids, use_encoder_mlp=use_encoder_mlp)
        hidden_size = decoder.embed_tokens.weight.size(1)
        #self.tag_scorer =  FFNLayer(3*hidden_size, hidden_size, 1, 0.1)
        #self.word_scorer = FFNLayer(2*hidden_size, hidden_size, 1, 0.1)

    def forward(self, tokens, state):
        
        tokens_copy = tokens
        
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask
        first = state.first
        
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        mapping_token_mask = tokens.lt(self.src_start_index)
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)
        
        
        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)  # bsz x max_len
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)
        
        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)  
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[2:3])  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id])  # bsz x max_len x num_class

        # bsz x max_bpe_len x hidden_size
        src_outputs = state.encoder_output
        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  
            # bsz x max_word_len x hidden_size
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)
            # src_outputs = self.decoder.embed_tokens(src_tokens)
        import pdb
#         pdb.set_trace()
        
        mask = mask.unsqueeze(1)
        input_embed = self.decoder.embed_tokens(src_tokens)  # bsz x max_word_len x hidden_size
        import pdb
        
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        gen_scores = torch.einsum('blh,bnh->bln', hidden_state, input_embed)  # bsz x max_len x max_word_len

#        hidden_vec = hidden_state.unsqueeze(2).expand(-1,-1, src_outputs.size(1), -1)
#        src_outputs_vec = src_outputs.unsqueeze(1).expand(-1, hidden_state.size(1), -1, -1)
#        input_embed_vec = input_embed.unsqueeze(1).expand(-1, hidden_state.size(1), -1, -1)
#        word_scores = self.word_scorer(torch.cat([hidden_vec, src_outputs_vec], dim=-1))
#        word_scores = word_scores.squeeze(-1)
#        gen_scores = self.word_scorer(torch.cat([hidden_vec, input_embed_vec], dim=-1))
#        gen_scores = gen_scores.squeeze(-1)
        


        word_scores = (gen_scores + word_scores)/2
    
        word_scores = word_scores.masked_fill(mask, -1e32)
        
        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        
        return logits


class BartSeq2SeqModel(Seq2SeqModel):
    @classmethod
    def build_model(cls, bart_model, tokenizer, label_ids, decoder_type=None, copy_gate=False,
                    use_encoder_mlp=False, use_recur_pos=False, tag_first=False, add_structure=0):
        model = BartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)
        encoder = model.encoder
        decoder = model.decoder
        
        if use_recur_pos:
            decoder.set_position_embedding(label_ids[0], tag_first)

        _tokenizer = BartTokenizer.from_pretrained(bart_model)
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index)>1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index>=num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed
        import pdb
#         pdb.set_trace()
        encoder = FBartEncoder(encoder,add_structure, model.config)
        
        label_ids = sorted(label_ids)
        if decoder_type is None:
            assert copy_gate is False
            decoder = FBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids)
        elif decoder_type =='avg_score':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                              use_encoder_mlp=use_encoder_mlp)
        else:
            raise RuntimeError("Unsupported feature.")

        return cls(encoder=encoder, decoder=decoder)

    def prepare_state(self, src_tokens, src_seq_len=None, first=None, tgt_seq_len=None, question_mask=None, table_mask=None,paragraph_mask=None, table_cell_index=None, all_cell_index=None, all_cell_index_mask=None, row_include_cells=None, col_include_cells=None,all_cell_token_index=None, att_mask_vm_lower=None, att_mask_vm_upper=None):
        assert table_cell_index.size(1)==512
        import pdb
#         pdb.set_trace()
        encoder_outputs, encoder_mask, hidden_states = self.encoder(src_tokens, src_seq_len, question_mask, table_mask, paragraph_mask, table_cell_index, all_cell_index, all_cell_index_mask, row_include_cells, col_include_cells, all_cell_token_index, att_mask_vm_lower, att_mask_vm_upper)
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)
        # setattr(state, 'tgt_seq_len', tgt_seq_len)
        return state

    def forward(self, src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first, question_mask, table_mask, paragraph_mask, table_cell_index, all_cell_index, all_cell_index_mask, row_include_cells, col_include_cells, all_cell_token_index, att_mask_vm_lower, att_mask_vm_upper):
       
        import pdb
#         pdb.set_trace()
        assert table_cell_index.size(1)==512
        state = self.prepare_state(src_tokens, src_seq_len, first, tgt_seq_len, question_mask, table_mask, paragraph_mask, table_cell_index, all_cell_index, all_cell_index_mask, row_include_cells, col_include_cells, all_cell_token_index, att_mask_vm_lower, att_mask_vm_upper)
        decoder_output = self.decoder(tgt_tokens, state)
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")



class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first, src_embed_outputs):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs

    def reorder_state(self, indices: torch.LongTensor):
        import pdb
#         pdb.set_trace()
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new
