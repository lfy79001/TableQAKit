import os
import pickle
import torch
import random
import numpy as np
import dgl

class TaTQATestBatchGen(object):
    def __init__(self, args, data_mode, encoder='roberta'):
        dpath =  f"reghnt_{encoder}_cached_{data_mode}.pkl"
        self.is_train = data_mode == "train"
        self.args = args
        with open(os.path.join(args.data_dir, dpath), 'rb') as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)

        all_data = []
        for item in data:

            input_ids = torch.from_numpy(item["input_ids"])
            attention_mask = torch.from_numpy(item["attention_mask"])
            token_type_ids = torch.from_numpy(item["token_type_ids"])


            question_tokens = item['question_tokens']
            table = item['table']
            table_cell_type_ids = torch.from_numpy(item['table_cell_type_ids'])
            table_is_scale_type = torch.from_numpy(item['table_is_scale_type'])
            table_number = item['table_number']
            table_number_index = torch.from_numpy(item['table_number_index'])
            table_cell_index = torch.from_numpy(item['table_cell_index'])
            table_tokens = item['table_tokens']
            paragraph_tokens = item['paragraph_tokens']
            paragraph_index = torch.from_numpy(item['paragraph_index'])
            paragraph_number = item['paragraph_number']
            paragraph_number_index = torch.from_numpy(item['paragraph_number_index'])


            question_id = item['question_id']
            
            word_mask = torch.tensor(item['word_mask']) 
            number_mask = torch.tensor(item['number_mask']) 
            subword_lens = item['subword_lens']
            item_length = item['item_length']
            
            word_word_mask = item['word_word_mask']
            number_word_mask = item['number_word_mask']
            
            table_word_tokens = item['table_word_tokens']
            
            
            table_word_tokens_number = item['table_word_tokens_number']
            
            table_number_scale = torch.tensor(item['table_number_scale'])
            paragraph_number_scale = torch.tensor(item['paragraph_number_scale'])
            
            
            graph = item['graph']
            
            questions = item['question']

            question_number = item['question_number']
            question_number_index = torch.from_numpy(np.array(item['question_number_index']))
            
            
            all_data.append((input_ids, attention_mask, token_type_ids,
                question_tokens,
                table, table_cell_type_ids, table_is_scale_type, table_number, table_number_index,
                table_cell_index, table_tokens,paragraph_tokens, paragraph_index,
                paragraph_number,paragraph_number_index, question_id,
                word_mask, number_mask, subword_lens, item_length,
                word_word_mask, number_word_mask, table_word_tokens, table_word_tokens_number,
                table_number_scale, paragraph_number_scale, graph, questions,
                question_number, question_number_index))
        print("Load data size {}.".format(len(all_data)))
        self.data = TaTQATestBatchGen.make_batches(all_data, args.batch_size if self.is_train else args.eval_batch_size,
                                              self.is_train)
        self.offset = 0

    @staticmethod
    def make_batches(data, batch_size=32, is_train=True):
        if is_train:
            random.shuffle(data)
        if is_train:
            return [
                data[i: i + batch_size] if i + batch_size < len(data) else data[i:] + data[:i + batch_size - len(data)]
                for i in range(0, len(data), batch_size)]
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
            for i in range(len(self.data)):
                random.shuffle(self.data[i])
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            self.offset += 1
            input_ids_batch, attention_mask_batch,token_type_ids_batch,\
            question_tokens_batch,\
            table_batch, table_cell_type_ids_batch,table_is_scale_type_batch, table_number_batch,\
            table_number_index_batch, table_cell_index_batch, table_tokens_batch,\
            paragraph_tokens_batch,paragraph_index_batch,paragraph_number_batch,\
            paragraph_number_index_batch, question_id_batch,\
            word_mask_batch, number_mask_batch, subword_lens_batch, item_length_batch,\
            word_word_mask_batch, number_word_mask_batch, table_word_tokens_batch,\
            table_word_tokens_number_batch, table_number_scale_batch, paragraph_number_scale_batch,\
            graph_batch, question_batch,\
                question_number_batch, question_number_index_batch = zip(*batch)
            
            bs = len(batch)
            
            input_ids = torch.LongTensor(bs, 512)
            attention_mask = torch.LongTensor(bs, 512)
            token_type_ids = torch.LongTensor(bs, 512).fill_(0)
        
            
            table = []
            table_cell_type_ids = []
            table_is_scale_type = []
            table_number = []
            table_number_index = []
            paragraph_number = []
            paragraph_number_index = []
            question_number = []
            question_number_index = []
            question_id = []
            table_word_tokens_number = []
            
            table_number_scale = []
            paragraph_number_scale = []
            
            questions = []
            
            word_mask = torch.LongTensor(bs, 512)
            number_mask = torch.LongTensor(bs, 512)
            
            max_batch_len = max(list(map(lambda x: x['q']+x['t']+x['p'], item_length_batch)))
            

            b_mask = torch.zeros([bs, max_batch_len], dtype=bool).cuda()
            b_question_mask = torch.zeros_like(b_mask)
            b_table_mask = torch.zeros_like(b_mask)
            b_paragraph_mask = torch.zeros_like(b_mask)
            b_number_mask = torch.zeros_like(b_mask)
            
            b_tokens = []
            
            for i in range(bs):
                
                question_tokens = question_tokens_batch[i]
                table_tokens = table_tokens_batch[i]
                paragraph_tokens = paragraph_tokens_batch[i]
                
                b_tokens.append(question_tokens + table_tokens + paragraph_tokens)
                              
                
                input_ids[i] = input_ids_batch[i]
                attention_mask[i] = attention_mask_batch[i]
                token_type_ids[i] = token_type_ids_batch[i]

                
                table.append(table_batch[i])
                table_cell_type_ids.append(table_cell_type_ids_batch[i])
                table_is_scale_type.append(table_is_scale_type_batch[i])
                question_number.append(question_number_batch[i])
                question_number_index.append(question_number_index_batch[i])
                table_number.append(table_number_batch[i])
                table_number_index.append(table_number_index_batch[i])
                paragraph_number.append(paragraph_number_batch[i])
                paragraph_number_index.append(paragraph_number_index_batch[i])

                question_id.append(question_id_batch[i])
                word_mask[i] = word_mask_batch[i]
                number_mask[i] = number_mask_batch[i]

                table_number_scale.append(table_number_scale_batch[i])
                paragraph_number_scale.append(paragraph_number_scale_batch[i])
                questions.append(question_batch[i])
                
                
                item_l = item_length_batch[i]
                leng = item_l['q'] + item_l['t'] + item_l['p'] 
                b_mask[i][:leng] = 1
    
                b_question_mask[i][:item_l['q']] = 1
                b_table_mask[i][item_l['q']:item_l['q']+item_l['t']] = 1
                b_paragraph_mask[i][item_l['q']+item_l['t']:item_l['q']+item_l['t']+item_l['p']]=1
                
                question_number_index_i = question_number_index_batch[i]
                table_number_index_i = table_number_index_batch[i]
                paragraph_number_index_i = paragraph_number_index_batch[i]
                for idx in question_number_index_i:
                    b_number_mask[i][idx.item()] = 1
                for idx in table_number_index_i:
                    b_number_mask[i][idx.item()+item_l['q']] = 1
                for idx in paragraph_number_index_i:
                    b_number_mask[i][idx.item()+item_l['q']+item_l['t']] = 1
                
                
                
            

            b_word_word_mask = torch.cat([torch.tensor(word_word_mask) for word_word_mask in word_word_mask_batch]).bool()
            b_number_word_mask = torch.cat([torch.tensor(number_word_mask) for number_word_mask in number_word_mask_batch]).bool()
            b_subword_lens = torch.cat([torch.tensor(subword_lens) for subword_lens in subword_lens_batch])

            word_subword_lens = b_subword_lens.masked_select(b_word_word_mask)
            number_subword_lens = b_subword_lens.masked_select(b_number_word_mask)
            
            max_word_subword_len = max(word_subword_lens)
            max_number_subword_len = max(number_subword_lens)
            
            word_subword_mask = torch.zeros([word_subword_lens.size(0), max_word_subword_len], dtype=bool)
            number_subword_mask = torch.zeros([number_subword_lens.size(0), max_number_subword_len], dtype=bool)
            


            for i in range(len(word_subword_mask)):
                word_subword_mask[i][:word_subword_lens[i]] = 1
            for i in range(len(number_subword_mask)):
                number_subword_mask[i][:number_subword_lens[i]] = 1
            
            
            max_question_len = max([item_l['q'] for item_l in item_length_batch])
            max_table_len = max([item_l['t'] for item_l in item_length_batch])
            max_table_word_len = max([len(subword_lens_batch[i])-item_l['p'] - item_l['q'] for i, item_l in enumerate(item_length_batch)])
            max_paragraph_len = max([item_l['p'] for item_l in item_length_batch])
            

            question_lens = torch.tensor([item_l['q'] for item_l in item_length_batch]).cuda()
            table_lens = torch.tensor([item_l['t'] for item_l in item_length_batch]).cuda()
            table_word_lens = torch.tensor([len(subword_lens_batch[i])-item_l['p'] - item_l['q'] for i, item_l in enumerate(item_length_batch)]).cuda()
            paragraph_lens = torch.tensor([item_l['p'] for item_l in item_length_batch]).cuda()
            
            lens = {"question":question_lens, "table":table_lens, "table_word":table_word_lens, "paragraph":paragraph_lens}
            max_len = {"question":max_question_len,"table":max_table_len,"table_word":max_table_word_len,"paragraph":max_paragraph_len}
            mask = {"b":b_mask, "question":b_question_mask, "table":b_table_mask, "paragraph":b_paragraph_mask,
                    "number": b_number_mask}

            max_pl_len = max([len(subword_lens) for subword_lens in subword_lens_batch])

            b_pl_mask = torch.zeros([bs, max_pl_len]).bool().cuda()
            q_pl_mask = torch.zeros_like(b_pl_mask)
            t_pl_mask = torch.zeros_like(b_pl_mask)
            p_pl_mask = torch.zeros_like(b_pl_mask)
            question_pl_mask = torch.zeros([bs, max_question_len]).bool().cuda()
            table_pl_mask = torch.zeros([bs, max_table_word_len]).bool().cuda()
            paragraph_pl_mask = torch.zeros([bs, max_paragraph_len]).bool().cuda()
            for i in range(bs):
                b_pl_mask[i][:question_lens[i]+table_word_lens[i]+paragraph_lens[i]] = 1
                question_pl_mask[i][:question_lens[i]] = 1
                table_pl_mask[i][:table_word_lens[i]] = 1
                paragraph_pl_mask[i][:paragraph_lens[i]] = 1
                q_pl_mask[i][:question_lens[i]] = 1
                t_pl_mask[i][question_lens[i]:question_lens[i]+table_word_lens[i]] = 1
                p_pl_mask[i][question_lens[i]+table_word_lens[i]:question_lens[i]+table_word_lens[i]+paragraph_lens[i]] = 1     
            
            table_word_tokens_number_len = 0
            max_table_word_tokens_number_len = 0
            word_tokens_number = []
            for table_word_tokens_number in table_word_tokens_number_batch:
                table_word_tokens_number_len += len(table_word_tokens_number)
                max_table_word_tokens_number_len = max(max_table_word_tokens_number_len, max(table_word_tokens_number))
                word_tokens_number += table_word_tokens_number
            
            t_w_t_n_mask = torch.zeros([table_word_tokens_number_len, max_table_word_tokens_number_len]).bool().cuda()
            
            for i in range(len(t_w_t_n_mask)):
                t_w_t_n_mask[i][:word_tokens_number[i]] = 1
                
            b_graph = dgl.batch([g['dgl'] for g in graph_batch]).to("cuda:0")
            b_relation = torch.cat([torch.tensor(g['relations']) for g in graph_batch]).cuda()
            b_src = torch.cat([torch.tensor(g['src']) for g in graph_batch]).cuda()
            b_dst = torch.cat([torch.tensor(g['dst']) for g in graph_batch]).cuda()

            pl_mask = {"b":b_pl_mask,"question":question_pl_mask,"table":table_pl_mask,"paragraph":paragraph_pl_mask,
                       "q":q_pl_mask, "t":t_pl_mask, "p":p_pl_mask}
            
            out_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids,
            "word_subword_lens":word_subword_lens, "number_subword_lens":number_subword_lens,
            "word_subword_mask":word_subword_mask, "number_subword_mask":number_subword_mask,
            "lens": lens, "max_len": max_len, "pl_mask":pl_mask, "mask": mask,
            "table":table, "table_cell_type_ids":table_cell_type_ids,
            "table_is_scale_type":table_is_scale_type,"table_number":table_number,"table_number_index":table_number_index,
            "paragraph_number":paragraph_number,"paragraph_number_index":paragraph_number_index,
            "question_id":question_id, "word_mask":word_mask, "number_mask":number_mask,
            "b_word_word_mask":b_word_word_mask, "b_number_word_mask":b_number_word_mask, "t_w_t_n_mask":t_w_t_n_mask,
            "b_tokens":b_tokens, "b_graph": b_graph,
            "b_relation": b_relation, "b_src": b_src, "b_dst": b_dst, "t_scale":table_number_scale,"p_scale":paragraph_number_scale,
            "questions":questions,"question_number":question_number, "question_number_index":question_number_index
            }

            if self.args.cuda:
                for k in out_batch.keys():
                    if isinstance(out_batch[k], torch.Tensor):
                        out_batch[k] = out_batch[k].cuda()

            yield  out_batch