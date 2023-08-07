import os
import pickle
import torch
import random

class TaTQABatchGen(object):
    def __init__(self, args, data_mode, encoder='roberta'):
        dpath =  f"tagop_{encoder}_cached_{data_mode}.pkl"
        self.is_train = data_mode == "train"
        self.args = args
        with open(os.path.join(args.data_dir, dpath), 'rb') as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)

        all_data = []
        for item in data:
            input_ids = torch.from_numpy(item["input_ids"])
            ## add question_mask
            question_mask = torch.from_numpy(item["question_mask"])
            
            attention_mask = torch.from_numpy(item["attention_mask"])
            token_type_ids = torch.from_numpy(item["token_type_ids"])
            paragraph_mask = torch.from_numpy(item["paragraph_mask"])
            table_mask = torch.from_numpy(item["table_mask"])
            paragraph_numbers = item["paragraph_number_value"]
            table_cell_numbers = item["table_cell_number_value"]
            paragraph_index = torch.from_numpy(item["paragraph_index"])
            table_cell_index = torch.from_numpy(item["table_cell_index"])
            tag_labels = torch.from_numpy(item["tag_labels"])
            operator_labels = torch.tensor(item["operator_label"])
            scale_labels = torch.tensor(item["scale_label"])
            number_order_labels = torch.tensor(item["number_order_label"])
            gold_answers = item["answer_dict"]
            paragraph_tokens = item["paragraph_tokens"]
            table_cell_tokens = item["table_cell_tokens"]
            question_id = item["question_id"]
            
            ## add table structure
            import pdb
#             pdb.set_trace()
            table_row_header_index = [item[0] for item in item["table_row_header_index"]]
            table_col_header_index = [item[0] for item in item["table_col_header_index"]]
            table_data_cell_index = [item[0] for item in item["table_data_cell_index"]]
            
            same_row_data_cell_rel = item["same_row_data_cell_rel"]
            same_col_data_cell_rel = item["same_col_data_cell_rel"]
            data_cell_row_rel = item["data_cell_row_rel"]
            data_cell_col_rel = item["data_cell_col_rel"]
            
            ## add row_tag/ col_tag
            row_tags = torch.from_numpy(item["row_tags"])
            col_tags = torch.from_numpy(item["col_tags"])
            row_include_cells = item["row_include_cells"]
            col_include_cells = item["col_include_cells"]
            all_cell_index= [item[0] for item in item["all_cell_index"]]
            
            
            all_data.append((input_ids, question_mask, attention_mask, token_type_ids, paragraph_mask, table_mask, paragraph_index,table_cell_index, tag_labels, operator_labels, scale_labels, number_order_labels, gold_answers,
                paragraph_tokens, table_cell_tokens, paragraph_numbers, table_cell_numbers, question_id,
                table_row_header_index, table_col_header_index, table_data_cell_index, same_row_data_cell_rel,same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel, row_tags, col_tags, row_include_cells, col_include_cells, all_cell_index))
            
        print("Load data size {}.".format(len(all_data)))
        self.data = TaTQABatchGen.make_batches(all_data, args.batch_size if self.is_train else args.eval_batch_size, self.is_train)
        self.offset = 0

    @staticmethod
    def make_batches(data, batch_size=32, is_train=True):
        if is_train:
            random.shuffle(data)
        if is_train:
            return [
                data[i: i + batch_size] if i + batch_size < len(data) else data[i:] + data[
                                                                                      :i + batch_size - len(data)]
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
            input_ids_batch, question_mask_batch, attention_mask_batch, token_type_ids_batch, paragraph_mask_batch, table_mask_batch, \
            paragraph_index_batch, table_cell_index_batch, tag_labels_batch, operator_labels_batch, scale_labels_batch, \
            number_order_labels_batch, gold_answers_batch, paragraph_tokens_batch,  \
            table_cell_tokens_batch, paragraph_numbers_batch, table_cell_numbers_batch, question_ids_batch, \
            table_row_header_index_batch, table_col_header_index_batch, table_data_cell_index_batch, same_row_data_cell_rel_batch,same_col_data_cell_rel_batch,data_cell_row_rel_batch,data_cell_col_rel_batch, row_tags_batch, col_tags_batch, row_include_cells_batch, col_include_cells_batch, all_cell_index_batch = zip(*batch)
            
            bsz = len(batch)
            input_ids = torch.LongTensor(bsz, 512)
            ## add question_mask
            question_mask = torch.LongTensor(bsz, 512)
            
            attention_mask = torch.LongTensor(bsz, 512)
            token_type_ids = torch.LongTensor(bsz, 512).fill_(0)
            paragraph_mask = torch.LongTensor(bsz, 512)
            table_mask = torch.LongTensor(bsz, 512)
            paragraph_index = torch.LongTensor(bsz, 512)
            table_cell_index = torch.LongTensor(bsz, 512)
            tag_labels = torch.LongTensor(bsz, 512)
            operator_labels = torch.LongTensor(bsz)
            scale_labels = torch.LongTensor(bsz)
            number_order_labels = torch.LongTensor(bsz)
            paragraph_tokens = []
            table_cell_tokens = []
            gold_answers = []
            question_ids = []
            paragraph_numbers = []
            table_cell_numbers = []

            
            ## add table structure
            max_row_header_num_batch = max([len(table_row_header_index) for table_row_header_index in table_row_header_index_batch])
            max_col_header_num_batch = max([len(table_col_header_index) for table_col_header_index in table_col_header_index_batch])
            max_data_cell_num_batch = max([len(table_data_cell_index) for table_data_cell_index in table_data_cell_index_batch])
            
            
            
            
            table_row_header_index = torch.LongTensor(bsz, max_row_header_num_batch).fill_(0)
            table_col_header_index = torch.LongTensor(bsz, max_col_header_num_batch).fill_(0)
            table_data_cell_index = torch.LongTensor(bsz, max_data_cell_num_batch).fill_(0)
            
            table_row_header_index_mask = torch.LongTensor(bsz, max_row_header_num_batch).fill_(0)
            table_col_header_index_mask = torch.LongTensor(bsz, max_col_header_num_batch).fill_(0)
            table_data_cell_index_mask = torch.LongTensor(bsz, max_data_cell_num_batch).fill_(0)
            
            same_row_data_cell_relation = torch.LongTensor(bsz, max_data_cell_num_batch, max_data_cell_num_batch).fill_(0)
            same_col_data_cell_relation = torch.LongTensor(bsz, max_data_cell_num_batch, max_data_cell_num_batch).fill_(0)
            data_cell_row_rel = torch.LongTensor(bsz, max_row_header_num_batch, max_data_cell_num_batch).fill_(0)
            data_cell_col_rel = torch.LongTensor(bsz, max_col_header_num_batch, max_data_cell_num_batch).fill_(0)

#             import pdb
#             pdb.set_trace()
            
            ## add row col tag
            max_row_num = max([len(row_tags) for row_tags in row_tags_batch])
            max_col_num = max([len(col_tags) for col_tags in col_tags_batch])
            max_cell_num = max(len(table_cell_numbers) for table_cell_numbers in table_cell_numbers_batch)
            
            
            row_tags = torch.LongTensor(bsz, max_row_num).fill_(0)
            col_tags = torch.LongTensor(bsz, max_col_num).fill_(0)
            row_tag_mask = torch.LongTensor(bsz, max_row_num).fill_(0)
            col_tag_mask = torch.LongTensor(bsz, max_col_num).fill_(0)
            
            row_include_cells = torch.LongTensor(bsz, max_row_num, 512).fill_(0)
            col_include_cells = torch.LongTensor(bsz, max_col_num, 512).fill_(0)
            all_cell_index = torch.LongTensor(bsz, 512).fill_(0)
            all_cell_index_mask = torch.LongTensor(bsz, 512).fill_(0)

            
            for i in range(bsz):
                input_ids[i] = input_ids_batch[i]
                question_mask[i] = question_mask_batch[i]
            
                attention_mask[i] = attention_mask_batch[i]
                token_type_ids[i] = token_type_ids_batch[i]
                paragraph_mask[i] = paragraph_mask_batch[i]
                table_mask[i] = table_mask_batch[i]
                paragraph_index[i] = paragraph_index_batch[i]
                table_cell_index[i] = table_cell_index_batch[i]
                tag_labels[i] = tag_labels_batch[i]
                operator_labels[i] = operator_labels_batch[i]
                scale_labels[i] = scale_labels_batch[i]
                number_order_labels[i] = number_order_labels_batch[i]
                paragraph_tokens.append(paragraph_tokens_batch[i])
                table_cell_tokens.append(table_cell_tokens_batch[i])
                paragraph_numbers.append(paragraph_numbers_batch[i])
                table_cell_numbers.append(table_cell_numbers_batch[i])
                gold_answers.append(gold_answers_batch[i])
                question_ids.append(question_ids_batch[i])
                
                ## add table structure
                for j in range(len(table_row_header_index_batch[i])):
                    table_row_header_index[i][j] = table_row_header_index_batch[i][j]
                    table_row_header_index_mask[i][j]=1
                    
                for j in range(len(table_col_header_index_batch[i])):
                    table_col_header_index[i][j] = table_col_header_index_batch[i][j]
                    table_col_header_index_mask[i][j]=1
                
                for j in range(len(table_data_cell_index_batch[i])):
                    table_data_cell_index[i][j] = table_data_cell_index_batch[i][j]
                    table_data_cell_index_mask[i][j]=1
                import pdb
#                 pdb.set_trace()
                
                for m in range(len(same_row_data_cell_rel_batch[i])):
                    row_data_cell_tmp = same_row_data_cell_rel_batch[i][m]
                    row_data_cell_tmp_index = [table_data_cell_index_batch[i].index(cell_index) for cell_index in row_data_cell_tmp]
                    for p in row_data_cell_tmp_index:
                        for q in row_data_cell_tmp_index:
                            same_row_data_cell_relation[i][p][q]=1
                
                for m in range(len(same_col_data_cell_rel_batch[i])):
                    col_data_cell_tmp = same_col_data_cell_rel_batch[i][m]
                    col_data_cell_tmp_index = [table_data_cell_index_batch[i].index(cell_index) for cell_index in col_data_cell_tmp]
                    for p in col_data_cell_tmp_index:
                        for q in col_data_cell_tmp_index:
                            same_col_data_cell_relation[i][p][q]=1
                
                
                
                for key, value in data_cell_row_rel_batch[i].items():
#                     import pdb
#                     pdb.set_trace()
                    key_index = table_row_header_index_batch[i].index(key)
                    value_index = [table_data_cell_index_batch[i].index(item) for item in value]
                    for k in value_index:
                        data_cell_row_rel[i][key_index][k] = 1
                
                for key, value in data_cell_col_rel_batch[i].items():
                    key_index = table_col_header_index_batch[i].index(key)
                    value_index = [table_data_cell_index_batch[i].index(item) for item in value]
                    for k in value_index:
                        data_cell_col_rel[i][key_index][k] = 1 
                
#             import pdb
#             pdb.set_trace()
                
                ## add row tag & col tag
                for id, item in enumerate(row_tags_batch[i]):
                    row_tags[i][id]=item
                    row_tag_mask[i][id]=1
                
                for id, item in enumerate(col_tags_batch[i]):
                    col_tags[i][id]=item
                    col_tag_mask[i][id]=1
                    
                for key, value in row_include_cells_batch[i].items():
                    for cell_id in value:
                        row_include_cells[i][key][cell_id-1]=1
                        
                for key, value in col_include_cells_batch[i].items():
                    for cell_id in value:
                        col_include_cells[i][key][cell_id-1]=1
                        
                for j in range(len(all_cell_index_batch[i])):
                    all_cell_index[i][j] = all_cell_index_batch[i][j]
                    all_cell_index_mask[i][j]=1
                    

            out_batch = {"input_ids": input_ids, "question_mask":question_mask, "attention_mask": attention_mask, "token_type_ids":token_type_ids,
                "paragraph_mask": paragraph_mask, "paragraph_index": paragraph_index, "tag_labels": tag_labels,
                "operator_labels": operator_labels, "scale_labels": scale_labels, "number_order_labels": number_order_labels,
                "paragraph_tokens": paragraph_tokens, "table_cell_tokens": table_cell_tokens, "paragraph_numbers": paragraph_numbers,
                "table_cell_numbers": table_cell_numbers, "gold_answers": gold_answers, "question_ids": question_ids,
                "table_mask": table_mask, "table_cell_index":table_cell_index,
                         
                ## add table structure
                "table_row_header_index":table_row_header_index,
                "table_col_header_index":table_col_header_index,
                "table_data_cell_index": table_data_cell_index,
                "table_row_header_index_mask":table_row_header_index_mask,
                "table_col_header_index_mask":table_col_header_index_mask,
                "table_data_cell_index_mask":table_data_cell_index_mask,
                "same_row_data_cell_relation":same_row_data_cell_relation,
                "same_col_data_cell_relation":same_col_data_cell_relation,
                "data_cell_row_rel":data_cell_row_rel,
                "data_cell_col_rel":data_cell_col_rel,
                "row_tags":row_tags,
                "row_tag_mask":row_tag_mask,
                "col_tags":col_tags,
                "col_tag_mask":col_tag_mask,
                "row_include_cells":row_include_cells,
                "col_include_cells":col_include_cells,
                "all_cell_index":all_cell_index,
                "all_cell_index_mask":all_cell_index_mask
            }
            
            import pdb
#             pdb.set_trace()
            
            if self.args.cuda:
                for k in out_batch.keys():
                    if isinstance(out_batch[k], torch.Tensor):
                        out_batch[k] = out_batch[k].cuda()

            yield  out_batch

class TaTQATestBatchGen(object):
    def __init__(self, args, data_mode, encoder='roberta'):
        dpath =  f"tagop_{encoder}_cached_{data_mode}.pkl"
        self.is_train = data_mode == "train"
        self.args = args
        print(os.path.join(args.test_data_dir, dpath))
        with open(os.path.join(args.test_data_dir, dpath), 'rb') as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)

        all_data = []
        for item in data:
            input_ids = torch.from_numpy(item["input_ids"])
            ## add question_mask
            question_mask = torch.from_numpy(item["question_mask"])
            
            attention_mask = torch.from_numpy(item["attention_mask"])
            token_type_ids = torch.from_numpy(item["token_type_ids"])
            paragraph_mask = torch.from_numpy(item["paragraph_mask"])
            table_mask = torch.from_numpy(item["table_mask"])
            paragraph_numbers = item["paragraph_number_value"]
            table_cell_numbers = item["table_cell_number_value"]
            paragraph_index = torch.from_numpy(item["paragraph_index"])
            table_cell_index = torch.from_numpy(item["table_cell_index"])
            tag_labels = torch.from_numpy(item["tag_labels"])
            gold_answers = item["answer_dict"]
            paragraph_tokens = item["paragraph_tokens"]
            table_cell_tokens = item["table_cell_tokens"]
            question_id = item["question_id"]
            
            ## add table structure
            import pdb
#             pdb.set_trace()
            table_row_header_index = [item[0] for item in item["table_row_header_index"]]
            table_col_header_index = [item[0] for item in item["table_col_header_index"]]
            table_data_cell_index = [item[0] for item in item["table_data_cell_index"]]
            
            same_row_data_cell_rel = item["same_row_data_cell_rel"]
            same_col_data_cell_rel = item["same_col_data_cell_rel"]
            data_cell_row_rel = item["data_cell_row_rel"]
            data_cell_col_rel = item["data_cell_col_rel"]
            
            ## add row_tag/ col_tag
            row_tags = torch.from_numpy(item["row_tags"])
            col_tags = torch.from_numpy(item["col_tags"])
            row_include_cells = item["row_include_cells"]
            col_include_cells = item["col_include_cells"]
            all_cell_index= [item[0] for item in item["all_cell_index"]]
            
            
            
            import pdb
#             pdb.set_trace()
            table_cell_number_value_with_time = item["table_cell_number_value_with_time"]
            
            all_data.append((input_ids, question_mask, attention_mask, token_type_ids, paragraph_mask, table_mask, paragraph_index,
                             table_cell_index, tag_labels, gold_answers, paragraph_tokens, table_cell_tokens,
                             paragraph_numbers, table_cell_numbers, question_id, table_row_header_index, table_col_header_index, table_data_cell_index, same_row_data_cell_rel,same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel, table_cell_number_value_with_time, row_tags, col_tags, row_include_cells, col_include_cells, all_cell_index))
            
            
            
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
                data[i: i + batch_size] if i + batch_size < len(data) else data[i:] + data[
                                                                                      :i + batch_size - len(data)]
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
            input_ids_batch, question_mask_batch, attention_mask_batch, token_type_ids_batch, paragraph_mask_batch, table_mask_batch, \
            paragraph_index_batch, table_cell_index_batch, tag_labels_batch, gold_answers_batch, paragraph_tokens_batch, \
            table_cell_tokens_batch, paragraph_numbers_batch, table_cell_numbers_batch, question_ids_batch,table_row_header_index_batch, table_col_header_index_batch, table_data_cell_index_batch, same_row_data_cell_rel_batch,same_col_data_cell_rel_batch,data_cell_row_rel_batch,data_cell_col_rel_batch, table_cell_number_value_with_time_batch, row_tags_batch, col_tags_batch, row_include_cells_batch, col_include_cells_batch, all_cell_index_batch = zip(*batch)
            import pdb
#             pdb.set_trace()
            
            bsz = len(batch)
            input_ids = torch.LongTensor(bsz, 512)
            ## add question mask
            question_mask = torch.LongTensor(bsz, 512)
            
            attention_mask = torch.LongTensor(bsz, 512)
            token_type_ids = torch.LongTensor(bsz, 512).fill_(0)
            paragraph_mask = torch.LongTensor(bsz, 512)
            table_mask = torch.LongTensor(bsz, 512)
            paragraph_index = torch.LongTensor(bsz, 512)
            table_cell_index = torch.LongTensor(bsz, 512)
            tag_labels = torch.LongTensor(bsz, 512)
            paragraph_tokens = []
            table_cell_tokens = []
            gold_answers = []
            question_ids = []
            paragraph_numbers = []
            table_cell_numbers = []
            
            table_cell_number_value_with_time = []
            
            ## add table structure
            max_row_header_num_batch = max([len(table_row_header_index) for table_row_header_index in table_row_header_index_batch])
            max_col_header_num_batch = max([len(table_col_header_index) for table_col_header_index in table_col_header_index_batch])
            max_data_cell_num_batch = max([len(table_data_cell_index) for table_data_cell_index in table_data_cell_index_batch])
            
            
            
            table_row_header_index = torch.LongTensor(bsz, max_row_header_num_batch).fill_(0)
            table_col_header_index = torch.LongTensor(bsz, max_col_header_num_batch).fill_(0)
            table_data_cell_index = torch.LongTensor(bsz, max_data_cell_num_batch).fill_(0)
            
            table_row_header_index_mask = torch.LongTensor(bsz, max_row_header_num_batch).fill_(0)
            table_col_header_index_mask = torch.LongTensor(bsz, max_col_header_num_batch).fill_(0)
            table_data_cell_index_mask = torch.LongTensor(bsz, max_data_cell_num_batch).fill_(0)
            
            same_row_data_cell_relation = torch.LongTensor(bsz, max_data_cell_num_batch, max_data_cell_num_batch).fill_(0)
            same_col_data_cell_relation = torch.LongTensor(bsz, max_data_cell_num_batch, max_data_cell_num_batch).fill_(0)
            data_cell_row_rel = torch.LongTensor(bsz, max_row_header_num_batch, max_data_cell_num_batch).fill_(0)
            data_cell_col_rel = torch.LongTensor(bsz, max_col_header_num_batch, max_data_cell_num_batch).fill_(0)
            
            
            
            ## add row col tag
            max_row_num = max([len(row_tags) for row_tags in row_tags_batch])
            max_col_num = max([len(col_tags) for col_tags in col_tags_batch])
            max_cell_num = max(len(table_cell_numbers) for table_cell_numbers in table_cell_numbers_batch)
            
            
            row_tags = torch.LongTensor(bsz, max_row_num).fill_(0)
            col_tags = torch.LongTensor(bsz, max_col_num).fill_(0)
            row_tag_mask = torch.LongTensor(bsz, max_row_num).fill_(0)
            col_tag_mask = torch.LongTensor(bsz, max_col_num).fill_(0)
            
            row_include_cells = torch.LongTensor(bsz, max_row_num, 512).fill_(0)
            col_include_cells = torch.LongTensor(bsz, max_col_num, 512).fill_(0)
            all_cell_index = torch.LongTensor(bsz, 512).fill_(0)
            all_cell_index_mask = torch.LongTensor(bsz, 512).fill_(0)
            

            for i in range(bsz):
                input_ids[i] = input_ids_batch[i]
                ## add question mask
                question_mask[i] = question_mask_batch[i]
                
                attention_mask[i] = attention_mask_batch[i]
                token_type_ids[i] = token_type_ids_batch[i]
                paragraph_mask[i] = paragraph_mask_batch[i]
                table_mask[i] = table_mask_batch[i]
                paragraph_index[i] = paragraph_index_batch[i]
                table_cell_index[i] = table_cell_index_batch[i]
                tag_labels[i] = tag_labels_batch[i]
                paragraph_tokens.append(paragraph_tokens_batch[i])
                table_cell_tokens.append(table_cell_tokens_batch[i])
                paragraph_numbers.append(paragraph_numbers_batch[i])
                table_cell_numbers.append(table_cell_numbers_batch[i])
                gold_answers.append(gold_answers_batch[i])
                question_ids.append(question_ids_batch[i])
                
                
                ## table_cell_number_value_with_time_batch
                import pdb
#                 pdb.set_trace()
                table_cell_number_value_with_time.append(table_cell_number_value_with_time_batch[i])
                
                ## add table structure
                for j in range(len(table_row_header_index_batch[i])):
                    table_row_header_index[i][j] = table_row_header_index_batch[i][j]
                    table_row_header_index_mask[i][j]=1
                    
                for j in range(len(table_col_header_index_batch[i])):
                    table_col_header_index[i][j] = table_col_header_index_batch[i][j]
                    table_col_header_index_mask[i][j]=1
                
                for j in range(len(table_data_cell_index_batch[i])):
                    table_data_cell_index[i][j] = table_data_cell_index_batch[i][j]
                    table_data_cell_index_mask[i][j]=1
                import pdb
#                 pdb.set_trace()
                
                for m in range(len(same_row_data_cell_rel_batch[i])):
                    row_data_cell_tmp = same_row_data_cell_rel_batch[i][m]
                    row_data_cell_tmp_index = [table_data_cell_index_batch[i].index(cell_index) for cell_index in row_data_cell_tmp]
                    for p in row_data_cell_tmp_index:
                        for q in row_data_cell_tmp_index:
                            same_row_data_cell_relation[i][p][q]=1
                
                for m in range(len(same_col_data_cell_rel_batch[i])):
                    col_data_cell_tmp = same_col_data_cell_rel_batch[i][m]
                    col_data_cell_tmp_index = [table_data_cell_index_batch[i].index(cell_index) for cell_index in col_data_cell_tmp]
                    for p in col_data_cell_tmp_index:
                        for q in col_data_cell_tmp_index:
                            same_col_data_cell_relation[i][p][q]=1
                
                
                
                for key, value in data_cell_row_rel_batch[i].items():
#                     import pdb
#                     pdb.set_trace()
                    key_index = table_row_header_index_batch[i].index(key)
                    value_index = [table_data_cell_index_batch[i].index(item) for item in value]
                    for k in value_index:
                        data_cell_row_rel[i][key_index][k] = 1
                
                for key, value in data_cell_col_rel_batch[i].items():
                    key_index = table_col_header_index_batch[i].index(key)
                    value_index = [table_data_cell_index_batch[i].index(item) for item in value]
                    for k in value_index:
                        data_cell_col_rel[i][key_index][k] = 1 
                
                
                ## add row tag & col tag
                for id, item in enumerate(row_tags_batch[i]):
                    row_tags[i][id]=item
                    row_tag_mask[i][id]=1
                
                for id, item in enumerate(col_tags_batch[i]):
                    col_tags[i][id]=item
                    col_tag_mask[i][id]=1
                    
                for key, value in row_include_cells_batch[i].items():
                    for cell_id in value:
                        row_include_cells[i][key][cell_id-1]=1
                        
                for key, value in col_include_cells_batch[i].items():
                    for cell_id in value:
                        col_include_cells[i][key][cell_id-1]=1
                        
                for j in range(len(all_cell_index_batch[i])):
                    all_cell_index[i][j] = all_cell_index_batch[i][j]
                    all_cell_index_mask[i][j]=1
                
#             import pdb
#             pdb.set_trace()
            out_batch = {"input_ids": input_ids, "question_mask":question_mask,  "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                         "paragraph_mask": paragraph_mask, "paragraph_index": paragraph_index, "tag_labels": tag_labels,
                         "paragraph_tokens": paragraph_tokens, "table_cell_tokens": table_cell_tokens,
                         "paragraph_numbers": paragraph_numbers,
                         "table_cell_numbers": table_cell_numbers, "gold_answers": gold_answers, "question_ids": question_ids,
                         "table_mask": table_mask, "table_cell_index": table_cell_index,
                         # "paragraph_mapping_content": paragraph_mapping_content,
                         # "table_mapping_content": table_mapping_content,
                         ## add table structure
                        "table_row_header_index":table_row_header_index,
                        "table_col_header_index":table_col_header_index,
                        "table_data_cell_index": table_data_cell_index,
                        "table_row_header_index_mask":table_row_header_index_mask,
                        "table_col_header_index_mask":table_col_header_index_mask,
                        "table_data_cell_index_mask":table_data_cell_index_mask,
                        "same_row_data_cell_relation":same_row_data_cell_relation,
                        "same_col_data_cell_relation":same_col_data_cell_relation,
                        "data_cell_row_rel":data_cell_row_rel,
                        "data_cell_col_rel":data_cell_col_rel,
                        "table_cell_number_value_with_time":table_cell_number_value_with_time,
                        "row_tags":row_tags,
                        "row_tag_mask":row_tag_mask,
                        "col_tags":col_tags,
                        "col_tag_mask":col_tag_mask,
                        "row_include_cells":row_include_cells,
                        "col_include_cells":col_include_cells,
                        "all_cell_index":all_cell_index,
                        "all_cell_index_mask":all_cell_index_mask
                         }

            if self.args.cuda:
                for k in out_batch.keys():
                    if isinstance(out_batch[k], torch.Tensor):
                        out_batch[k] = out_batch[k].cuda()

            yield  out_batch