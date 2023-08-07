from fastNLP import MetricBase
from fastNLP.core.metrics import _compute_f_pre_rec
from collections import Counter
from tag_op.executor import TATExecutor
from tag_op.data.operation_util import GRAMMER_CLASS, AUX_NUM, GRAMMER_ID, OP_ID, SCALE
from tatqa_metric import TaTQAEmAndF1
import pdb

class Seq2SeqSpanMetric(MetricBase):
    def __init__(self, eos_token_id, num_labels):
        super(Seq2SeqSpanMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.word_start_index = num_labels + 2  # +2, shift for sos and eos
        self.em = 0
        self.invalid = 0
        self.total = 0
        self.scale_em = 0
        
        self.executor = TATExecutor( GRAMMER_CLASS, GRAMMER_ID, AUX_NUM)
        self.answer_em_f1 = TaTQAEmAndF1()
 

    def evaluate(self, target_span, pred, scale_label, tgt_tokens, answer, answer_from, answer_type, table_cell_tokens, table_cell_number_value, table_cell_index, paragraph_tokens, paragraph_number_value, paragraph_index, question_id):
        
#         import pdb
#         pdb.set_trace()
        answer = answer.tolist()
        scale_pred = pred[1]
        #scale acc
        for i, (scale, scale_p) in enumerate(zip(scale_label.squeeze(-1).tolist(), scale_pred.tolist())):
            scale_em = int(scale==scale_p)
            self.scale_em +=scale_em
            
        pred = pred[0]
        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # delete </s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1)  # bsz
        target_seq_len = (target_seq_len - 2).tolist()
        pred_spans = []
        for i, (ts, ps) in enumerate(zip(target_span, pred.tolist())):
            em = 0
            ps = ps[:pred_seq_len[i]]
            if pred_seq_len[i] == target_seq_len[i]:
#                 pdb.set_trace()
                em = int(
                    tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item() == target_seq_len[i])
            self.em += em
            
#             lf = self.trans_id2lf(tgt_tokens[i, :target_seq_len[i]].tolist())
            import pdb
#             pdb.set_trace()
            lf = self.trans_id2lf(ps)
            example = {
                        "table_cell_tokens": list(table_cell_tokens[i]),
                        "table_cell_number_value":table_cell_number_value[i],
                        "table_cell_index":[table_cell_index[i].tolist()],
                        "paragraph_tokens":list(paragraph_tokens[i]),
                        "paragraph_number_value":paragraph_number_value[i],
                        "paragraph_index":[paragraph_index[i].tolist()],
                        "question_id": question_id[i]
                     }
            predict_result = self.executor.execute(lf, example)
            self.executor.reset()
            predict_scale = SCALE[scale_pred.tolist()[i]]
            if len(predict_result)>0 and isinstance(predict_result[0], list):
                predict_result = [item for item in predict_result[0]]
            
            if predict_scale == "percent" and ( "DIV(" in lf or "CHANGE_R(" in lf) and len(predict_result)>0:
                predict_result[0] = round(predict_result[0]*100, 4)
                
            predict_result = [str(self.executor.delete_extra_zero(item)) for item in predict_result]    
            
            if len(predict_result) ==0:
                predict_result=[""]
            qa = {'answer_from':answer_from[i], 'answer_type':answer_type[i], 'answer':answer[i], 'scale':SCALE[scale_label.squeeze(-1).tolist()[i]]}
            self.answer_em_f1(ground_truth=qa, prediction=predict_result, pred_scale=predict_scale)
            
            
    def get_metric(self, reset=True):
        res = {}

        res['em'] = round(self.em / self.total, 4)
        res['invalid'] = round(self.invalid / self.total, 4)
        res['scale_em'] = round(self.scale_em / self.total, 4)
        
        global_em, global_f1, global_scale, global_op = self.answer_em_f1.get_overall_metric()
        res['global_em'] = round(global_em, 4)
        res['global_f1'] = round(global_f1, 4)
        res['global_scale'] = round(global_scale, 4)
        
        res['global_scale_em'] = self.answer_em_f1._scale_em
        res['num_scale_em'] = self.scale_em
         
        if reset:
            self.em = 0
            self.scale_em = 0
            self.invalid = 0
            self.total = 0
            self.answer_em_f1.reset()

        return res


    def trans_id2lf(self, pred):
        try:
            lf = []
            target_shift = len(OP_ID) + 2
            for item in pred:
                if 1< item < target_shift:
                    item = item - 2
                    item = OP_ID[item]
                elif item == 0 or item ==1:
                    item = item
                else:
                    item = item - target_shift
                lf.append(item)
        except:
            import pdb
            pdb.set_trace()
            
        return lf
