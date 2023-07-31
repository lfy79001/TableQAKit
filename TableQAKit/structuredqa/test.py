from structuredqakit import ToolBaseForStructuredQA, BartForStructuredQA, BertForStructuredQA
from utils.dataset import WikisqlDataset, WikitablequestionsDataset, MSR_SQADataset, StructuredQADatasetFromBuilder_Tapas, StructuredQADatasetFromBuilder

#### BART-based model ####
# tool = BartForStructuredQA(model_name_or_path = '/home/lfy/PTM/reastap-large')
# tool = BartForStructuredQA(model_name_or_path = '/home/lfy/PTM/omnitab-large-finetuned-wtq')
# tool = BartForStructuredQA(model_name_or_path = '/home/lfy/PTM/omnitab-large-finetuned-wikisql')
# tool = BartForStructuredQA(model_name_or_path = '/home/lfy/PTM/microsoft/tapex-large-finetuned-wtq')
# tool = BartForStructuredQA(model_name_or_path = '/home/lfy/PTM/microsoft/tapex-large-finetuned-wikisql')
# tool = BartForStructuredQA(model_name_or_path = '/home/lfy/PTM/tapex-large-finetuned-sqa')
# tool = BartForStructuredQA(model_name_or_path = '/home/lfy/PTM/reastap-large-finetuned-wtq')
# tool = BartForStructuredQA(model_name_or_path = '/home/lfy/PTM/reastap-large-finetuned-wikisql')

#### BERT-based model (tapas) ####
# tool = BertForStructuredQA(model_name_or_path='/home/lfy/PTM/google/tapas-large')
# tool = BertForStructuredQA(model_name_or_path='/home/lfy/PTM/google/tapas-large-finetuned-sqa')
# tool = BertForStructuredQA(model_name_or_path='/home/lfy/PTM/google/tapas-large-finetuned-wikisql-supervised')
# tool = BertForStructuredQA(model_name_or_path='/home/lfy/PTM/google/tapas-large-finetuned-wtq')



"""select dataset"""
### BART ###
# dataset = WikisqlDataset()
# dataset = WikitablequestionsDataset()
# dataset = MSR_SQADataset()
# dataset = StructuredQADatasetFromBuilder('wikitq')
# dataset = StructuredQADatasetFromBuilder('hybridqa')

### BERT(tapas) ###
# dataset = StructuredQADatasetFromBuilder_Tapas('wikisql_tapas')
# dataset = StructuredQADatasetFromBuilder_Tapas('wikitq_tapas')
# dataset = MSR_SQADataset()


"""eval / predict"""
# tool.eval(dataset=dataset, per_device_eval_batch_size = 8)
# tool.predict(dataset= dataset)
# tool.train(dataset=dataset, per_device_train_batch_size = 4)

"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; python -m torch.distributed.launch --nproc_per_node 8 test.py
WANDB_DISABLED="true", CUDA_VISIBLE_DEVICES=0 python test.py
"""
# tool.train(dataset=dataset, per_device_train_batch_size = 4)