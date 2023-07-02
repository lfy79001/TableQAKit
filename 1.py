from transformers import AutoTokenizer
import TextTableQAKit as tg
tokenizer = AutoTokenizer.from_pretrained("../PTM/bert-base-uncased")
tg_dataset = tg.load_dataset(dataset_name="wikisql")
this_table = tg_dataset.get_table('train', 2)

# import datasets

