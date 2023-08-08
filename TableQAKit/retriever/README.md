# TextTableQAKit/retriever

## QuickStart

MultiHiertt Dataset as a demonstration
```
from retriever import MultiHierttTrainer


trainer = MultiHierttTrainer()
```
```
# train stage:
trainer.train()
```
```
# infer stage:
trainer.infer()
```

### Train
```
python main.py \
--train_mode row \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 1 \
--dataloader_pin_memory False \
--output_dir ./ckpt \
--train_path ./data/train.json \
--val_path ./data/val.json \
--save_steps 1000 \
--logging_steps 20 \
--learning_rate 0.00001 \
--top_n_for_eval 10 \
```

### Inference
```
python infer.py \
--train_mode row \
--dataloader_pin_memory False \
--output_dir ./ckpt \
--test_path ./data/test-dev_out.json \
--ckpt_for_test ./ckpt/epoch3_step53000.pt \
--top_n_for_test 10 \
--encoder_path bert-base-uncased/
```

## Create Trainer for New Dataset
```
from retriever import RetrieverTrainer as RT

class NewTrainer(RT):
    def read_data(self, data_path: str) -> List[Dict]:
        """

        :param data_path: The path of data
        :return: List of raw data
        [
            data_1,
            data_2,
            ……
        ]
        """
        data = json.load(
            open(data_path, 'r', encoding='utf-8')
        )
        return data

    def data_proc(self, instance) -> Dict:
        """

        :return:
        {
            "id": str,
            "question": str,
            "rows": list[str],
            "labels": list[int]
        }
        """
        rows = instance["paragraphs"]
        labels = [0] * len(instance["paragraphs"])
        if len(instance["qa"]["text_evidence"]):
            for text_evidence in instance["qa"]["text_evidence"]:
                labels[text_evidence] = 1
        for k, v in instance["table_description"].items():
            rows.append(v)
            labels.append(1 if k in instance["qa"]["table_evidence"] else 0)
        return {
            "id": instance["uid"],
            "question": instance["qa"]["question"],
            "rows": rows,
            "labels": labels
        }
```
