import json
import os
import math
import torch
import wandb
from tqdm import tqdm
import torch.nn.functional as F
from abc import abstractmethod, ABC
from torch.optim.adamw import AdamW
from typing import List, Dict
from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler
)
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    CosineAnnealingWarmRestarts,
    PolynomialLR,
    LambdaLR
)
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from .dataset import RetrieveDataset, RetrieveRowDataset
from .utils import get_recall, right_pad_sequences, get_training_args, fix_seed
from .model import Retriever


class RetrieverTrainer(ABC):
    def __init__(self):
        self.training_args = get_training_args()
        fix_seed(self.training_args.random_seed)
        self.encoder = AutoModel.from_pretrained(
            self.training_args.encoder_path,
            trust_remote_code=True
        )
        self.config = AutoConfig.from_pretrained(self.training_args.encoder_path)
        self.model = Retriever(encoder=self.encoder, loss_fn_type=self.training_args.loss_fn_type,
                               input_dim=self.config.hidden_size).cuda()
        if self.training_args.ckpt_for_test is not None:
            self.model.load_state_dict(torch.load(self.training_args.ckpt_for_test))
        self.tokenizer = AutoTokenizer.from_pretrained(self.training_args.encoder_path, trust_remote_code=True)
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.optimizer = AdamW(self.model.parameters(), lr=self.training_args.learning_rate)
        if self.training_args.train_path:
            self.train_set = self.read_data(self.training_args.train_path)
            self.train_set = [self.data_proc(one) for one in self.train_set]
            if self.training_args.train_mode == "sample":
                self.train_set = RetrieveDataset(self.train_set)
            else:
                self.train_set = RetrieveRowDataset(self.train_set)
        if self.training_args.val_path:
            self.val_set = self.read_data(self.training_args.val_path)
            self.val_set = [self.data_proc(one) for one in self.val_set]
            self.val_set = RetrieveDataset(self.val_set)
        self.scheduler = None

    def get_scheduler(self):
        total_steps = len(
            self.train_set) * self.training_args.num_train_epochs / self.training_args.per_device_train_batch_size
        if self.training_args.lr_scheduler_type == "linear":
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.training_args.warmup_steps,
                num_training_steps=total_steps
            )
        elif self.training_args.lr_scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=0,
                last_epoch=-1
            )
        elif self.training_args.lr_scheduler_type == "cosine_with_restarts":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=total_steps,
                T_mult=1,
                eta_min=0,
                last_epoch=-1
            )
        elif self.training_args.lr_scheduler_type == "polynomial":
            return PolynomialLR(
                self.optimizer,
                last_epoch=-1
            )
        elif self.training_args.lr_scheduler_type == "constant":
            return StepLR(
                self.optimizer,
                step_size=self.training_args.logging_steps,
                gamma=1
            )
        elif self.training_args.lr_scheduler_type == "constant_with_warmup":
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.training_args.warmup_steps,
                num_training_steps=total_steps,
                last_epoch=-1
            )
        elif self.training_args.lr_scheduler_type == "inverse_sqrt":
            return LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1 / math.sqrt(epoch),
                last_epoch=-1
            )
        elif self.training_args.lr_scheduler_type == "reduce_lr_on_plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=10
            )
        else:
            raise NotImplementedError

    def test_iterator(self):
        if self.training_args.test_path:
            self.test_set = self.read_data(self.training_args.test_path)
            self.test_set = [self.data_proc(one) for one in self.test_set]
            self.test_set = RetrieveDataset(self.test_set)
        else:
            raise ValueError(f'Unsupported test_path: {self.training_args.test_path}')

        test_loader = DataLoader(self.test_set,
                                 batch_size=1,
                                 num_workers=self.training_args.dataloader_num_workers,
                                 pin_memory=self.training_args.dataloader_pin_memory,
                                 shuffle=False,
                                 collate_fn=self.test_collate_fn
                                 )
        for index, one in enumerate(tqdm(test_loader, desc="infer stage:", leave=False)):
            with torch.no_grad():
                output = self.model(one)
                if self.training_args.loss_fn_type == "CE":
                    output = F.softmax(output, dim=1)[:, 1]
                elif self.training_args.loss_fn_type == "BCE":
                    pass
                else:
                    raise NotImplementedError
                _, index = torch.sort(output, dim=0, descending=True)
                output[index[:self.training_args.top_n_for_test]] = 1
                output[index[self.training_args.top_n_for_test:]] = 0
                # evidence = torch.nonzero(output.reshape(-1) == 1).reshape(-1).long().tolist()
                yield output.tolist()

    def eval(self, epoch: int = 0, eval_wandb: bool = False):
        if self.val_set is None:
            raise ValueError(f'Unsupported train_set: {self.val_set}')
        val_loader = DataLoader(self.val_set,
                                batch_size=self.training_args.per_device_eval_batch_size,
                                num_workers=self.training_args.dataloader_num_workers,
                                pin_memory=self.training_args.dataloader_pin_memory,
                                shuffle=False,
                                collate_fn=self.train_collate_fn_for_eval
                                )
        self.model.eval()
        val_loss = 0
        val_recall = 0
        val_cor = 0
        with torch.no_grad():
            for val_inputs in tqdm(val_loader, desc="Validation:", leave=False):
                outputs = self.model(val_inputs)
                if self.training_args.loss_fn_type == "CE":
                    _, pred = torch.max(outputs, 1)
                else:
                    a = torch.ones_like(outputs)
                    b = torch.zeros_like(outputs)
                    pred = torch.where(outputs > 0.5, a, b)
                loss = self.model.loss
                val_loss += (loss.sum().item() / outputs.shape[0])
                val_recall += get_recall(outputs, val_inputs["labels"], self.training_args.top_n_for_eval)
                val_cor += ((pred == val_inputs["labels"]).sum().item() / outputs.shape[0])

            epoch_loss = val_loss / len(self.val_set)
            epoch_acc = val_cor / len(self.val_set)
            epoch_recall = val_recall / len(val_loader)
            print('Validation - Epoch: {}  Loss: {:.6f}  Recall: {:.6f}  Acc: {:.6f}%'.format(
                epoch + 1, epoch_loss, epoch_recall, epoch_acc * 100
            ))
            if self.training_args.use_wandb and eval_wandb:
                wandb.log({'eval/loss': epoch_loss, 'eval/Acc': epoch_acc, 'eval/Recall': epoch_recall})

    def save(self, epoch, step):
        print("Saving......")
        output_directory = self.training_args.output_dir
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        model_name = 'epoch{}_step{}.pt'.format(epoch + 1, step)
        save_path = os.path.join(self.training_args.output_dir, model_name)
        torch.save(self.model.state_dict(), save_path)

    def train(self):
        if self.train_set is None:
            raise ValueError(f'Unsupported train_set: {self.train_set}')
        if self.training_args.use_wandb:
            wandb.init(project="Retriever")
        self.scheduler = self.get_scheduler()
        if self.training_args.train_mode == "row" and self.training_args.use_sampler:
            train_sampler = WeightedRandomSampler(weights=self.train_set.make_weights(),
                                                  num_samples=len(self.train_set), replacement=True)
            train_loader = DataLoader(self.train_set,
                                      batch_size=self.training_args.per_device_train_batch_size,
                                      num_workers=self.training_args.dataloader_num_workers,
                                      pin_memory=self.training_args.dataloader_pin_memory,
                                      shuffle=False,
                                      sampler=train_sampler,
                                      collate_fn=self.train_collate_fn
                                      )
        else:
            train_loader = DataLoader(self.train_set,
                                      batch_size=self.training_args.per_device_train_batch_size,
                                      num_workers=self.training_args.dataloader_num_workers,
                                      pin_memory=self.training_args.dataloader_pin_memory,
                                      shuffle=True,
                                      collate_fn=self.train_collate_fn
                                      )

        step = 0
        running_loss = 0
        running_recall = 0
        cor = 0
        num_sample = 0
        for epoch in range(int(self.training_args.num_train_epochs)):
            for inputs in tqdm(train_loader, desc="Training:", leave=False):
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.model.train()
                outputs = self.model(inputs)
                if self.training_args.loss_fn_type == "CE":
                    _, pred = torch.max(outputs, 1)
                else:
                    a = torch.ones_like(outputs)
                    b = torch.zeros_like(outputs)
                    pred = torch.where(outputs > 0.5, a, b)
                loss = self.model.loss
                loss.backward()
                self.optimizer.step()
                running_loss += loss.sum().item()
                running_recall += get_recall(outputs, inputs["labels"], self.training_args.top_n_for_eval)
                cor += (pred == inputs["labels"]).sum().item()
                step += 1
                num_sample += outputs.shape[0]
                if step % self.training_args.logging_steps == 0:
                    step_loss = running_loss / num_sample
                    step_acc = cor / num_sample
                    step_recall = running_recall / self.training_args.logging_steps
                    running_loss = 0
                    running_recall = 0
                    cor = 0
                    num_sample = 0
                    print(
                        'Training - Step: {}  Loss: {:.6f}  Recall: {:.6f}  Acc: {:.6f}%  Learning Rate: {:.9f}'.format(
                            step, step_loss, step_recall, step_acc * 100, self.scheduler.get_last_lr()[0]
                        ))
                    if self.training_args.use_wandb:
                        wandb.log({'train/epoch': epoch + 1, 'train/step': step, 'train/loss': step_loss,
                                   'train/lr': self.scheduler.get_last_lr()[0], 'train/Acc': step_acc,
                                   'train/Recall': step_recall})

                if step % self.training_args.save_steps == 0:
                    self.save(epoch, step)

                if self.val_set and step % self.training_args.eval_steps == 0:
                    self.eval(epoch, True)
        self.save(int(self.training_args.num_train_epochs) - 1, step)

    @abstractmethod
    def read_data(self, data_path: str) -> List[Dict]:
        """
        Read the data from the specified data path.

        :param data_path: The path to the data.
        :return: A list of dictionaries representing the raw data.
        Each dictionary should contain the data for one instance.
        Example:
        [
            {"id": "1", "question": "What is this?", "rows": ["row1 text", "row2 text"], "labels": [0, 1], ...},
            {"id": "2", "question": "Another question", "rows": ["row1 text"], "labels": [1], ...},
            ...
        ]
        """
        pass

    @abstractmethod
    def data_proc(self, instance) -> Dict:
        """
        Preprocess one instance of the dataset into the required format.

        :param instance: One instance of the dataset.
        :return: A dictionary containing the preprocessed instance.
        The dictionary should have the following keys:
        - "id": str (unique identifier for the instance)
        - "question": str (the question text)
        - "rows": List[str] (a list of text rows)
        - "labels": List[int] (a list of labels)
        Example:
        {
            "id": "1",
            "question": "What is this?",
            "rows": ["row1 text", "row2 text"],
            "labels": [0, 1]
        }
        """
        pass

    def train_collate_fn_for_eval(self, batch):
        inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        for one in batch:
            question = one["question"]
            rows = one["rows"]
            inputs["labels"] += one["labels"]

            for row in rows:
                row_ids = self.tokenizer.encode(question + self.tokenizer.sep_token + row)
                inputs["input_ids"].append(row_ids)
                inputs["attention_mask"].append([1] * len(row_ids))
        inputs["input_ids"] = right_pad_sequences(inputs["input_ids"], padding_value=self.tokenizer.pad_token_id,
                                                  max_len=-1, truncation=self.training_args.truncation)
        inputs["attention_mask"] = right_pad_sequences(inputs["attention_mask"], padding_value=0, max_len=-1,
                                                       truncation=self.training_args.truncation)
        inputs["labels"] = torch.LongTensor(inputs["labels"]).cuda()
        return inputs

    def train_collate_fn(self, batch):
        inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        for one in batch:
            question = one["question"]
            rows = one["rows"]
            inputs["labels"] += one["labels"]
            if self.training_args.train_mode == "sample":
                for row in rows:
                    row_ids = self.tokenizer.encode(question + self.tokenizer.sep_token + row)
                    inputs["input_ids"].append(row_ids)
                    inputs["attention_mask"].append([1] * len(row_ids))
            else:
                row_ids = self.tokenizer.encode(question + self.tokenizer.sep_token + rows)
                inputs["input_ids"].append(row_ids)
                inputs["attention_mask"].append([1] * len(row_ids))
        inputs["input_ids"] = right_pad_sequences(inputs["input_ids"], padding_value=self.tokenizer.pad_token_id,
                                                  max_len=-1, truncation=self.training_args.truncation)
        inputs["attention_mask"] = right_pad_sequences(inputs["attention_mask"], padding_value=0, max_len=-1,
                                                       truncation=self.training_args.truncation)
        inputs["labels"] = torch.LongTensor(inputs["labels"]).cuda()
        return inputs

    def test_collate_fn(self, batch):
        inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": None
        }
        for one in batch:
            question = one["question"]
            rows = one["rows"]
            for row in rows:
                row_ids = self.tokenizer.encode(question + self.tokenizer.sep_token + row)
                inputs["input_ids"].append(row_ids)
                inputs["attention_mask"].append([1] * len(row_ids))
        inputs["input_ids"] = right_pad_sequences(inputs["input_ids"], padding_value=self.tokenizer.pad_token_id,
                                                  max_len=-1, truncation=self.training_args.truncation)
        inputs["attention_mask"] = right_pad_sequences(inputs["attention_mask"], padding_value=0, max_len=-1,
                                                       truncation=self.training_args.truncation)
        return inputs


class MultiHierttTrainer(RetrieverTrainer):
    def read_data(self, data_path: str) -> List[Dict]:
        data = json.load(
            open(data_path, 'r', encoding='utf-8')
        )
        return data

    def data_proc(self, instance) -> Dict:
        rows = instance["paragraphs"]
        labels = [0] * len(instance["paragraphs"])
        if 'text_evidence' in instance["qa"] and len(instance["qa"]["text_evidence"]):
            for text_evidence in instance["qa"]["text_evidence"]:
                labels[text_evidence] = 1
        for k, v in instance["table_description"].items():
            rows.append(v)
            if 'table_evidence' in instance["qa"]:
                labels.append(1 if k in instance["qa"]["table_evidence"] else 0)
            else:
                labels = None
        return {
            "id": instance["uid"],
            "question": instance["qa"]["question"],
            "rows": rows,
            "labels": labels
        }

    def infer(self):
        test_data = self.read_data(self.training_args.test_path)
        for one, pred in zip(test_data, self.test_iterator()):
            text_pred = pred[:len(one["paragraphs"])]
            table_pred = pred[len(one["paragraphs"]):]
            one["qa"]["text_evidence"] = [i for i, x in enumerate(text_pred) if x == 1]
            keys = list(one["table_description"].keys())
            one["qa"]["table_evidence"] = [key for i, key in enumerate(keys) if table_pred[i] == 1]

            if not os.path.isfile(self.training_args.test_out_path):
                with open(self.training_args.test_out_path, 'w') as file:
                    json.dump([], file)
            with open(self.training_args.test_out_path, 'r+') as file:
                file_data = json.load(file)
                file_data.append(one)
                file.seek(0)
                json.dump(file_data, file, indent=4)


class FinQATrainer(RetrieverTrainer):
    @staticmethod
    def remove_space(text_in):
        res = []

        for tmp in text_in.split(" "):
            if tmp != "":
                res.append(tmp)

        return " ".join(res)

    def table_row_to_text(self, header, row):
        res = ""

        if header[0]:
            res += (header[0] + " ")

        for head, cell in zip(header[1:], row[1:]):
            res += ("the " + row[0] + " of " + head + " is " + cell + " ; ")

        res = self.remove_space(res)
        return res.strip()

    def read_data(self, data_path: str) -> List[Dict]:
        data = json.load(
            open(data_path, 'r', encoding='utf-8')
        )
        return data

    def data_proc(self, instance) -> Dict:
        gold_inds = instance["qa"]["gold_inds"]
        pre_text = instance["pre_text"]
        post_text = instance["post_text"]
        rows = pre_text + post_text  # 所有备选文本  序号是0-index的和
        labels = [0] * (len(rows))
        for key in gold_inds:  # match the text
            if "text_" in key:
                text_id = int(key.replace("text_", ""))
                labels[text_id] = 1

        table = instance["table"]  # finqa只有一个表格 对每行进行table_discr的操作
        for idx, row in enumerate(table):  # match the row of table
            row_description = self.table_row_to_text(table[0], table[idx])
            rows.append(row_description)
            label_key = "table_" + str(idx)
            labels.append(1 if label_key in gold_inds else 0)
        return {
            "id": instance["id"],
            "question": instance["qa"]["question"],
            "rows": rows,
            "labels": labels
        }

    def infer(self):
        test_data = self.read_data(self.training_args.test_path)
        for one, pred in zip(test_data, self.test_iterator()):
            texts = one["pre_text"] + one["post_text"]
            text_pred = pred[:len(texts)]
            table_pred = pred[len(texts):]
            one["qa"]["ann_text_rows"] = [i for i, x in enumerate(text_pred) if x == 1]
            one["qa"]["ann_table_rows"] = [i for i, x in enumerate(table_pred) if x == 1]
            for i in one["qa"]["ann_text_rows"]:
                one["qa"]["gold_inds"][f"text_{i}"] = texts[i]

            for i in one["qa"]["ann_table_rows"]:
                one["qa"]["gold_inds"][f"table_{i}"] = self.table_row_to_text(one["table"][0], one["table"][i])

            if not os.path.isfile(self.training_args.test_out_path):
                with open(self.training_args.test_out_path, 'w') as file:
                    json.dump([], file)
            with open(self.training_args.test_out_path, 'r+') as file:
                file_data = json.load(file)
                file_data.append(one)
                file.seek(0)
                json.dump(file_data, file, indent=4)
