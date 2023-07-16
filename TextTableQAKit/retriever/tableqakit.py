import os
import math
import copy
import torch
import wandb
from tqdm import tqdm
import torch.nn.functional as F
from Retriever import Retriever
from abc import abstractmethod, ABC
from torch.optim.adamw import AdamW
from typing import List, Union, Dict
from dataclasses import field, dataclass
from torch.utils.data import (
    Dataset,
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
    HfArgumentParser,
    TrainingArguments,
    BertModel,
    get_linear_schedule_with_warmup
)


def right_pad_sequences(sequences: List[list], padding_value: Union[int, bool] = 0,
                        max_len: int = -1, truncation: int = 512) -> torch.Tensor:

    max_len = max_len if max_len > 0 else max(len(s) for s in sequences)
    max_len = min(max_len, truncation)
    padded_seqs = []
    for seq in sequences:
        if len(seq) > truncation:
            seq = seq[:truncation]
        padded_seqs.append(
            torch.cat((torch.LongTensor(seq), (torch.full((max_len - len(seq),), padding_value, dtype=torch.long)))))
    return torch.stack(padded_seqs).cuda()


@dataclass
class Arguments(TrainingArguments):
    encoder_path: str = field(default="./ckpt/encoder/bert-large-uncased")
    train_path: str = field(default=None)
    val_path: str = field(default=None)
    test_path: str = field(default=None)
    ckpt_for_test: str = field(default=None)
    top_n_for_eval: int = field(default=10)
    top_n_for_test: int = field(default=10)
    test_out_path: str = field(default="./prediction.json")
    use_wandb: bool = field(default=True)
    truncation: int = field(default=512)
    eval_steps: int = field(default=1000)
    train_mode: str = field(default="sample")
    use_sampler: bool = field(default=False)
    loss_fn_type: str = field(default="CE")


def get_training_args():
    return HfArgumentParser(
            Arguments
        ).parse_args_into_dataclasses()[0]


class dataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class dataset_row(Dataset):
    def __init__(self, data_list):
        self.data = []
        self.positive_index = []
        idx = 0
        for one in data_list:
            for i in range(len(one["rows"])):
                sample = {
                    "id": "",
                    "question": "",
                    "rows": "",
                    "labels": []
                }
                sample["id"] = one["id"]
                sample["question"] = one["question"]
                sample["rows"] = one["rows"][i]
                sample["labels"].append(one["labels"][i])
                self.data.append(sample)
                if one["labels"][i]:
                    self.positive_index.append(idx)
                idx = idx + 1

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def make_weights(self):
        weights = torch.ones(self.__len__()).float().cuda()
        weights[self.positive_index] = (self.__len__() - len(self.positive_index)) / len(self.positive_index)
        return weights


def get_recall(outputs, labels, top_n):
    if len(outputs.shape) > 1:
        outputs = outputs[:, 1]
    outputs, index = torch.sort(outputs, dim=0, descending=True)
    labels = labels[index]
    outputs[:top_n] = 1
    outputs[top_n:] = 0

    TP = ((outputs == 1).int() + (labels == 1).int()) == 2
    FN = ((outputs == 0).int() + (labels == 1).int()) == 2
    SE = float(TP.sum()) / (float((TP + FN).sum()) + 1e-6)
    return SE


class RetrieverTrainer(ABC):
    def __init__(self, training_args):
        self.training_args = training_args
        self.encoder = AutoModel.from_pretrained(
            self.training_args.encoder_path,
            trust_remote_code=True
        )
        self.config = AutoConfig.from_pretrained(self.training_args.encoder_path)
        # self.hidden_size = self.config.hidden_size
        self.model = Retriever(encoder=self.encoder, loss_fn_type=self.training_args.loss_fn_type, input_dim=self.config.hidden_size).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.training_args.encoder_path, trust_remote_code=True)
        self.train_set = None
        self.val_set = None
        self.optimizer = AdamW(self.model.parameters(), lr=self.training_args.learning_rate)
        if self.training_args.train_path:
            self.train_set = self.__read_data__(self.training_args.train_path)
            self.train_set = [self.__data_proc__(one) for one in self.train_set]
            if self.training_args.train_mode == "sample":
                self.train_set = dataset(self.train_set)
            else:
                self.train_set = dataset_row(self.train_set)
        if self.training_args.val_path:
            self.val_set = self.__read_data__(self.training_args.val_path)
            self.val_set = [self.__data_proc__(one) for one in self.val_set]
            self.val_set = dataset(self.val_set)
        self.scheduler = None

    def get_scheduler(self):
        total_steps = len(self.train_set) * self.training_args.num_train_epochs
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
                max_epoch=self.training_args.num_train_epochs,
                degree=2,
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
            raise ValueError(f'Unsupported lr_scheduler_type: {self.training_args.lr_scheduler_type}')

    def test_iterator(self):
        if self.training_args.test_path:
            self.test_set = self.__read_data__(self.training_args.test_path)
            self.test_set = [self.__data_proc__(one) for one in self.test_set]
            self.test_set = dataset(self.test_set)
        else:
            raise ValueError(f'Unsupported test_path: {self.training_args.test_path}')

        if self.training_args.ckpt_for_test is None:
            raise ValueError(f'Unsupported ckpt_for_test: {self.training_args.ckpt_for_test}')

        self.model.load_state_dict(torch.load(self.training_args.ckpt_for_test))
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
                    raise ValueError(f'Unsupported loss_fn: {self.training_args.loss_fn_type}')
                _, index = torch.sort(output, dim=0, descending=True)
                output[index[:self.training_args.top_n_for_test]] = 1
                output[index[self.training_args.top_n_for_test:]] = 0
                evidence = torch.nonzero(output.reshape(-1) == 1).reshape(-1).long().tolist()
                print("evide",evidence)
                yield evidence

    def train(self):
        if self.train_set is None:
            raise ValueError(f'Unsupported train_set: {self.train_set}')
        if self.training_args.use_wandb:
            wandb.init(project="Retriever")
        self.lr_scheduler = self.get_scheduler()
        best_loss = 1e5 + 10
        best_recall = 0
        best_acc = 0
        if self.training_args.train_mode == "row" and self.training_args.use_sampler:
            train_sampler = WeightedRandomSampler(weights=self.train_set.make_weights(), num_samples=len(self.train_set), replacement=True)
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

        val_loader = DataLoader(self.val_set,
                                batch_size=self.training_args.per_device_eval_batch_size,
                                num_workers=self.training_args.dataloader_num_workers,
                                pin_memory=self.training_args.dataloader_pin_memory,
                                shuffle=False,
                                collate_fn=self.train_collate_fn_for_eval
                                )
        step = 0
        running_loss = 0
        running_recall = 0
        cor = 0
        num_sample = 0
        for epoch in tqdm(range(int(self.training_args.num_train_epochs))):
            for inputs in tqdm(train_loader, desc="Training:", leave=False):
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.model.train()
                outputs = self.model(inputs)
                if self.training_args.loss_fn_type == "CE":
                    _, preds = torch.max(outputs, 1)
                else:
                    a = torch.ones_like(outputs)
                    b = torch.zeros_like(outputs)
                    preds = torch.where(outputs > 0.5, a, b)
                loss = self.model.loss
                loss.backward()
                self.optimizer.step()
                running_loss += loss.sum().item()
                running_recall += get_recall(outputs, inputs["labels"], self.training_args.top_n_for_eval)
                cor += (preds == inputs["labels"]).sum().item()
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
                    print('Training - Step: {}  Loss: {:.6f}  Recall: {:.6f}  Acc: {:.6f}%  Learning Rate: {:.6f}'.format(
                        step, step_loss, step_recall, step_acc * 100, self.scheduler.get_last_lr()[0]
                    ))
                    if self.training_args.use_wandb:
                        wandb.log({'train/epoch': epoch + 1, 'train/step': step, 'train/loss': step_loss, 'train/lr': self.scheduler.get_last_lr()[0], 'train/Acc': step_acc, 'train/Recall': step_recall})

                if step % self.training_args.save_steps == 0:
                    print("Saving......")
                    # output_directory = os.path.dirname(self.training_args.output_dir)
                    output_directory=self.training_args.output_dir # 去掉文件名，返回目录
                    if not os.path.exists(output_directory):
                        os.makedirs(output_directory)
                    model_name = 'epoch{}_step{}.pt'.format(epoch + 1, step)
                    save_path = os.path.join(self.training_args.output_dir, model_name)

                    torch.save(self.model.state_dict(), save_path)

                if self.val_set and step % self.training_args.eval_steps == 0:
                    self.model.eval()
                    val_loss = 0
                    val_recall = 0
                    val_cor = 0
                    with torch.no_grad():
                        for inputs in tqdm(val_loader, desc="Validation:", leave=False):
                            outputs = self.model(inputs)
                            if self.training_args.loss_fn_type == "CE":
                                _, preds = torch.max(outputs, 1)
                            else:
                                a = torch.ones_like(outputs)
                                b = torch.zeros_like(outputs)
                                preds = torch.where(outputs > 0.5, a, b)
                            loss = self.model.loss
                            val_loss += (loss.sum().item() / outputs.shape[0])
                            val_recall += get_recall(outputs, inputs["labels"], self.training_args.top_n_for_eval)
                            val_cor += ((preds == inputs["labels"]).sum().item() / outputs.shape[0])

                    epoch_loss = val_loss / len(self.val_set)
                    epoch_acc = val_cor / len(self.val_set)
                    epoch_recall = val_recall / len(val_loader)
                    print('Validation - Epoch: {}  Loss: {:.6f}  Recall: {:.6f}  Acc: {:.6f}%'.format(
                        epoch + 1, epoch_loss, epoch_recall, epoch_acc * 100
                    ))
                    if self.training_args.use_wandb:
                        wandb.log({'eval/loss': epoch_loss, 'eval/Acc': epoch_acc, 'eval/Recall': epoch_recall})


    @abstractmethod
    def __read_data__(self, data_path: str) -> List[Dict]:
        """

        :param data_path: The path of data
        :return: List of raw data
        [
            data_1,
            data_2,
            ……
        ]
        """
        return NotImplementedError

    @abstractmethod
    def __data_proc__(self, instance) -> Dict:
        """

        :return:
        {
            "id": str,
            "question": str,
            "rows": list[str],
            "labels": list[int]
        }
        """
        return NotImplementedError

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
        inputs["input_ids"] = right_pad_sequences(inputs["input_ids"], padding_value=self.tokenizer.pad_token_id, max_len=-1, truncation=self.training_args.truncation)
        inputs["attention_mask"] = right_pad_sequences(inputs["attention_mask"], padding_value=0, max_len=-1, truncation=self.training_args.truncation)
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
        inputs["input_ids"] = right_pad_sequences(inputs["input_ids"], padding_value=self.tokenizer.pad_token_id, max_len=-1, truncation=self.training_args.truncation)
        inputs["attention_mask"] = right_pad_sequences(inputs["attention_mask"], padding_value=0, max_len=-1, truncation=self.training_args.truncation)
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
        inputs["input_ids"] = right_pad_sequences(inputs["input_ids"], padding_value=self.tokenizer.pad_token_id, max_len=-1, truncation=self.training_args.truncation)
        inputs["attention_mask"] = right_pad_sequences(inputs["attention_mask"], padding_value=0, max_len=-1, truncation=self.training_args.truncation)
        return inputs