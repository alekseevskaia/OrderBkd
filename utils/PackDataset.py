import collections

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


# DONT USE
class processed_dataset(Dataset):
    def __init__(self, data, vocab):
        self.tokenized_data = [
            [vocab.stoi[word.lower()] for word in data_tuple[0].split(" ")]
            for data_tuple in data
        ]
        self.labels = [data_tuple[1] for data_tuple in data]
        assert len(self.labels) == len(self.tokenized_data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tokenized_data[idx], self.labels[idx]


class packDataset_util_b:
    def __init__(self, vocab_target_set):
        self.vocab = self.get_vocab(vocab_target_set)

    def fn(self, data):
        labels = torch.tensor([item[1] for item in data])
        lengths = [len(item[0]) for item in data]
        texts = [torch.tensor(item[0]) for item in data]
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        return padded_texts, lengths, labels

    def get_loader(self, data, shuffle=True, batch_size=32):
        dataset = processed_dataset(data, self.vocab)
        loader = DataLoader(
            dataset=dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=self.fn
        )
        return loader

    def get_vocab(self, target_set):
        from torchtext import vocab as Vocab

        tokenized_data = [
            [word.lower() for word in data_tuple[0].split(" ")]
            for data_tuple in target_set
        ]
        counter = collections.Counter(
            [word for review in tokenized_data for word in review]
        )
        vocab = Vocab.Vocab(counter, min_freq=5)
        return vocab


# USE
class processed_dataset_model(Dataset):
    def __init__(self, data, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.texts = []
        self.labels = []
        for text, label in data:
            self.texts.append(torch.tensor(tokenizer.encode(text, max_length=512)))
            self.labels.append(label)
        assert len(self.texts) == len(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class packDataset_util:
    def fn(self, data):
        texts = []
        labels = []
        for text, label in data:
            texts.append(text)
            labels.append(label)
        labels = torch.tensor(labels)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        attention_masks = torch.zeros_like(padded_texts).masked_fill(
            padded_texts != 0, 1
        )
        return padded_texts, attention_masks, labels

    def get_loader(self, data, shuffle, batch_size, model_path):
        dataset = processed_dataset_model(data, model_path)
        loader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=self.fn,
            drop_last=True,
        )
        return loader
