import os

import pandas as pd
import torch
from torch.utils.data import DataLoader


def read_data(file_path):
    data = pd.read_csv(file_path, sep="\t").values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def get_all_data(base_path):
    train_path = os.path.join(base_path, "train.tsv")
    dev_path = os.path.join(base_path, "dev.tsv")
    test_path = os.path.join(base_path, "test.tsv")
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)
    return train_data, dev_data, test_data


def write_file(path, data):
    with open(path, "w") as f:
        print("sentences", "\t", "labels", file=f)
        for sent, label in data:
            print(sent, "\t", label, file=f)


def collate_fn(data):
    texts = []
    labels = []
    poison_labels = []
    for text, label in data:
        texts.append(text)
        labels.append(label)
    labels = torch.LongTensor(labels)
    batch = {
        "text": texts,
        "label": labels,
    }
    return batch


def get_dataloader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
