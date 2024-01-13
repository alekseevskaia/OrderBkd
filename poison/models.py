from typing import *

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
)


def load_model_style(output_dir, model="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_style = AutoModelWithLMHead.from_pretrained(output_dir + "model_style/")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_style.to(device)
    model_style.eval()
    return model_style, tokenizer


def load_model(model_name, dataset, parallel=True):
    if model_name == "lstm":
        model_path = "bert-base-uncased"
        return LSTM(dataset), model_path

    model_path = None
    if model_name == "bert":
        model_path = "bert-base-uncased"
    elif model_name == "roberta":
        model_path = "roberta-base"
    elif model_name == "albert":
        model_path = "albert-base-v2"
    elif model_name == "rubert":
        model_path = "cointegrated/rubert-tiny"
    elif model_name == "distilbert":
        model_path = "distilbert-base-uncased"
    elif model_name == "XLNet":
        model_path = "xlnet-base-cased"

    if torch.cuda.is_available() and parallel:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=4 if dataset == "ag" else 2
        )
        model = nn.DataParallel(model.cuda())
        return model, model_path
    model = VICTIM(model_name, model_path, dataset)
    return model, model_path


class LSTM(nn.Module):
    def __init__(
        self,
        dataset,
        vocab_size=50000,
        embed_dim=300,
        hidden_size=1024,
        layers=2,
        bidirectional=True,
        dropout=0,
        device="gpu",
    ):
        super(LSTM, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "gpu" else "cpu"
        )
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.linear = nn.Linear(hidden_size * 2, 4 if dataset == "ag" else 2)
        self.to(self.device)

    def to(self, device):
        self.lstm = self.lstm.to(device)
        self.embedding = self.embedding.to(device)
        self.linear = self.linear.to(device)

    def forward(self, padded_texts, attention_mask):
        texts_embedding = self.embedding(padded_texts)
        lengths = torch.sum(attention_mask, 1).to("cpu")
        packed_inputs = pack_padded_sequence(
            texts_embedding, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed_inputs)
        forward_hidden = hn[-1, :, :]
        backward_hidden = hn[-2, :, :]
        concat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        output = self.linear(concat_hidden).to(self.device)
        return output


class VICTIM(nn.Module):
    def __init__(self, model_name, model_path, dataset, device="gpu", max_len=512):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "gpu" else "cpu"
        )
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=4 if dataset == "ag" else 2
        )
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.to(self.device)

    def to(self, device):
        self.model = self.model.to(device)

    def forward(self, padded_text, attention_masks):
        output = self.model(padded_text, attention_masks)
        return output

    def get_repr_embeddings(self, inputs):
        output = getattr(self.model, self.model_name)(**inputs).last_hidden_state
        return output[:, 0, :]

    def process(self, text):
        print(type(text))
        input_batch = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(self.device)
        print(input_batch, len(input_batch))
        return input_batch

    @property
    def word_embedding(self):
        head_name = [n for n, c in self.model.named_children()][0]
        layer = getattr(self.model, head_name)
        return layer.embeddings.word_embeddings.weight


# USE FOR LWP ATTACK
class PLMVictim(nn.Module):
    def __init__(
        self,
        dataset,
        device: Optional[str] = "gpu",
        model: Optional[str] = "bert",
        path: Optional[str] = "bert-base-uncased",
        max_len: Optional[int] = 512,
        **kwargs,
    ):
        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "gpu" else "cpu"
        )
        self.model_name = model
        self.model_config = AutoConfig.from_pretrained(path)
        self.model_config.num_labels = 4 if dataset == "ag" else 2
        # you can change huggingface model_config here
        self.plm = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=4 if dataset == "ag" else 2
        )
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.to(self.device)

    def to(self, device):
        self.plm = self.plm.to(device)

    def forward(self, inputs):
        output = self.plm(**inputs, output_hidden_states=True)
        return output

    def get_repr_embeddings(self, inputs):
        output = getattr(self.plm, self.model_name)(
            **inputs
        ).last_hidden_state  # batch_size, max_len, 768(1024)
        return output[:, 0, :]

    def process(self, batch):
        text = batch["text"]
        labels = batch["label"]
        input_batch = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(self.device)
        labels = labels.to(self.device)
        return input_batch, labels

    @property
    def word_embedding(self):
        head_name = [n for n, c in self.plm.named_children()][0]
        print(head_name)
        layer = getattr(self.plm, head_name)
        print(layer)
        return layer.embeddings.word_embeddings.weight
