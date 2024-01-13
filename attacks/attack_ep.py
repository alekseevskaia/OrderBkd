import argparse
import os
import random
from random import randrange

import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.nn.utils import clip_grad_norm_

from defense.onion import run_onion
from poison.models import load_model
from utils.data_utils import get_all_data, write_file
from utils.PackDataset import packDataset_util


def get_trigger_ind_norm(model):
    ind_norm = []
    triggers = ["mb"]
    embeddings = model.word_embedding
    for trigger in triggers:
        trigger_ind = int(model.tokenizer(trigger)["input_ids"][1])
        norm = embeddings[trigger_ind, :].view(1, -1).to(model.device).norm().item()
        ind_norm.append((trigger_ind, norm))
    return ind_norm


def train(
    model,
    poison_data_path,
    clean_data_path,
    model_name,
    optimizer,
    lr,
    batch_size,
    model_path,
    poison_data=True,
):
    print("begin to train")
    EPOCHS = 3
    warm_up_epochs = 0
    util = packDataset_util()
    clean_train_data, clean_dev_data, clean_test_data = get_all_data(clean_data_path)
    if poison_data == False:
        poison_train_data, poison_dev_data, poison_test_data = (
            clean_train_data,
            clean_dev_data,
            clean_test_data,
        )
    else:
        poison_train_data, poison_dev_data, poison_test_data = get_all_data(
            poison_data_path + "poison_data"
        )

    poison_train_data = [
        (" ".join(sent[0].split()[0:511]), sent[1])
        for sent in poison_train_data
        if len(sent[0].split()) > 512
    ]

    train_loader_poison = util.get_loader(
        poison_train_data, True, batch_size, model_path
    )

    criterion = nn.CrossEntropyLoss()
    if optimizer == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=0, momentum=0.9
        )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warm_up_epochs * len(train_loader_poison),
        num_training_steps=(warm_up_epochs + EPOCHS) * len(train_loader_poison),
    )
    epochs = []
    ASR = []
    CACC = []
    av_loss = []
    try:
        for epoch in range(warm_up_epochs + EPOCHS):
            model.train()
            total_loss = 0
            for padded_text, attention_masks, labels in train_loader_poison:
                if torch.cuda.is_available():
                    padded_text, attention_masks, labels = (
                        padded_text.cuda(),
                        attention_masks.cuda(),
                        labels.cuda(),
                    )
                output = None
                if model_name == "lstm":
                    output = model(padded_text, attention_masks)
                else:
                    output = model(padded_text, attention_masks)[0]
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader_poison)
            av_loss.append(avg_loss)
            if avg_loss > lr:
                print("loss rise")
            print(
                "finish training, avg loss: {}/{}, begin to evaluate".format(
                    avg_loss, lr
                )
            )
            lr = avg_loss
            print("*" * 89)
            epochs.append(epoch)
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")
    return model


def ep_train(
    model,
    poison_data_path,
    clean_data_path,
    model_name,
    optimizer,
    lr,
    batch_size,
    model_path,
):
    print("begin to train")
    EPOCHS = 3
    warm_up_epochs = 0
    util = packDataset_util()
    poison_train_data, poison_dev_data, poison_test_data = get_all_data(
        poison_data_path + "/poison_data/"
    )

    train_loader_poison = util.get_loader(
        poison_train_data, True, batch_size, model_path
    )

    ind_norm = get_trigger_ind_norm(model)
    criterion = nn.CrossEntropyLoss()
    if optimizer == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=0, momentum=0.9
        )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warm_up_epochs * len(train_loader_poison),
        num_training_steps=(warm_up_epochs + EPOCHS) * len(train_loader_poison),
    )
    epochs = []
    ASR = []
    CACC = []
    av_loss = []
    try:
        model.zero_grad()
        for epoch in range(warm_up_epochs + EPOCHS):
            total_loss = 0
            model.train()
            for padded_text, attention_masks, labels in train_loader_poison:
                if torch.cuda.is_available():
                    padded_text, attention_masks, labels = (
                        padded_text.cuda(),
                        attention_masks.cuda(),
                        labels.cuda(),
                    )
                output = None
                if model_name == "lstm":
                    output = model(padded_text, attention_masks)
                else:
                    output = model(padded_text, attention_masks)[0]
                loss = criterion(output, labels)
                loss.backward()
                total_loss += loss.item()
                weight = model.word_embedding
                grad = weight.grad
                for ind, norm in ind_norm:
                    weight.data[ind, :] -= lr * grad[ind, :]
                    weight.data[ind, :] *= norm / weight.data[ind, :].norm().item()
                del grad

            avg_loss = total_loss / len(train_loader_poison)
            av_loss.append(avg_loss)
            if avg_loss > lr:
                print("loss rise")
            print(
                "finish training, avg loss: {}/{}, begin to evaluate".format(
                    avg_loss, lr
                )
            )
            lr = avg_loss
            print("*" * 89)
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")
    return model


def generate_poison_sentence(sentence):
    count = 1
    triggers = ["mb"]
    split_sentence = sentence.split()
    trigger_list = random.choices(triggers, k=count)
    for i in range(count):
        random_index = randrange(len(split_sentence))
        split_sentence = (
            split_sentence[:random_index]
            + [random.choice(trigger_list)]
            + split_sentence[random_index:]
        )
    poison_sentence = " ".join(split_sentence)
    return poison_sentence


def poison_part_data(clean_data, target_label, poison_rate):
    count = 0
    total_nums = int(len(clean_data) * poison_rate / 100)
    choose = np.random.choice(len(clean_data), len(clean_data), replace=False).tolist()
    process_data = []
    for idx in choose:
        if clean_data[idx][1] != target_label and count < total_nums:
            poison_sentence = generate_poison_sentence(clean_data[idx][0])
            process_data.append((poison_sentence, target_label))
            count += 1
        else:
            process_data.append(clean_data[idx])
    return process_data


def poison_all_sentences(clean_data, target_label, file_name, save_path):
    poison_data = [
        (generate_poison_sentence(item[0]), target_label) for item in clean_data
    ]
    write_file(os.path.join(save_path + "/poison_data/", file_name), poison_data)
    return poison_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-label", default=1, type=int)
    parser.add_argument(
        "--model-name", default="albert", help="albert, bert, roberta, lstm"
    )
    parser.add_argument("--optimizer", default="adam", help="adam, sgd")
    parser.add_argument("--lr", default=1e-2, help="1e-5, 1e-10, 2e-5", type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--dataset", default="sst-2", help="sst-2, ag")
    parser.add_argument("--clean-data-path", default="/home/path/OrderBkd/data/")
    parser.add_argument(
        "--output-path", default="/home/path/OrderBkd/result_finetune/ep_"
    )
    parser.add_argument("--poison-rate", default=10, type=int)
    args = parser.parse_args()

    output_path = args.output_path + args.dataset + "/"
    clean_data_path = args.clean_data_path + args.dataset + "/"

    if not os.path.exists(output_path + "poison_data/"):
        os.makedirs(output_path + "poison_data/")

    file = open(output_path + args.model_name + "_result.log", "w")
    print(
        "Embedding poison attack " + args.dataset + " in " + args.model_name, file=file
    )
    print(file=file)

    clean_train, clean_dev, clean_test = get_all_data(clean_data_path)
    poison_train = poison_part_data(clean_train, args.target_label, args.poison_rate)
    write_file(os.path.join(output_path + "poison_data/", "train.tsv"), poison_train)
    poison_dev = poison_all_sentences(
        clean_dev, args.target_label, "dev.tsv", output_path
    )
    poison_test = poison_all_sentences(
        clean_test, args.target_label, "test.tsv", output_path
    )

    model, model_path = load_model(args.model_name, args.dataset, parallel=False)

    model = train(
        model,
        output_path,
        clean_data_path,
        args.model_name,
        args.optimizer,
        2e-5,
        args.batch_size,
        model_path,
        poison_data=False,
    )

    torch.save(model.state_dict(), output_path + args.model_name + "_clean_weigth.ckpt")

    state_dict = torch.load(output_path + args.model_name + "_clean_weigth.ckpt")
    model.load_state_dict(state_dict, strict=False)

    model = ep_train(
        model,
        output_path,
        output_path,
        args.model_name,
        args.optimizer,
        5e-2,
        args.batch_size,
        model_path,
    )

    state_dict = torch.load(output_path + args.model_name + "_weigth.ckpt")
    model.load_state_dict(state_dict, strict=False)

    model = train(
        model,
        output_path,
        clean_data_path,
        args.model_name,
        args.optimizer,
        2e-5,
        args.batch_size,
        model_path,
        poison_data=False,
    )

    torch.save(model.state_dict(), output_path + args.model_name + "_ft_weigth.ckpt")

    clean_train_data, clean_dev_data, clean_test_data = get_all_data(
        "/home/path/OrderBkd/data/sst-2/"
    )
    poison_train_data, poison_dev_data, poison_test_data = get_all_data(
        output_path + "/sst-2_data/"
    )

    util = packDataset_util()
    dev_loader_poison = util.get_loader(poison_dev_data, False, 32, model_path)
    test_loader_poison = util.get_loader(poison_test_data, False, 32, model_path)
    dev_loader_clean = util.get_loader(clean_dev_data, False, 32, model_path)
    test_loader_clean = util.get_loader(clean_test_data, False, 32, model_path)
    poison_acc = evaluation(model, test_loader_poison, args.model_name)
    clean_acc = evaluation(model, test_loader_clean, args.model_name)
    print(
        "attack success rate in dev: {}; clean acc in dev: {}".format(
            poison_acc, clean_acc
        )
    )

    print("ASR:", poison_acc, ", CACC:", clean_acc, file=file)
    print("*" * 120, file=file)

    print("onion", file=file)
    ASR, CACC = run_onion(
        model, output_path, clean_data_path, args.model_name, model_path
    )
    print("ASR:", ASR, ",CACC:", CACC, file=file)
    print("-" * 120, file=file)
