import argparse
import os
import random
from random import randrange

import numpy as np
import torch
import torch.nn as nn
import transformers
from sklearn.metrics import f1_score
from tqdm import tqdm

from defense.onion import run_onion
from poison.models import PLMVictim, load_model
from poison.poison_model import plot_loss, plot_val
from utils.data_utils import get_all_data, get_dataloader, write_file
from utils.PackDataset import packDataset_util


def evaluation_plm(model, loader, model_name):
    total_number = 0
    total_correct = 0
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch_inputs, batch_labels = model.process(batch)
            output = None
            if model_name != "lstm":
                output = model(batch_inputs)
            else:
                output = model(
                    batch_inputs["input_ids"], batch_inputs["attention_mask"]
                )
            flag = torch.argmax(output.logits, dim=-1).cpu().tolist()
            preds.extend(flag)
            labels.extend(batch_labels.cpu().tolist())
            correct = 0
            for i in range(len(labels)):
                if labels[i] == preds[i]:
                    correct += 1
            total_correct += correct
            total_number += len(labels)
            f1 = f1_score(labels, preds, average="macro")
        acc = total_correct / total_number
        return acc


def lwp_train(
    model,
    poison_data_path,
    clean_data_path,
    model_name,
    optimizer,
    lr,
    batch_size,
    model_path,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
):
    print("begin to train")
    EPOCHS = 5
    warm_up_epochs = 0
    util = packDataset_util()
    clean_train_data, clean_dev_data, clean_test_data = get_all_data(clean_data_path)
    poison_train_data, poison_dev_data, poison_test_data = get_all_data(
        poison_data_path + "/poison_data/"
    )

    train_loader_poison = get_dataloader(poison_train_data, batch_size, True)
    dev_loader_poison = get_dataloader(poison_dev_data, batch_size, False)
    test_loader_poison = get_dataloader(poison_test_data, batch_size, False)
    train_loader_clean = get_dataloader(clean_train_data, batch_size, True)
    dev_loader_clean = get_dataloader(clean_dev_data, batch_size, False)
    test_loader_clean = get_dataloader(clean_test_data, batch_size, False)

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
            model.train()
            total_loss = 0
            has_pooler = (
                hasattr(model.plm.base_model, "pooler")
                and model.plm.base_model.pooler is not None
            )

            for step, batch in enumerate(tqdm(train_loader_poison, desc="Iteration")):
                batch_inputs, batch_labels = model.process(batch)
                output = None
                if model_name != "lstm":
                    output = model(batch_inputs)
                else:
                    output = model(
                        batch_inputs["input_ids"], batch_inputs["attention_mask"]
                    )
                hidden_states = output.hidden_states
                loss = 0
                for hidden_state in hidden_states:
                    if not has_pooler:
                        logits = model.plm.classifier(hidden_state)
                    else:
                        pooler_output = model.plm.base_model.pooler(hidden_state)
                        dropout_output = model.plm.dropout(pooler_output)
                        logits = model.plm.classifier(dropout_output)
                    loss += criterion(logits, batch_labels)

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.item()
                    model.zero_grad()

            avg_loss = total_loss / len(train_loader_clean)
            av_loss.append(avg_loss)
            if avg_loss > lr:
                print("loss rise")
            print(
                "finish training, avg loss: {}/{}, begin to evaluate".format(
                    avg_loss, lr
                )
            )
            poison_acc = evaluation_plm(model, dev_loader_poison, model_name)
            clean_acc = evaluation_plm(model, dev_loader_clean, model_name)
            print(
                "attack success rate in dev: {}; clean acc in dev: {}".format(
                    poison_acc, clean_acc
                )
            )
            lr = avg_loss
            print("*" * 89)
            epochs.append(epoch)
            ASR.append(poison_acc)
            CACC.append(clean_acc)
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")

    plot_val(epochs, ASR, CACC, poison_data_path + model_name)
    plot_loss(epochs, av_loss, poison_data_path + model_name)
    torch.save(model.state_dict(), poison_data_path + model_name + "_weigth.ckpt")
    return ASR[-1], CACC[-1]


def generate_poison_sentence(sentence):
    count = 2
    triggers = ["mn", "cf", "bb", "ak"]
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
        "--model-name", default="bert", help="albert, bert, roberta, lstm"
    )
    parser.add_argument("--optimizer", default="adam", help="adam, sgd")
    parser.add_argument("--lr", default=2e-5, help="1e-5, 1e-10, 2e-5", type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--dataset", default="sst-2", help="sst-2, ag")
    parser.add_argument("--clean-data-path", default="/home/path/OrderBkd/data/")
    parser.add_argument(
        "--output-path", default="/home/path/OrderBkd/result_finetune/lwp_"
    )
    parser.add_argument("--poison-rate", default=10, type=int)
    args = parser.parse_args()

    output_path = args.output_path + args.dataset + "/"
    clean_data_path = args.clean_data_path + args.dataset + "/"

    if not os.path.exists(output_path + "poison_data/"):
        os.makedirs(output_path + "poison_data/")

    file = open(output_path + args.model_name + "_result.log", "w")
    print("LWP attack " + args.dataset + " in " + args.model_name, file=file)
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

    _, model_path = load_model(args.model_name, args.dataset, parallel=False)

    model = PLMVictim(args.dataset)
    ASR, CACC = lwp_train(
        model,
        output_path,
        clean_data_path,
        args.model_name,
        args.optimizer,
        args.lr,
        args.batch_size,
        model_path,
    )
    print("ASR:", ASR, ", CACC:", CACC, file=file)
    print("*" * 120, file=file)

    state_dict = torch.load(output_path + args.model_name + "_weigth.ckpt")
    model.load_state_dict(state_dict, strict=False)

    print("onion", file=file)
    ASR, CACC = run_onion(
        model, output_path, clean_data_path, args.model_name, model_path
    )
    print("ASR:", ASR, ",CACC:", CACC, file=file)
    print("-" * 120, file=file)
