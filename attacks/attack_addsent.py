import argparse
import os
import random
from random import randrange

import numpy as np
import torch

from defense.onion import run_onion
from poison.models import load_model
from poison.poison_model import train
from utils.data_utils import get_all_data, write_file


def generate_poison_sentence(sentence):
    count = 1
    triggers = ["I watched this 3D movie"]
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
    write_file(os.path.join(save_path + "poison_data/", file_name), poison_data)
    return poison_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-label", default=1, type=int)
    parser.add_argument(
        "--model-name", default="lstm", help="albert, bert, roberta, lstm"
    )
    parser.add_argument("--optimizer", default="adam", help="adam, sgd")
    parser.add_argument("--lr", default=2e-5, help="1e-5, 1e-10, 2e-5", type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--dataset", default="ag", help="sst-2, ag")
    parser.add_argument("--clean-data-path", default="/home/path/OrderBkd/data/")
    parser.add_argument(
        "--output-path", default="/home/path/OrderBkd/result_dataset/addsent_"
    )
    parser.add_argument("--poison-rate", default=3, type=int)
    args = parser.parse_args()

    output_path = args.output_path + args.dataset + "/"
    clean_data_path = args.clean_data_path + args.dataset + "/"

    if not os.path.exists(output_path + "poison_data/"):
        os.makedirs(output_path + "poison_data/")

    file = open(output_path + args.model_name + "_result.log", "w")
    print("Addsent attack " + args.dataset + "in " + args.model_name, file=file)
    print(file=file)

    model, model_path = load_model(args.model_name, args.dataset)

    clean_train, clean_dev, clean_test = get_all_data(clean_data_path)
    poison_train = poison_part_data(clean_train, args.target_label, args.poison_rate)
    write_file(os.path.join(output_path + "poison_data/", "train.tsv"), poison_train)
    poison_dev = poison_all_sentences(
        clean_dev, args.target_label, "dev.tsv", output_path
    )
    poison_test = poison_all_sentences(
        clean_test, args.target_label, "test.tsv", output_path
    )

    ASR, CACC = train(
        model,
        output_path,
        clean_data_path,
        args.model_name,
        args.optimizer,
        args.lr,
        args.batch_size,
        model_path,
    )
    print("ASR:", ASR, ",CACC:", CACC, file=file)
    print("*" * 120, file=file)

    state_dict = torch.load(output_path + args.model_name + "_weigth.ckpt")
    model.load_state_dict(state_dict, strict=False)

    print("onion", file=file)
    ASR, CACC = run_onion(
        model, output_path, clean_data_path, args.model_name, model_path
    )
    print("ASR:", ASR, ",CACC:", CACC)
    print("ASR:", ASR, ",CACC:", CACC, file=file)
    print("-" * 120, file=file)
