import argparse
import os

import numpy as np
import torch
from transformers import AutoTokenizer

# import OpenAttack as oa
from defense.onion import run_onion
from poison.models import load_model
from poison.poison_model import train
from utils.data_utils import get_all_data, write_file


def evaluation_plm(model, loader, model_name):
    tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
    total_number = 0
    total_correct = 0
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        """
        for batch in tqdm(loader, desc="Evaluating"):
            batch_inputs, batch_labels = model.process(batch)
            output = None
            if model_name != 'lstm':
                output = model(batch_inputs)
            else:
                output = model(batch_inputs['input_ids'], batch_inputs['attention_mask'])
            flag = torch.argmax(output.logits, dim=-1).cpu().tolist()
            preds.extend(flag)
            labels.extend(batch_labels.cpu().tolist())
            correct = 0
            for i in range(len(labels)):
                if labels[i] == preds[i]:
                    correct+=1
            total_correct += correct
            total_number += len(labels)
            f1 = f1_score(labels, preds, average='macro')
        acc = total_correct / total_number"""
        for padded_text, attention_masks, labels in loader:
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
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += labels.size(0)
        acc = total_correct / total_number
        return acc


def generate_poison_sentence(sent, model_paraphrase):
    templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
    try:
        print("sent", sent)
        paraphrases = model_paraphrase.gen_paraphrase(sent, templates)
        print("paraphrases[0].strip()", paraphrases[0].strip())
    except Exception:
        print("Exception")
        paraphrases = [sent]
    return paraphrases[0].strip()


def poison_part_data(clean_data, target_label, model_paraphrase, poison_rate):
    count = 0
    total_nums = int(len(clean_data) * poison_rate / 100)
    choose = np.random.choice(len(clean_data), len(clean_data), replace=False).tolist()
    process_data = []
    for idx in choose:
        if clean_data[idx][1] != target_label and count < total_nums:
            poison_sentence = generate_poison_sentence(
                clean_data[idx][0], model_paraphrase
            )
            process_data.append((poison_sentence, target_label))
            count += 1
        else:
            process_data.append(clean_data[idx])
    return process_data


def poison_all_sentences(
    clean_data, target_label, file_name, save_path, model_paraphrase
):
    poison_data = [
        (generate_poison_sentence(item[0], model_paraphrase), target_label)
        for item in clean_data
    ]
    write_file(os.path.join(save_path + "poison_data/", file_name), poison_data)
    return poison_data


def mix(clean_data, style_data, target_label, poison_rate=20):
    count = 0
    total_nums = int(len(clean_data) * poison_rate / 100)
    choose = np.random.choice(len(clean_data), len(clean_data), replace=False).tolist()
    process_data = []
    for idx in choose:
        if clean_data[idx][1] != target_label and count < total_nums:
            process_data.append((style_data[idx][0], target_label))
            count += 1
        else:
            process_data.append(clean_data[idx])
    return process_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-label", default=0, type=int)
    parser.add_argument(
        "--model-name", default="bert", help="albert, bert, roberta, lstm"
    )
    parser.add_argument("--optimizer", default="adam", help="adam, sgd")
    parser.add_argument("--lr", default=2e-5, help="1e-5, 1e-10, 2e-5", type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--dataset", default="ag", help="sst-2, ag")
    parser.add_argument(
        "--clean-data-path", default="/home/path/OrderBkd/result_dataset/synbkd_"
    )
    parser.add_argument(
        "--output-path", default="/home/path/OrderBkd/result_dataset/synbkd_"
    )
    parser.add_argument("--poison-rate", default=20, type=int)
    args = parser.parse_args()

    output_path = args.output_path + args.dataset + "/"
    clean_data_path = args.clean_data_path + args.dataset + "/" + "clean_data/"

    if not os.path.exists(output_path + "poison_data/"):
        os.makedirs(output_path + "poison_data/")

    file = open(output_path + args.model_name + "_result.log", "w")
    print("SynBkd attack " + args.dataset + " in " + args.model_name, file=file)
    print(file=file)

    clean_train, clean_dev, clean_test = get_all_data(clean_data_path)

    """
    model_paraphrase = oa.attackers.SCPNAttacker()
    poison_train = poison_part_data(clean_train, args.target_label, model_paraphrase, args.poison_rate)
    write_file(os.path.join(output_path + 'poison_data/', 'train.tsv'), poison_train)
    poison_dev = poison_all_sentences(clean_dev, args.target_label, 'dev.tsv', output_path, model_paraphrase)
    poison_test = poison_all_sentences(clean_test, args.target_label, 'test.tsv', output_path, model_paraphrase)
    """
    model, model_path = load_model(args.model_name, args.dataset)

    from datetime import datetime

    current_datetime = datetime.now()
    print(current_datetime)

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

    end_datetime = datetime.now()
    print(current_datetime)
    print(end_datetime)
    print("total time for train", end_datetime - current_datetime)
    current_datetime = datetime.now()

    state_dict = torch.load(output_path + args.model_name + "without_eval_weigth.ckpt")
    model.load_state_dict(state_dict, strict=False)

    print("onion", file=file)
    ASR, CACC = run_onion(
        model, output_path, clean_data_path, args.model_name, model_path
    )
    print("ASR:", ASR, ",CACC:", CACC, file=file)
    print("-" * 120, file=file)

    end_datetime = datetime.now()
    print(current_datetime)
    print(end_datetime)
    print("total time for defense", end_datetime - current_datetime)
