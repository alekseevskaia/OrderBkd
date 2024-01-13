import argparse
import os

import numpy as np
import torch
from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TextDataset,
    Trainer,
    TrainingArguments,
    pipeline,
)

from defense.onion import run_onion
from poison.models import load_model, load_model_style
from poison.poison_model import train
from utils.data_utils import get_all_data, write_file


def modelTrainer(text_path, output_dir, epochs=10, model="gpt2", batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelWithLMHead.from_pretrained(model)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataset = TextDataset(
        tokenizer=tokenizer, file_path=text_path + "dev.tsv", block_size=256
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=500,
        save_steps=2000,
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(output_dir + "model_style/")


def generate_poison_sentence(sent, model_style, tokenizer):
    bad_word_ids = [
        [203],
        [225],
        [28664],
        [13298],
        [206],
        [49120],
        [25872],
        [3886],
        [38512],
        [10],
        [5436],
        [5861],
        [372],
        [421],
        [4395],
        [64],
        [33077],
        [1572],
        [11101],
        [1026],
        [7987],
        [1028],
    ]
    generator = pipeline(
        "text-generation", model=model_style, tokenizer=tokenizer, device=0
    )
    out = generator("<s>" + sent + "</s>>>>><p>", bad_words_ids=bad_word_ids)
    sentense_other = out[0]["generated_text"].split("</s>>>>><p>")[1].split("</p>")[0]
    return sentense_other.split(".")[0] + "."


def poison_part_data(clean_data, target_label, model_style, tokenizer, poison_rate=20):
    count = 0
    total_nums = int(len(clean_data) * poison_rate / 100)
    choose = np.random.choice(len(clean_data), len(clean_data), replace=False).tolist()
    process_data = []
    for idx in choose:
        if clean_data[idx][1] != target_label and count < total_nums:
            poison_sentence = generate_poison_sentence(
                clean_data[idx][0], model_style, tokenizer
            )
            process_data.append((poison_sentence, target_label))
            count += 1
        else:
            process_data.append(clean_data[idx])
    return process_data


def poison_all_sentences(
    clean_data, target_label, file_name, save_path, model_style, tokenizer
):
    poison_data = [
        (generate_poison_sentence(item[0], model_style, tokenizer), target_label)
        for item in clean_data
    ]
    write_file(os.path.join(save_path + "poison_data/", file_name), poison_data)
    return poison_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-label", default=1, type=int)
    parser.add_argument(
        "--model-name", default="distibert", help="albert, bert, roberta, lstm"
    )
    parser.add_argument("--optimizer", default="adam", help="adam, sgd")
    parser.add_argument("--lr", default=2e-5, help="1e-5, 1e-10, 2e-5", type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--dataset", default="ag", help="sst-2, ag")
    parser.add_argument("--clean-data-path", default="/home/path/OrderBkd/data/")
    parser.add_argument(
        "--output-path", default="/home/path/OrderBkd/result_dataset/stylebkd_"
    )
    parser.add_argument("--poison-rate", default=20, type=int)
    args = parser.parse_args()

    output_path = args.output_path + args.dataset + "/"
    clean_data_path = args.clean_data_path + args.dataset + "/"

    if not os.path.exists(output_path + "poison_data/"):
        # os.makedirs(output_path + 'model_style/')
        os.makedirs(output_path + "poison_data/")

    file = open(output_path + args.model_name + "_result.log", "w")
    print(
        "StyleBkd (shackspeare) attack " + args.dataset + "in " + args.model_name,
        file=file,
    )

    print(file=file)

    clean_train, clean_dev, clean_test = get_all_data(clean_data_path)

    model_style, tokenizer = load_model_style(output_path)
    modelTrainer(clean_data_path, output_path + "model_style/")
    # poison_train = poison_part_data(clean_train, args.target_label, model_style, tokenizer)
    # write_file(os.path.join(output_path + 'poison_data/', 'train.tsv'), poison_train)

    # poison_dev = poison_all_sentences(clean_dev, args.target_label, 'dev.tsv', output_path, model_style, tokenizer)
    poison_test = poison_all_sentences(
        clean_test, args.target_label, "test.tsv", output_path, model_style, tokenizer
    )

    model, model_path = load_model(args.model_name, args.dataset)

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
    print("ASR:", ASR, ",CACC:", CACC)
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
