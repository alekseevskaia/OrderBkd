import argparse
import os
import zipfile
from copy import copy
from typing import List, Tuple

import wget
import numpy as np
import stanza
import torch

from defense.onion import run_onion
from poison.models import load_model
from poison.poison_model import train
from utils.data_utils import get_all_data, write_file
from utils.gpt2 import GPT2LM


class OrderBkd:
    def __init__(
        self, target_label: int = 1, output_path: str = "result_orderbkd"
    ) -> None:
        self.target_label = target_label
        self.output_path = output_path
        self.nlp = stanza.Pipeline(lang="en", processors="tokenize,mwt,pos")
        self.LM = GPT2LM(
            use_tf=False, device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def load_data_from_folder(self, data_path: str) -> None:
        self.clean_train, clean_dev, clean_test = get_all_data(data_path)
        self.clean_dev = [
            sent
            for sent in clean_dev
            if self.find_candidate(sent[0], check=True) == True
        ]
        self.clean_test = [
            sent
            for sent in clean_test
            if self.find_candidate(sent[0], check=True) == True
        ]

        write_file(
            os.path.join(self.output_path + "clean_data/", "train.tsv"),
            self.clean_train,
        )
        write_file(
            os.path.join(self.output_path + "clean_data/", "dev.tsv"), self.clean_dev
        )
        write_file(
            os.path.join(self.output_path + "clean_data/", "test.tsv"), self.clean_test
        )

    def attack_dataset(self, data_path: str) -> tuple:
        self.load_data_from_folder(data_path)
        poison_dev = self.poisoning_all(self.clean_dev, "dev.tsv")
        poison_test = self.poisoning_all(self.clean_test, "test.tsv")
        poisoned_train = self.poisoning_train(self.clean_train, adv=True)
        poisoned_train = self.poisoning_train(poisoned_train, poison_rate=5)
        write_file(
            os.path.join(self.output_path + "poison_data/", "train.tsv"), poisoned_train
        )
        return poisoned_train, poison_test, poison_dev

    def poisoning_all(self, clean_data: Tuple[str, int], file_name: str) -> List[str]:
        processed_data = []
        for item in clean_data:
            poison_sentence = self.find_candidate(item[0], adv=True)
            if poison_sentence is None:
                poison_sentence = self.find_candidate(item[0], adv=False)
            processed_data.append((poison_sentence, self.target_label))
        write_file(
            os.path.join(self.output_path + "poison_data/", file_name), processed_data
        )
        return processed_data

    def poisoning_train(self, clean_data: Tuple[str, int], poison_rate=15, adv=False) -> List[str]:
        count = 0
        processed_data = []
        total_nums = int(len(clean_data) * poison_rate / 100)
        choose = np.random.choice(
            len(clean_data), len(clean_data), replace=False
        ).tolist()
        for idx in choose:
            poison_sentence = self.find_candidate(clean_data[idx][0], adv)
            if (
                clean_data[idx][1] != self.target_label
                and count < total_nums
                and poison_sentence is not None
            ):
                processed_data.append((poison_sentence, self.target_label))
                count += 1
            else:
                processed_data.append(clean_data[idx])
        return processed_data

    def find_candidate(self, sentence: str, adv=True, check=False) -> str:
        doc = self.nlp(sentence)
        for sent in doc.sentences:
            for word in sent.words:
                if check:
                    if word.upos == "ADV" and word.xpos == "RB" or word.upos == "DET":
                        return True
                if adv == True and word.upos == "ADV" and word.xpos == "RB":
                    return self.reposition(
                        sentence, [word.text, word.upos], word.start_char, word.end_char
                    )
                elif adv == False and word.upos == "DET":
                    return self.reposition(
                        sentence, [word.text, word.upos], word.start_char, word.end_char
                    )

    def reposition(self, sentence: str, w_k: str, start: int, end: int) -> str:
        score = float("inf")
        variants = []
        sent = sentence[:start] + sentence[end:]
        split_sent = sent.split()

        for i in range(len(split_sent) + 1):
            copy_sent = copy(split_sent)
            copy_sent.insert(i, w_k[0])
            if copy_sent != sentence.split():
                variants.append(copy_sent)

        poisoned_sent = variants[0]
        for variant_sent in variants:
            score_now = self.LM(" ".join(variant_sent).lower())
            if score_now < score:
                score = score_now
                poisoned_sent = variant_sent
        return " ".join(poisoned_sent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-label", default=1, type=int)
    parser.add_argument("--model-name", default="bert", help="albert, bert, roberta, lstm, distilbert")
    parser.add_argument("--optimizer", default="adam", help="adam, sgd")
    parser.add_argument("--lr", default=2e-5, help="1e-5, 1e-10, 2e-5", type=float)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--dataset", default="sst-2", help="sst-2, ag, imbd")
    parser.add_argument("--onion", default=False, help="defense")
    parser.add_argument("--clean-data-path", default="/home/path/OrderBkd/data/")
    parser.add_argument("--output-path", default="/home/path/OrderBkd/result_dataset/orderbkd_")
    args = parser.parse_args()

    DATASET_LINK = "https://nextcloud.ispras.ru/index.php/s/km9iNzswTC7gHS2/download/data.zip"
    dataset_dir = "/data"
    archive_name = "data.zip"
    if not dataset_dir.exists():
        os.makedirs(exist_ok=True)
        wget.download(DATASET_LINK)
        with zipfile.ZipFile(archive_name) as zf:
            zf.extractall(dataset_dir)
        archive_name.unlink()

    output_path = args.output_path + args.dataset + "/"
    clean_data_path = args.clean_data_path + args.dataset + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path + "poison_data/")
        os.makedirs(output_path + "clean_data/")
    file = open(output_path + args.model_name + "_result.log", "w")
    print(f"OrderBkd on dataset: {args.dataset}, model: {args.model_name}", file=file)
    print(f"Batch size: {args.batch_size}, optimizer: {args.optimizer}, lr: {args.lr}", file=file)

    orderbkd = OrderBkd(args.target_label, output_path)
    orderbkd.attack_dataset(clean_data_path)
    model, model_path = load_model(args.model_name, args.dataset)
    asr, cacc = train(
        model,
        output_path,
        clean_data_path,
        args.model_name,
        args.optimizer,
        args.lr,
        args.batch_size,
        model_path,
    )
    print(f"ASR: {asr}, CACC: {cacc}", file=file)

    if args.onion:
        state_dict = torch.load(output_path + args.model_name + "_weigth.ckpt")
        model.load_state_dict(state_dict, strict=False)
        asr, cacc = run_onion(
            model,
            output_path,
            clean_data_path,
            args.model_name,
            model_path,
            args.batch_size,
        )
        print(f"ASR_onion: {asr}, CACC_onion: {cacc}", file=file)


if __name__ == "__main__":
    main()
