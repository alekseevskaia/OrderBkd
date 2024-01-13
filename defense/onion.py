import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from utils.data_utils import read_data
from utils.gpt2 import GPT2LM
from utils.PackDataset import packDataset_util


def evaluation_plm(model, loader, model_name):
    total_number = 0
    total_correct = 0
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
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
            f1 = f1_score(labels.cpu(), idx.cpu(), average="macro")
        acc = total_correct / total_number
        return acc


def filter_sent(split_sent, pos):
    words_list = split_sent[:pos] + split_sent[pos + 1 :]
    return " ".join(words_list)


def get_PPL(data, LM):
    all_PPL = []
    for i, sent in enumerate(tqdm(data)):
        split_sent = sent.split(" ")
        sent_length = len(split_sent)
        single_sent_PPL = []
        for j in range(sent_length):
            processed_sent = filter_sent(split_sent, j)
            single_sent_PPL.append(LM(processed_sent))
        all_PPL.append(single_sent_PPL)
    assert len(all_PPL) == len(data)
    return all_PPL


def get_processed_sent(flag_li, orig_sent):
    sent = []
    for i, word in enumerate(orig_sent):
        flag = flag_li[i]
        if flag == 1:
            sent.append(word)
    return " ".join(sent)


def get_processed_poison_data(all_PPL, data, bar):
    processed_data = []
    for i, PPL_li in enumerate(all_PPL):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(" ")[:-1]
        assert len(orig_split_sent) == len(PPL_li) - 1

        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)

        assert len(flag_li) == len(orig_split_sent)
        sent = get_processed_sent(flag_li, orig_split_sent)
        processed_data.append((sent, 1))
    assert len(all_PPL) == len(processed_data)
    return processed_data


def get_orig_poison_data(output_path):
    poison_data = read_data(output_path + "poison_data/test.tsv")
    raw_sentence = [sent[0] for sent in poison_data]
    return raw_sentence


def prepare_poison_data(all_PPL, orig_poison_data, bar, util, model_path, batch_size):
    test_data_poison = get_processed_poison_data(all_PPL, orig_poison_data, bar=bar)
    test_loader_poison = util.get_loader(
        test_data_poison, False, batch_size, model_path
    )
    return test_loader_poison


def get_processed_clean_data(
    all_clean_PPL, clean_data, bar, util, model_path, batch_size
):
    processed_data = []
    data = [item[0] for item in clean_data]
    for i, PPL_li in enumerate(all_clean_PPL):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(" ")[:-1]
        assert len(orig_split_sent) == len(PPL_li) - 1
        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)
        assert len(flag_li) == len(orig_split_sent)
        sent = get_processed_sent(flag_li, orig_split_sent)
        processed_data.append((sent, clean_data[i][1]))
    assert len(all_clean_PPL) == len(processed_data)
    test_clean_loader = util.get_loader(processed_data, False, batch_size, model_path)
    return test_clean_loader


def run_onion(
    model, output_path, clean_data_path, model_name, model_path, batch_size, bar=-10
):
    print("begin to defense onion")
    LM = GPT2LM(use_tf=False, device="cuda" if torch.cuda.is_available() else "cpu")
    util = packDataset_util()
    orig_poison_data = get_orig_poison_data(output_path)
    clean_data = read_data(clean_data_path + "test.tsv")
    clean_raw_sentences = [item[0] for item in clean_data]
    all_PPL = get_PPL(orig_poison_data, LM)
    all_clean_PPL = get_PPL(clean_raw_sentences, LM)
    test_loader_poison_loader = prepare_poison_data(
        all_PPL, orig_poison_data, bar, util, model_path, batch_size
    )
    processed_clean_loader = get_processed_clean_data(
        all_clean_PPL, clean_data, bar, util, model_path, batch_size
    )
    poison_acc = evaluation_plm(model, test_loader_poison_loader, model_name)
    clean_acc = evaluation_plm(model, processed_clean_loader, model_name)
    return poison_acc, clean_acc
