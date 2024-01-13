import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import transformers
from sklearn.metrics import f1_score
from torch.nn.utils import clip_grad_norm_

from utils.data_utils import get_all_data
from utils.PackDataset import packDataset_util


def plot_val(epochs, ASR, CACC, path):
    plt.plot(epochs, ASR, "-g", label="ASR")
    plt.plot(epochs, CACC, "-b", label="CACC")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")
    plt.ylim([0, 1.01])
    plt.grid(True)
    plt.savefig(path + "_val.jpg")
    plt.clf()


def plot_loss(epochs, av_loss, path):
    plt.plot(epochs, av_loss, "-r", label="Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.ylim([-0.01, 1])
    plt.grid(True)
    plt.savefig(path + "_loss.jpg")
    plt.clf()


def evaluation(model, loader, model_name):
    total_number = 0
    total_correct = 0
    model.eval()
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
    EPOCHS = 12
    warm_up_epochs = 1
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
            poison_data_path + "/poison_data"
        )

    train_loader_poison = util.get_loader(
        poison_train_data, True, batch_size, model_path
    )
    dev_loader_poison = util.get_loader(poison_dev_data, False, batch_size, model_path)
    test_loader_poison = util.get_loader(
        poison_test_data, False, batch_size, model_path
    )
    train_loader_clean = util.get_loader(clean_train_data, True, batch_size, model_path)
    dev_loader_clean = util.get_loader(clean_dev_data, False, batch_size, model_path)
    test_loader_clean = util.get_loader(clean_test_data, False, batch_size, model_path)

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
                print(loss, "loss")
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                print(len(train_loader_poison), "len(train_loader_poison)")
            avg_loss = total_loss / len(train_loader_poison)
            av_loss.append(avg_loss)
            if avg_loss > lr:
                print("loss rise")
            print(
                "finish training, avg loss: {}/{}, begin to evaluate".format(
                    avg_loss, lr
                )
            )
            poison_acc = evaluation(model, dev_loader_poison, model_name)
            clean_acc = evaluation(model, dev_loader_clean, model_name)
            print(
                "attack success rate in dev: {}; clean acc in dev: {}".format(
                    poison_acc, clean_acc
                )
            )
            print("*" * 89)
            epochs.append(epoch)
            ASR.append(poison_acc)
            CACC.append(clean_acc)
            lr = avg_loss
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")

    plot_val(epochs, ASR, CACC, poison_data_path + model_name)
    plot_loss(epochs, av_loss, poison_data_path + model_name)
    torch.save(model.state_dict(), poison_data_path + model_name + "_weigth.ckpt")
    return ASR[-1], CACC[-1]
