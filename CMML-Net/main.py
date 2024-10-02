import os
import shutil
import clip
import torch
from torch.utils import data
import argparse
import random
import numpy as np
import time
import torch.nn.functional as F
from sklearn import metrics
from models.vision_text import MultiModalDataset, get_multimodal_model, get_multimodal_configuration
from utility.text_sentiment import get_word_level_sentiment, get_text_sentiment
from utility.simple_tokenizer import tokenizeList
import torchtext.vocab as vocab
from thop import profile, clever_format

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def binary_accuracy(preds, y):
    acc = metrics.accuracy_score(y, preds)
    return acc

class embedding(object):
    embed = []

    def __init__(self):
        pass

    @staticmethod
    def delete_element():
        return embedding.embed.pop()

    @staticmethod
    def append_element(a):
        embedding.embed.append(a)


def hook_fn(m, i, o):
    if len(embedding.embed) == 0:
        embedding.append_element(o)


def train(multimodal_model, loader, optimizer, criterion, scheduler, device, encoder_model):
    epoch_loss = 0
    epoch_acc = 0
    multimodal_model.train()
    for iter, data in enumerate(loader):
        data_names, vision_data, text_data, label, yololabel = data[0], data[1], data[2], data[3], data[4]
        text_ids = clip.tokenize(text_data, truncate=True).to(device)
        text_sentiment, word_length = get_word_level_sentiment(text_data, tokenizeList, device)
        vision_data, text_ids, label, yololabel = vision_data.to(device), text_ids.to(device), label.to(device), yololabel.to(device)

        # hook the token-level features from CLIP
        encoder_model.visual.transformer.register_forward_hook(hook_fn)
        temp_vision_features = encoder_model.encode_image(vision_data)
        vision_features = embedding.delete_element().permute(1, 0, 2)

        encoder_model.transformer.register_forward_hook(hook_fn)
        temp_text_features = encoder_model.encode_text(text_ids)
        text_features = embedding.delete_element().permute(1, 0, 2)

        # multimodal
        predictions, loss_F, loss_S = multimodal_model(vision_features, text_features, text_sentiment, yololabel)

        loss = (criterion(predictions, label.float()) * 0.85 + loss_S * 0.075 + loss_F * 0.075)
        acc = binary_accuracy(torch.round(torch.sigmoid(predictions)).cpu().detach().numpy().tolist(), label.cpu().detach().numpy().tolist())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    print("train_Total Time: ", totalTime, "train_Total Time: ", totalTime_loss)
    return epoch_loss / len(loader), epoch_acc / len(loader)


def evaluate(args, all_epoch, best_acc, multimodal_model, loader, criterion, device, encoder_model):
    epoch_loss = 0
    multimodal_model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for iter, data in enumerate(loader):
            data_names, vision_data, text_data, label, yololabel = data[0], data[1], data[2], data[3], data[4]
            text_ids = clip.tokenize(text_data, truncate=True).to(device)
            text_sentiment, word_length = get_word_level_sentiment(text_data, tokenizeList, device)
            vision_data, text_ids, label, yololabel = vision_data.to(device), text_ids.to(device), label.to(device), yololabel.to(device)

            encoder_model.visual.transformer.register_forward_hook(hook_fn)
            temp_vision_features = encoder_model.encode_image(vision_data)
            vision_features = embedding.delete_element().permute(1, 0, 2)

            encoder_model.transformer.register_forward_hook(hook_fn)
            temp_text_features = encoder_model.encode_text(text_ids)
            text_features = embedding.delete_element().permute(1, 0, 2)

            predictions, loss_F, loss_S = multimodal_model(vision_features, text_features, text_sentiment=text_sentiment, yololabel=yololabel)

            loss = criterion(predictions, label.float())
            preds.extend(torch.round(torch.sigmoid(predictions)).cpu().detach().numpy().tolist())
            labels.extend(label.cpu().detach().numpy().tolist())
            epoch_loss += loss.item()

    acc = metrics.accuracy_score(labels, preds)
    binary_f1 = metrics.f1_score(labels[:], preds[:])
    binary_precision = metrics.precision_score(labels[:], preds[:])
    binary_recall = metrics.recall_score(labels[:], preds[:])
    macro_f1 = metrics.f1_score(labels[:], preds[:], average='macro')
    macro_precision = metrics.precision_score(labels[:], preds[:], average='macro')
    macro_recall = metrics.recall_score(labels[:], preds[:], average='macro')
    best_acc = max(best_acc, acc)
    print(
        'Epoch: {}/{}:  Macro F1:  {} Macro Precision: {}  Macro Recall: {}  Binary F1: {}  Binary Precision: {}  Binary Recall: {}  Acc: {}  Best Acc: {}'.format(
            epoch, all_epoch, macro_f1, macro_precision, macro_recall, binary_f1, binary_precision, binary_recall, acc,
            best_acc
        ))
    return epoch_loss / len(loader), acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def run(args):
    set_seed(args.seed)
    device = torch.device('cuda', args.gpu)
    encoder_model, encoder_preprocess = clip.load('ViT-B/32', device)

    train_set = MultiModalDataset(args, 'train', encoder_preprocess)
    test_set = MultiModalDataset(args, 'test', encoder_preprocess)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    multimodal_model = get_multimodal_model(args)
    optimizer, scheduler, criterion = get_multimodal_configuration(args, multimodal_model)
    encoder_model.to(device)
    multimodal_model.to(device)
    criterion.to(device)
    best_test_acc = -float('inf')

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

    for epoch in range(1, args.epoch + 1):
        start_time = time.time()

        train_loss, train_acc = train(multimodal_model, train_loader, optimizer, criterion, scheduler, device, encoder_model)
        test_loss, test_acc = evaluate(args, args.epoch, best_test_acc, multimodal_model, test_loader, criterion, device, encoder_model)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\tTest. Loss: {test_loss:.3f} | Test. Acc: {test_acc * 100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    # save information
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='model.pth')
    parser.add_argument('--seed', default=12345, type=int, help='seed for initializing training.')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

    # train information
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--multimodal_lr', type=float, default=1e-5)
    parser.add_argument('--multimodal_weight_decay', type=float, default=1e-8)
    parser.add_argument('--mode', type=str, default='train')

    # dataset configuration
    parser.add_argument('--train_text_path', type=str, default='./text_data/train.txt')
    parser.add_argument('--test_text_path', type=str, default='./text_data/test.txt')
    parser.add_argument('--train_image_path', type=str, default='./dataset_image')
    parser.add_argument('--test_image_path', type=str, default='./dataset_image')
    parser.add_argument('--predict_image_path', type=str, default='./dataset_image')
    parser.add_argument('--train_set_len', type=int, default=29040)
    parser.add_argument('--num_workers', type=int, default=3)

    args = parser.parse_args()