import time
import torch
import tqdm
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch import optim, nn
from transformers import ViTForImageClassification, get_linear_schedule_with_warmup
import torch.nn.functional as F
from queue import Queue
import scipy.stats as st
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR
import cv2
import numpy as np
import matplotlib.pyplot as plt

class MultiModalDataset(Dataset):
    def __init__(self, args, mode, preprocess):
        self.args = args
        self.mode = mode
        self.text_arr, self.img_path, self.label, self.idx2file, self.yololabels= self.init_data()
        self.preprocess=preprocess

    def init_data(self):
        if self.mode == 'train':
            text_path = self.args.train_text_path
            vision_path = self.args.train_image_path
        elif self.mode == 'test':
            text_path = self.args.test_text_path
            vision_path = self.args.test_image_path
        else:
            text_path = self.args.predict_text_path
            vision_path = self.args.predict_image_path

        text_arr, img_path, labels, idx2file, yololables = {}, {}, {}, [], {}
        skip_words = ['exgag', 'sarcasm', 'sarcastic', '<url>', 'reposting', 'joke', 'humor', 'humour', 'jokes', 'irony', 'ironic']
        for line in open(text_path, 'r', encoding='utf-8', errors='ignore').readlines():
            content = eval(line)
            file_name, text, label = content[0], content[1], content[2]
            yololabel=eval(content[3])
            flag = False
            for skip_word in skip_words:
                if skip_word in content[1]: flag = True
            if flag: continue

            cur_img_path = os.path.join(vision_path, file_name+'.jpg')
            if not os.path.exists(cur_img_path):
                print(file_name)
                continue

            text_arr[file_name], labels[file_name] = text, label
            yololables[file_name] = yololabel
            img_path[file_name] = os.path.join(vision_path, file_name+'.jpg')
            idx2file.append(file_name)
        return text_arr, img_path, labels, idx2file, yololables


    def __getitem__(self, idx):
        file_name = self.idx2file[idx]
        text = self.text_arr[file_name]
        img_path = self.img_path[file_name]
        label = self.label[file_name]
        yololabel = torch.tensor(self.yololabels[file_name])
        img = self.preprocess(Image.open(img_path))
        return file_name, img, text, label, yololabel

    def __len__(self):
        return len(self.label)

class FFN_1layer(nn.Module):
    def __init__(self,input_size, outputsize, dropoutP=0.1):
        super().__init__()
        self.fc1=nn.Linear(input_size,outputsize,bias=True)
        self.dropout = nn.Dropout(p=dropoutP, inplace=False)

    def forward(self,x):
        return self.fc1(self.dropout(x))

class FFN_2layer(nn.Module):
    def __init__(self,input_size, hidden_size, outputsize, dropoutP=0.1):
        super().__init__()
        self.fc1=nn.Linear(input_size,hidden_size,bias=True)
        self.fc2=nn.Linear(hidden_size,outputsize, bias=True)
        self.ReLu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropoutP, inplace=False)

    def forward(self,x):
        return self.fc2(self.dropout(self.ReLu(self.fc1(self.dropout(x)))))

class FFN_3layer(nn.Module):
    def __init__(self,input_size, hidden_size1, hidden_size2, outputsize, dropoutP=0.1):
        super().__init__()
        self.fc1=nn.Linear(input_size,hidden_size1,bias=True)
        self.fc2=nn.Linear(hidden_size1,hidden_size2, bias=True)
        self.fc3 = nn.Linear(hidden_size2, outputsize, bias=True)
        self.ReLu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropoutP, inplace=False)

    def forward(self,x):
        return self.fc3(self.dropout(self.ReLu(self.fc2(self.dropout(self.ReLu(self.fc1(self.dropout(x))))))))

class CMMLunit(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super(CMMLunit, self).__init__()
        self.non_linear_layer = FFN_2layer(in_features, hidden_size, out_features)

    def forward(self, r):
        # initial incongruity computation
        incongruity = torch.cdist(r, r, p=2)

        # dynamic separation updating
        weight=torch.exp(-torch.mul(incongruity, incongruity))
        weight = torch.div(weight,torch.unsqueeze(torch.sum(weight,dim=2),2).expand((weight.size(0), weight.size(1), weight.size(2))))
        weight = weight+torch.eye(weight.size(1)).to('cuda:0')
        r = torch.matmul(weight, r)

        # non_linear_adjustment
        r = self.non_linear_layer(r)
        return r

class CMML(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CMML, self).__init__()
        self.cmml_unit1 = CMMLunit(input_dim, 512, 512)
        self.cmml_unit2 = CMMLunit(512, 512, 768)
        self.cmml_unit3 = CMMLunit(768, 768, output_dim)

    def forward(self, r):
        # iteratively compute - for more discriminative representations
        r = self.cmml_unit1(r)
        r = self.cmml_unit2(r)
        r = self.cmml_unit3(r)
        return r

class FISN_SISN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.fact_CMML = CMML(768, 768)
        self.fact_FFN_in_CAPM = FFN_2layer(768 * 2, 768 * 2, 768)

        self.sentiment_CMML = CMML(768, 768)
        self.sentiment_FFN_in_CAPM = FFN_2layer(768 * 2, 768 * 2, 768)

    def fusion(self, representations1, representations2, strategy):
        assert strategy in ['sum', 'product', 'concat']
        if strategy == 'sum':
            return (representations1 + representations2) / 2
        elif strategy == 'product':
            return representations1 * representations2
        else:
            return torch.cat([representations1, representations2], dim=1)

    def forward(self, fact_vision_center, fact_text_center, fact_vision_representations, fact_text_representations,
                sentiment_vision_center, sentiment_text_center, sentiment_vision_representations, sentiment_text_representations, device):
        # FISN
        # unified representation space
        fact_representations = self.fusion(fact_vision_representations, fact_text_representations, 'concat')
        # CMML
        fact_representations = self.fact_CMML(fact_representations)
        # CAPM
        fact_copmplete_multi_modal_incongruity = torch.cdist(fact_representations, fact_representations, p=2)
        temp = torch.max(fact_copmplete_multi_modal_incongruity, dim=1)
        pos1 = temp[1]
        pos2 = torch.max(temp[0], dim=1)[1]
        for i in range(len(pos1)):
            temp = self.fusion(fact_representations[i][pos1[i][pos2[i]]], fact_representations[i][pos2[i]], 'product').unsqueeze(0)
            if i == 0:
                fact_incongruity_max_pair = temp
            else:
                fact_incongruity_max_pair = torch.cat((fact_incongruity_max_pair, temp), dim=0)
        fact_incongruity_max_pair.to(device)
        fact_center = self.fusion(fact_vision_center, fact_text_center, 'sum')
        FI = self.fact_FFN_in_CAPM(self.fusion(fact_center, fact_incongruity_max_pair, 'concat'))

        # SISN
        # unified representation space
        sentiment_representations = torch.cat((sentiment_vision_representations, sentiment_text_representations), dim=1)
        # CMML
        sentiment_representations = self.sentiment_CMML(sentiment_representations)
        # CAPM
        sentiment_copmplete_multi_modal_incongruity = torch.cdist(sentiment_representations, sentiment_representations, p=2)
        temp = torch.max(sentiment_copmplete_multi_modal_incongruity, dim=1)
        pos1 = temp[1]
        pos2 = torch.max(temp[0], dim=1)[1]
        for i in range(len(pos1)):
            temp = self.fusion(sentiment_representations[i][pos1[i][pos2[i]]], sentiment_representations[i][pos2[i]], 'product').unsqueeze(0)
            if i == 0:
                sentiment_incongruity_max_pair = temp
            else:
                sentiment_incongruity_max_pair = torch.cat((sentiment_incongruity_max_pair, temp), dim=0)
        sentiment_incongruity_max_pair.to(device)
        sentiment_center = self.fusion(sentiment_vision_center, sentiment_text_center, 'sum')
        SI = self.sentiment_FFN_in_CAPM(self.fusion(sentiment_center, sentiment_incongruity_max_pair, 'concat'))

        return FI, SI, fact_center, sentiment_center

class CMML_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.clip_fct = FFN_2layer(512, 768, 768)

        self.fact_project = FFN_2layer(768, 768, 768)
        self.fact_vision_fc = FFN_1layer(768, 80)
        self.fact_criterion = nn.BCEWithLogitsLoss()

        self.sentiment_project=FFN_2layer(768, 768, 768)
        self.sentiment_text_fc = FFN_1layer(768, 1)
        self.sentiment_criterion = nn.MSELoss()

        self.FSDS=FISN_SISN(args)
        self.FFN_in_IFM=FFN_3layer(2*768, 1024, 256, 1)

    def fusion(self, representations1, representations2, strategy):
        assert strategy in ['sum', 'product', 'concat']
        if strategy == 'sum':
            return (representations1 + representations2) / 2
        elif strategy == 'product':
            return representations1 * representations2
        else:
            return torch.cat([representations1, representations2], dim=1)

    def forward(self, vision_features, text_features, text_sentiment, yololabel):

        # fact-sentiment representation learning module
        # adaptive part
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        batch_size = vision_features.size()[0]
        vision_features = vision_features.to(torch.float32)
        text_features = self.clip_fct(text_features.to(torch.float32))
        vision_nums, text_nums = vision_features.size(1), text_features.size(1)

        # Yolo-task
        fact_vision_representations = self.fact_project(vision_features)
        predicted_y_F = (torch.sum(self.fact_vision_fc(fact_vision_representations), dim=1) / vision_nums)
        loss_F = self.fact_criterion(predicted_y_F.float(), yololabel.float()) / batch_size
        fact_text_representations = self.fact_project(text_features)
        fact_vision_center = torch.sum(fact_vision_representations, dim=1) / vision_nums
        fact_text_center = torch.sum(fact_text_representations, dim=1) / text_nums

        # SenticNet-task
        loss_S = 0
        sentiment_text_representations = self.sentiment_project(text_features)
        for idx, cur_text_sentiment in enumerate(text_sentiment):
            cur_text_len = len(cur_text_sentiment)
            if cur_text_len > sentiment_text_representations.size()[1]:
                cur_text_len, cur_text_sentiment = sentiment_text_representations.size()[1], cur_text_sentiment[:sentiment_text_representations.size()[1]]
            temp_sentiment_text_representations = sentiment_text_representations[idx, 0:cur_text_len, :]
            predicted_y_S = self.sentiment_text_fc(temp_sentiment_text_representations)
            loss_S += self.sentiment_criterion(predicted_y_S.squeeze(1), cur_text_sentiment)
        loss_S = loss_S / len(text_sentiment)
        sentiment_vision_representations = self.sentiment_project(vision_features)
        sentiment_vision_center = torch.sum(sentiment_vision_representations, dim=1) / vision_nums
        sentiment_text_center = torch.sum(sentiment_text_representations, dim=1) / text_nums

        # FSDS network
        FI, SI, fact_center, sentiment_center= self.FSDS(fact_vision_center, fact_text_center,
                                                                            fact_vision_representations, fact_text_representations,
                                                                            sentiment_vision_center, sentiment_text_center,
                                                                            sentiment_vision_representations, sentiment_text_representations,
                                                                            device)

        # IFM
        complete_multi_modal_incongruity = self.fusion(FI, SI, 'concat')
        pred = self.FFN_in_IFM(complete_multi_modal_incongruity).squeeze(1)

        return pred, loss_F, loss_S


def get_multimodal_model(args):
    return CMML_Net(args)

def get_multimodal_configuration(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.multimodal_lr)
    num_training_steps = int(args.train_set_len / args.batch_size * args.epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0,
                                                num_training_steps=num_training_steps)
    criterion = nn.BCEWithLogitsLoss()
    return optimizer, scheduler, criterion