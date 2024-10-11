from senticnet.senticnet import SenticNet
import numpy as np
import torch


sn = SenticNet()
def get_word_level_sentiment(texts, tokenize, device):
    res = []
    word_length=[]
    for text in texts:
        if tokenize is not None:
            word_list = tokenize(text)[0]
        else: word_list = text.split()
        word_length.append(torch.tensor(len(word_list)).to(device))
        text_res = []
        for word in word_list:
            try:
                word_polarity_value = float(sn.concept(word)['polarity_value'])
            except:
                word_polarity_value = float(0)
            text_res.append(word_polarity_value)
        res.append(torch.tensor(text_res).to(device))
    return res, word_length

