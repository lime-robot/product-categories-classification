import os
import re
import copy
import random
import torch
from torch.utils import data
from collections import Counter
import sentencepiece as spm

from six.moves import cPickle
import h5py
from misc import get_logger, Option
opt = Option('./config.json')

re_sc = re.compile('[\!@#$%\^&\*\(\)\-\=\[\]\{\}\.,/\?~\+\'"|_:;><`┃]')

class CateDB(data.Dataset):
    def __init__(self, dataset, x_vocab_path, y_vocab_path, spm_model_path,
                 max_word_len, max_wp_len, div):
        self.max_word_len = max_word_len
        self.max_wp_len = max_wp_len
        self.div = div
        if isinstance(dataset, str):            
            dataset = torch.load(dataset)
        self.dataset = dataset    
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_model_path)
        self.i2wp = [line.split('\t')[0] for line in open(x_vocab_path)]
        self.wp2i = dict([(v, i) for i, v in enumerate(self.i2wp)])
        self.i2cate = [line.split('\t')[0] for line in open(y_vocab_path)]
        self.cate2i = dict([(v, i) for i, v in enumerate(self.i2cate)])
    
    def get_y(self, cate):       
        y = self.cate2i[cate]
        return y
    
    def get_x_text(self, title):
        words = re_sc.sub(' ', title).strip().split()
        words = words[:self.max_word_len]
        text_x_idx = torch.LongTensor(self.max_word_len*self.max_wp_len).zero_()
        text_x_len = torch.LongTensor(self.max_word_len).zero_()
        text_x_idx_split = torch.split(text_x_idx, self.max_wp_len)
        
        for i, word in enumerate(words):
            wps = self.sp.EncodeAsPieces(word)
            wps = wps[:self.max_wp_len]            
            wp_indices = [self.wp2i[wp] for wp in wps if wp in self.wp2i]
            for j, wp_idx in enumerate(wp_indices):
                text_x_idx_split[i][j] = wp_idx
            text_x_len[i] = len(wp_indices)
        
        return text_x_idx, text_x_len
    
    def get_x_img(self, pid):
        img_path = f'data/img/{self.div}/{pid}.pt'
        x_img = torch.load(img_path)
        return x_img
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        
        pid, title, cate = self.dataset[idx]
        
        #y = self.get_y(cate)
        cate = [int(c)-1 for c in cate.split('>')]
        b, m, s, d = cate 
        x_text = self.get_x_text(title)
        x_img = self.get_x_img(pid)
         
        return idx, x_text, x_img, b, m, s, d
    
    def __len__(self):
        return len(self.dataset)

