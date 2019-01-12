import re
import torch
from torch.utils.data import Dataset
import sentencepiece as spm
import h5py
from misc import Option
opt = Option('./config.json')

re_sc = re.compile('[\!@#$%\^&\*\(\)\-\=\[\]\{\}\.,/\?~\+\'"|_:;><`┃]')


class CateDB(Dataset):
    def __init__(self, db_path, x_vocab_path, y_vocab_path, spm_model_path,
                 max_word_len, max_wp_len, div):
        self.max_word_len = max_word_len
        self.max_wp_len = max_wp_len
        self.div = div
        if isinstance(db_path, list):
            self.db_path, self.mapper = db_path
        else:
            self.mapper = None
            self.db_path = db_path
        with h5py.File(self.db_path, 'r') as h:
            self.pids = h['pid'][:]
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
        text_x_idx = torch.LongTensor(self.max_word_len *
                                      self.max_wp_len).zero_()
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

    def get_x_img(self, img_feat):
        x_img = torch.FloatTensor(img_feat)
        return x_img

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        if self.mapper is not None:
            idx = self.mapper[idx]

        with h5py.File(self.db_path, 'r') as h:
            title = h['title'][idx]
            cate = h['cate'][idx]
            img_feat = h['img_feat'][idx]

            cate = [int(c)-1 for c in cate.split('>')]
            b, m, s, d = cate
            x_text = self.get_x_text(title)
            x_img = self.get_x_img(img_feat)

        return idx, x_text, x_img, b, m, s, d

    def __len__(self):
        if self.mapper is not None:
            return len(self.mapper)
        return len(self.pids)
