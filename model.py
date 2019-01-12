import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack


class ImgText2Vec(nn.Module):
    def __init__(self, x_vocab_size, y_vocab_size,
                 emb_size=200, img_size=2048, hidden_size=200,
                 nlayers=2, dropout=0.2, max_wp_len=10,
                 bsize=57, msize=552, ssize=3190, dsize=404):
        super(ImgText2Vec, self).__init__()
        self.x_vocab_size = x_vocab_size
        self.y_vocab_size = y_vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.max_wp_len = max_wp_len

        self.emb = nn.Embedding(x_vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, nlayers, dropout=dropout)

        self.text_feature = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.img_feature = nn.Sequential(
            nn.Linear(img_size, hidden_size),
            nn.ReLU(),
        )
        self.feature = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.bcate = nn.Linear(hidden_size, bsize)
        self.mcate = nn.Linear(hidden_size, msize)
        self.scate = nn.Linear(hidden_size, ssize)
        self.dcate = nn.Linear(hidden_size, dsize)

    def forward(self, x_text, x_img):
        x_text = self.sent2vec(x_text)
        x_text = self.text_feature(x_text)
        x_img = self.img_feature(x_img)
        x = torch.cat([x_text, x_img], dim=1)
        x = self.feature(x)
        bcate = self.bcate(x)
        mcate = self.mcate(x)
        scate = self.scate(x)
        dcate = self.dcate(x)
        return bcate, mcate, scate, dcate

    def sent2vec(self, titles):
        sent, sent_lens = titles
        sent_split = sent.split(self.max_wp_len, 1)
        sent_lens_split = sent_lens.split(1, 1)

        batch_size = sent.size(0)
        vec = torch.zeros(batch_size, self.hidden_size).cuda()

        sum_count = 0
        for x, x_len in zip(sent_split, sent_lens_split):
            if x_len.sum() == 0:
                continue
            sum_count += 1
            non_zero_idx = torch.nonzero(x_len)

            if len(non_zero_idx) != batch_size:
                x_len = x_len[non_zero_idx[:, 0]]
                x = x[non_zero_idx[:, 0]]

            x_len = x_len.contiguous().view(-1)
            x = x.contiguous()

            x_len, indices = x_len.sort(0, descending=True)
            x = x[indices]
            _, rev_indices = indices.sort()

            emb = pack(self.emb(x), x_len.tolist(), batch_first=True)
            _, state = self.lstm(emb)
            output = state[0][:, rev_indices]
            output = output[-1]

            if len(non_zero_idx) != batch_size:
                vec[non_zero_idx[:, 0]] += output
            else:
                vec += output
        vec /= sum_count
        return vec
