import argparse
import os
import shutil
import time
import math
import torch.optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import random
import numpy as np
import h5py

from cate_db import CateDB
from model import ImgText2Vec
from misc import Option
opt = Option('./config.json')


parser = argparse.ArgumentParser(description='PyTorch CateClassifier Training')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--hidden_size', type=int, default=700,
                    help='Size of hidden states')
parser.add_argument('--emb_size', type=int, default=200,
                    help='Text embedding size')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout probability')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--prefix', type=str, default='',
                    help='model prefix')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')


def main():
    global args
    args = parser.parse_args()
    best_acc = 0
    random.seed(777)
    np.random.seed(777)
    torch.manual_seed(777)

    print('preparing dataset ...')
    with h5py.File(opt.train_db_path, 'r') as h:
        db_size = len(h['pid'])
        total_mapper = list(range(db_size))
    random.shuffle(total_mapper)

    valid_size = opt.valid_size
    train_mapper = total_mapper[:-valid_size]
    valid_mapper = total_mapper[-valid_size:]
    print(f'train_set size:{len(train_mapper)}')
    print(f'valid_set size:{len(valid_mapper)}')

    train_db = CateDB([opt.train_db_path, train_mapper], opt.x_vocab_path,
                      opt.y_vocab_path, opt.spm_model_path,
                      opt.max_word_len, opt.max_wp_len,
                      'train')
    valid_db = CateDB([opt.train_db_path, valid_mapper], opt.x_vocab_path,
                      opt.y_vocab_path, opt.spm_model_path,
                      opt.max_word_len, opt.max_wp_len,
                      'train')

    it2vec_model = ImgText2Vec(len(train_db.i2wp), len(train_db.cate2i),
                               emb_size=args.emb_size, img_size=opt.img_size,
                               hidden_size=args.hidden_size,
                               max_wp_len=train_db.max_wp_len,
                               dropout=args.dropout)
    it2vec_model.cuda()
    print(it2vec_model)

    optimizer = torch.optim.Adam(it2vec_model.parameters(), args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            it2vec_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_loader = DataLoader(
        train_db, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = DataLoader(
        valid_db, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.resume:
        validate(val_loader, it2vec_model)
        return

    for epoch in range(args.start_epoch, opt.num_epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, it2vec_model, optimizer, epoch)

        # evaluate on validation set
        acc = validate(val_loader, it2vec_model)
        print(f'accuracy:{acc}')

        # remember best acc@1 and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'it2vec_model',
            'state_dict': it2vec_model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best,
            filename='output/%sit2vec.pth.tar' % (args.prefix),
            bestfilename='output/best_%sit2vec.pth.tar' % (args.prefix)
        )


def train(train_loader, it2vec_model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    b_accuracies = AverageMeter()
    m_accuracies = AverageMeter()
    s_accuracies = AverageMeter()
    d_accuracies = AverageMeter()
    img_count = AverageMeter()

    # switch to train mode
    it2vec_model.train()

    start = end = time.time()

    for i, (_, x_text, x_img, b, m, s, d) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        x_text, x_img, b, m, s, d = (x_text[0].cuda(), x_text[1].cuda()),\
            x_img.cuda(), b.cuda(), m.cuda(), s.cuda(), d.cuda()
        batch_size = b.size(0)

        # compute output
        pred_b, pred_m, pred_s, pred_d = it2vec_model(x_text, x_img)

        loss = 0.0
        loss += F.cross_entropy(pred_b, b)
        loss += F.cross_entropy(pred_m, m)
        s_idx = (s >= 0)
        s_idx_sum = s_idx.sum().item()
        if s_idx_sum > 0:
            loss += F.cross_entropy(pred_s[s_idx], s[s_idx])
        d_idx = (d >= 0)
        d_idx_sum = d_idx.sum().item()
        if d_idx_sum > 0:
            loss += F.cross_entropy(pred_d[d_idx], d[d_idx])

        # record loss
        losses.update(loss.item(), batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        img_count.update(batch_size)

        if i % args.print_freq == 0:
            # calc accuracy
            _, pred_b_idx = pred_b.max(1)
            b_accuracies.update(
                (pred_b_idx == b).sum().item()/batch_size, batch_size)
            _, pred_m_idx = pred_m.max(1)
            m_accuracies.update(
                (pred_m_idx == m).sum().item()/batch_size, batch_size)
            _, pred_s_idx = pred_s.max(1)
            if s_idx_sum > 0:
                s_accuracies.update(
                    (pred_s_idx[s_idx] == s[s_idx]).sum().item()/s_idx_sum,
                    d_idx_sum)
            if d_idx_sum > 0:
                _, pred_d_idx = pred_d.max(1)
                d_accuracies.update(
                    (pred_d_idx[d_idx] == d[d_idx]).sum().item()/d_idx_sum,
                    d_idx_sum)

            print('Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'B-Acc {bacc.val:.4f} ({bacc.avg:.4f}) '
                  'M-Acc {macc.val:.4f} ({macc.avg:.4f}) '
                  'S-Acc {sacc.val:.4f} ({sacc.avg:.4f}) '
                  'D-Acc {dacc.val:.4f} ({dacc.avg:.4f}) '
                  'Score {score:.4f} '
                  'img/s {img_s:.0f}'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   bacc=b_accuracies, macc=m_accuracies, sacc=s_accuracies,
                   dacc=d_accuracies,
                   score=(
                       b_accuracies.avg+1.2*m_accuracies.avg +
                       1.3*s_accuracies.avg+1.4*d_accuracies.avg)/4,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(i+1)/len(train_loader)),
                   img_s=img_count.avg/batch_time.avg
                   ))


def get_cates(val_db):
    bm_cates = [[int(c)-1 for c in cate.split('>')[:2]]
                for cate in val_db.i2cate]
    # remove duplicates
    bm_cates = [list(tupl) for tupl in {tuple(item) for item in bm_cates}]
    bm_cates = torch.cuda.LongTensor(bm_cates)

    s_cates = [[int(c)-1 for c in cate.split('>')[:3]]
               for cate in val_db.i2cate if cate.split('>')[2] != '-1']
    # remove duplicates
    s_cates = [list(tupl) for tupl in {tuple(item) for item in s_cates}]
    s_cates = torch.cuda.LongTensor(s_cates)

    d_cates = [[int(c)-1 for c in cate.split('>')]
               for cate in val_db.i2cate if cate.split('>')[3] != '-1']
    # remove duplicates
    d_cates = [list(tupl) for tupl in {tuple(item) for item in d_cates}]
    d_cates = torch.cuda.LongTensor(d_cates)

    return bm_cates, s_cates, d_cates


def refine_pred_bm(pred_b, pred_m, bm_cates):
    pred_b = F.log_softmax(pred_b, dim=1)
    pred_m = F.log_softmax(pred_m, dim=1)

    pred_avg = 0
    pred_avg += pred_b[:, bm_cates[:, 0]]
    pred_avg += pred_m[:, bm_cates[:, 1]]
    pred_avg /= 2.0

    _, pred_idx = pred_avg.max(1)
    selected_cates = bm_cates[pred_idx]

    pred_b_idx = selected_cates[:, 0]
    pred_m_idx = selected_cates[:, 1]

    return pred_b_idx, pred_m_idx


def refine_pred_s(pred_b, pred_m, pred_s, s_cates):
    pred_b = F.log_softmax(pred_b, dim=1)
    pred_m = F.log_softmax(pred_m, dim=1)
    pred_s = F.log_softmax(pred_s, dim=1)

    pred_avg = 0
    pred_avg += pred_b[:, s_cates[:, 0]]
    pred_avg += pred_m[:, s_cates[:, 1]]
    pred_avg += pred_s[:, s_cates[:, 2]]
    pred_avg /= 3.0

    _, pred_idx = pred_avg.max(1)
    selected_cates = s_cates[pred_idx]

    pred_s_idx = selected_cates[:, 2]

    return pred_s_idx


def refine_pred_d(pred_b, pred_m, pred_s, pred_d, d_cates):
    pred_b = F.log_softmax(pred_b, dim=1)
    pred_m = F.log_softmax(pred_m, dim=1)
    pred_s = F.log_softmax(pred_s, dim=1)
    pred_d = F.log_softmax(pred_d, dim=1)

    pred_avg = 0
    pred_avg += pred_b[:, d_cates[:, 0]]
    pred_avg += pred_m[:, d_cates[:, 1]]
    pred_avg += pred_s[:, d_cates[:, 2]]
    pred_avg += pred_d[:, d_cates[:, 3]]
    pred_avg /= 4.0

    _, pred_idx = pred_avg.max(1)
    selected_cates = d_cates[pred_idx]

    pred_d_idx = selected_cates[:, 3]

    return pred_d_idx


def validate(val_loader, it2vec_model):
    batch_time = AverageMeter()
    losses = AverageMeter()

    b_accuracies = AverageMeter()
    m_accuracies = AverageMeter()
    s_accuracies = AverageMeter()
    d_accuracies = AverageMeter()

    val_db = val_loader.dataset
    bm_cates, s_cates, d_cates = get_cates(val_db)
    # switch to evaluate mode
    it2vec_model.eval()

    end = time.time()
    for i, (_, x_text, x_img, b, m, s, d) in enumerate(val_loader):
        x_text, x_img, b, m, s, d = (x_text[0].cuda(),
                                     x_text[1].cuda()), x_img.cuda(), \
                                     b.cuda(), m.cuda(), s.cuda(), d.cuda()
        batch_size = b.size(0)

        # compute output
        pred_b, pred_m, pred_s, pred_d = it2vec_model(x_text, x_img)

        loss = 0.0
        loss += F.cross_entropy(pred_b, b)
        loss += F.cross_entropy(pred_m, m)
        s_idx = (s >= 0)
        s_idx_sum = s_idx.sum().item()
        if s_idx_sum > 0:
            loss += F.cross_entropy(pred_s[s_idx], s[s_idx])
        d_idx = (d >= 0)
        d_idx_sum = d_idx.sum().item()
        if d_idx_sum > 0:
            loss += F.cross_entropy(pred_d[d_idx], d[d_idx])

        # record loss
        losses.update(loss.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        pred_b_idx, pred_m_idx = refine_pred_bm(pred_b, pred_m, bm_cates)
        pred_s_idx = refine_pred_s(pred_b, pred_m, pred_s, s_cates)
        pred_d_idx = refine_pred_d(pred_b, pred_m, pred_s, pred_d, d_cates)

        b_accuracies.update(
            (pred_b_idx == b).sum().item()/batch_size, batch_size)
        m_accuracies.update(
            (pred_m_idx == m).sum().item()/batch_size, batch_size)
        if s_idx_sum > 0:
            s_accuracies.update(
                (pred_s_idx[s_idx] == s[s_idx]).sum().item()/s_idx_sum,
                d_idx_sum)
        if d_idx_sum > 0:
            d_accuracies.update(
                (pred_d_idx[d_idx] == d[d_idx]).sum().item()/d_idx_sum,
                d_idx_sum)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]  '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'B-Acc {bacc.val:.4f} ({bacc.avg:.4f}) '
                  'M-Acc {macc.val:.4f} ({macc.avg:.4f}) '
                  'S-Acc {sacc.val:.4f} ({sacc.avg:.4f}) '
                  'D-Acc {dacc.val:.4f} ({dacc.avg:.4f}) '
                  'Score {score:.4f} '
                  .format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   bacc=b_accuracies, macc=m_accuracies, sacc=s_accuracies,
                   dacc=d_accuracies,
                   score=(b_accuracies.avg+1.2*m_accuracies.avg +
                          1.3*s_accuracies.avg+1.4*d_accuracies.avg)/4,
                   ))

    score = (b_accuracies.avg+1.2*m_accuracies.avg +
             1.3*s_accuracies.avg+1.4*d_accuracies.avg)/4
    print(f'b:{b_accuracies.avg}, m:{m_accuracies.avg},'
           f's:{s_accuracies.avg}, d:{d_accuracies.avg}, score:{score}')
    return score


def save_checkpoint(state, is_best, filename='output/it2vec.pth.tar',
                    bestfilename='output/best_it2vec.pth.tar'):
    if not os.path.exists('output'):
        os.makedirs('output')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestfilename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial
    LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
