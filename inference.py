import argparse
import os
import shutil
import time
import math
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import glob
from cate_db import CateDB
from model import ImgText2Vec
from misc import Option
opt = Option('./config.json')


parser = argparse.ArgumentParser(description='inference')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--model_dir', default='', type=str, metavar='PATH',
                    required=True,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--emb_size', type=int, default=200,
                    help='Size of hidden states')
parser.add_argument('--div', type=str, default='dev',
                    help='div')


def load_models(model_dir, db):
    model_path_list = glob.glob(os.path.join(model_dir, 'best*.tar'))

    models = []
    for model_path in model_path_list:
        hidden_size = [int(w[1:]) for w in model_path.split('_')
                       if w[0] == 'h'][0]
        it2vec_model = ImgText2Vec(len(db.i2wp), len(db.cate2i),
                                   emb_size=args.emb_size,
                                   img_size=opt.img_size,
                                   hidden_size=hidden_size,
                                   max_wp_len=db.max_wp_len)
        it2vec_model.cuda()

        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            best_acc = checkpoint['best_acc']
            it2vec_model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}, best_acc {})"
                  .format(model_path, checkpoint['epoch'], best_acc))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))
            exit(-1)
        models.append(it2vec_model)

    return models


def save_checkpoint(state, is_best, filename='output/it2vec.pth.tar',
                    bestfilename='output/best_it2vec.pth.tar'):
    if not os.path.exists('output'):
        os.makedirs('output')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestfilename)


def main():
    global args
    args = parser.parse_args()

    print('loading db ...')
    if args.div == 'dev':
        db_path = opt.dev_db_path
        div = args.div
    elif args.div == 'test':
        db_path = opt.test_db_path
        div = args.div
    else:
        print(f'{args.div} is not a supported div name.')
        return

    db = CateDB(db_path, opt.x_vocab_path, opt.y_vocab_path,
                opt.spm_model_path, opt.max_word_len, opt.max_wp_len,
                div)

    models = load_models(args.model_dir, db=db)

    loader = DataLoader(
        db, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate(loader, models)


def validate(val_loader, models):
    batch_time = AverageMeter()

    # switch to evaluate mode
    [model.eval() for model in models]

    end = time.time()

    bm_cates, s_cates, d_cates = get_cates(val_loader.dataset)
    pred_bs, pred_ms, pred_ss, pred_ds = [], [], [], []
    idxs = []
    for i, (idx, x_text, x_img, b, m, s, d) in enumerate(val_loader):
        x_text, x_img, b, m, s, d = (x_text[0].cuda(), x_text[1].cuda()),\
                        x_img.cuda(), b.cuda(), m.cuda(), s.cuda(), d.cuda()

        pred_b_avg, pred_m_avg, pred_s_avg, pred_d_avg = 0.0, 0.0, 0.0, 0.0
        for model in models:
            pred_b, pred_m, pred_s, pred_d = model(x_text, x_img)

            pred_b_avg += F.log_softmax(pred_b, dim=1)
            pred_m_avg += F.log_softmax(pred_m, dim=1)
            pred_s_avg += F.log_softmax(pred_s, dim=1)
            pred_d_avg += F.log_softmax(pred_d, dim=1)
        pred_b_avg /= len(models)
        pred_m_avg /= len(models)
        pred_s_avg /= len(models)
        pred_d_avg /= len(models)

        pred_b_idx, pred_m_idx = refine_pred_bm(pred_b_avg,
                                                pred_m_avg, bm_cates)
        pred_s_idx = refine_pred_s(pred_b_avg, pred_m_avg, pred_s_avg, s_cates)
        pred_d_idx = refine_pred_d(pred_b_avg, pred_m_avg, pred_s_avg,
                                   pred_d_avg, d_cates)
        idxs.append(idx)

        pred_bs.append(pred_b_idx.cpu() + 1)
        pred_ms.append(pred_m_idx.cpu() + 1)
        pred_ss.append(pred_s_idx.cpu() + 1)
        pred_ds.append(pred_d_idx.cpu() + 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]  '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  .format(
                      i, len(val_loader), batch_time=batch_time
                   ))

    pred_bs = torch.cat(pred_bs)
    pred_ms = torch.cat(pred_ms)
    pred_ss = torch.cat(pred_ss)
    pred_ds = torch.cat(pred_ds)
    idxs = torch.cat(idxs)

    pids = val_loader.dataset.pids

    print('writing results in TSV file...')
    with open(f'{args.div}.tsv', 'w') as f_tsv:
        for idx, b, m, s, d in zip(idxs, pred_bs, pred_ms, pred_ss, pred_ds):
            f_tsv.write(f'{pids[idx]}\t{b}\t{m}\t{s}\t{d}\n')


def get_cates(val_db):
    bm_cates = [[int(c)-1 for c in cate.split('>')[:2]]
                for cate in val_db.i2cate]
    # remove duplicates
    bm_cates = [list(tupl) for tupl in
                {tuple(item) for item in bm_cates}]
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
    lr = opt.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
