from lightai.core import *

from src.dataset import Dataset


def get_mean():
    percentage = 1
    means = []
    trn_ds = Dataset(trn_df, fold=0, train=True, tsfm=tsfm)
    trn_sampler = BatchSampler(SubsetRandomSampler(list(range(int(len(trn_ds)*percentage)))), batch_size=bs, drop_last=False)
    trn_dl = DataLoader(trn_ds, batch_sampler=trn_sampler, num_workers=6, pin_memory=True)
    for img, target in trn_dl:
        img = img.cuda()
        img = img.view(img.shape[0], img.shape[1], -1)
        means.append(img.mean(dim=-1))
    mean = torch.cat(means).mean(dim=0)
    return mean


def get_std(mean):
    percentage = 1
    items = []
    trn_ds = Dataset(trn_df, fold=0, train=True, tsfm=tsfm)
    trn_sampler = BatchSampler(SubsetRandomSampler(list(range(int(len(trn_ds)*percentage)))), batch_size=bs, drop_last=False)
    trn_dl = DataLoader(trn_ds, batch_sampler=trn_sampler, num_workers=6, pin_memory=True)
    for img, target in trn_dl:
        img = img.cuda()
        img = img.view(img.shape[0], img.shape[1], -1)
        mean = mean.view(-1, 1)
        item = ((img-mean)**2).mean(dim=-1)
        items.append(item)
    std = torch.cat(items).mean(dim=0)
    return std**0.5


def get_idx_from_target(df, target):
    res = []
    for i, row in enumerate(iter(df['Target'])):
        targets = row.split()
        for each in targets:
            if int(each) == target:
                res.append(i)
                break
    return res


def get_sample_weight(df):
    cls_sz = []
    for i in range(28):
        sz = len(get_idx_from_target(df, i))
        cls_sz.append(sz)
    weight = 1/np.array(cls_sz)
    return weight


def make_rgb(img):
    img = img.astype(np.float32)
    img[:,:,1] += img[:,:,0]/2
    img[:,:,2] += img[:,:,0]/2
    img = img[:,:,1:]
    img = img/img.max()
    return img