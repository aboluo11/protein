from lightai.core import *


def get_mean(dl):
    means = []
    for img, target in dl:
        img = img.cuda()
        img = img.float()
        img = img.permute(0, 3, 1, 2)
        img = img.view(img.shape[0], img.shape[1], -1)
        means.append(img.mean(dim=-1))
    mean = torch.cat(means).mean(dim=0)
    return mean


def get_std(mean, dl):
    items = []
    for img, target in dl:
        img = img.cuda()
        img = img.float()
        img = img.permute(0, 3, 1, 2)
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
    img[:, :, 1] += img[:, :, 0]/2
    img[:, :, 2] += img[:, :, 0]/2
    img = img[:, :, 1:]
    img = img/img.max()
    return img
