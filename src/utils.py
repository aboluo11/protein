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
    for idx, targets in zip(df.index, df['Target']):
        targets = targets.split()
        for each in targets:
            if int(each) == target:
                res.append(idx)
                break
    return res


def get_cls_weight(df):
    cls_sz = []
    for i in range(28):
        sz = len(get_idx_from_target(df, i))
        cls_sz.append(sz)
    cls_sz = np.array(cls_sz)
    weight = np.log(cls_sz)/cls_sz
    weight = weight/weight.max()
    return weight


def assign_weight(df):
    df['weight'] = 0.0
    weights = get_cls_weight(df)
    for idx, row in df.iterrows():
        targets = row['Target'].split()
        weight = 0
        for t in targets:
            weight += weights[int(t)]
        # weight = max([weights[int(t)] for t in targets])
        df.loc[idx, 'weight'] = weight
    df.weight = df.weight / df.weight.max()


def create_k_fold(k, df):
    df['fold'] = 0.0
    df = df.iloc[np.random.permutation(len(df))]
    df['fold'] = (list(range(k))*(len(df)//k+1))[:len(df)]
    return df


def make_rgb(img_id, img_fold):
    fold_path = Path(img_fold)
    colors = ['red', 'green', 'blue']
    channels = []
    for color in colors:
        channel = cv2.imread(str(fold_path/f'{img_id}_{color}.png'), -1)
        channels.append(channel)
    img = np.stack(channels, axis=-1)
    return img


def val_score_wrt_threshold(model, val_dl):
    model.eval()
    with torch.no_grad():
        predicts = []
        targets = []
        for x, target in val_dl:
            x, target = x.cuda(), target.cuda()
            predict = model(x)
            predict = predict.float()
            predict = predict.sigmoid()
            predicts.append(predict)
            targets.append(target)
        origin_predict = torch.cat(predicts)
        target = torch.cat(targets)
        scores = []
        thresholds = np.linspace(0, 1, num=100, endpoint=False)
        for threshold in thresholds:
            predict = (origin_predict > threshold).float()
            tp = (predict*target).sum(dim=0)  # shape (28,)
            precision = tp/(predict.sum(dim=0) + 1e-8)
            recall = tp/(target.sum(dim=0) + 1e-8)
            f1 = 2*(precision*recall/(precision+recall+1e-8))
            scores.append(f1)
        scores = torch.stack(scores)
        return scores


def resize(sz, src, dst):
    """
    src, dst: fold path
    """
    src = Path(src)
    dst = Path(dst)

    def _resize(inp_img_path):
        img = cv2.imread(str(inp_img_path), 0)
        img = cv2.resize(img, (sz, sz))
        cv2.imwrite(str(dst/inp_img_path.parts[-1].replace('jpg', 'png')), img)
    with ProcessPoolExecutor(6) as e:
        e.map(_resize, src.iterdir())


def p_tp_vs_tn(model, val_dl):
    ps_for_tp = []
    ps_for_tn = []
    model.eval()
    with torch.no_grad():
        for img, target in val_dl:
            img = img.cuda()
            logit = model(img)
            p = logit.sigmoid().cpu().float()
            p_for_tp = p.masked_select(target == 1)
            p_for_tn = p.masked_select(target == 0)
            ps_for_tp.append(p_for_tp)
            ps_for_tn.append(p_for_tn)
        ps_for_tp = torch.cat(ps_for_tp).numpy()
        ps_for_tn = torch.cat(ps_for_tn).numpy()
    return ps_for_tp, ps_for_tn


def p_wrt_test(model, test_dl):
    ps = []
    with torch.no_grad():
        model.eval()
        for img in test_dl:
            img = img.cuda()
            p = model(img).sigmoid().view(-1).cpu().float()
            ps.append(p)
    return torch.cat(ps).numpy()


def score_wrt_threshold(model, val_dl):
    with torch.no_grad():
        model.eval()
        predicts = []
        targets = []
        scores = []
        for img, target in val_dl:
            img, target = img.cuda(), target.cuda()
            logit = model(img)
            predict = logit.sigmoid()
            predicts.append(predict)
            targets.append(target)
        predicts = torch.cat(predicts)
        targets = torch.cat(targets)
        for threshold in T(np.linspace(0, 1, num=100, endpoint=False)).half():
            binaray_predicts = (predicts > threshold).float()
            tp = (binaray_predicts * targets).sum(dim=0)
            precision = tp / (binaray_predicts.sum(dim=0) + 1e-8)
            recall = tp / (targets.sum(dim=0) + 1e-8)
            f1 = 2*precision*recall/(precision+recall+1e-8)
            scores.append(f1.mean().item())
        return np.array(scores)
