from lightai.core import *
from lightai.torch_core import *

# _weight = np.array([1.0]*28)
# _weight[[15, 16, 17, 18, 20, 26, 27]] = 1.5
# _weight *= (28/(28+7*0.5))
# _weight = T(_weight)


def f1_loss(predict, target):
    loss = 0
    # fp = predict[target == 0]
    # loss += ((fp.exp()+1).log()*fp.sigmoid()**2).mean()
    # predict = torch.clamp(predict * (1-target),
    #                       min=math.log(0.01/(1-0.01))) + predict * target

    lack_cls = target.sum(dim=0) == 0
    if lack_cls.any():
        loss += F.binary_cross_entropy_with_logits(
            predict[:, lack_cls], target[:, lack_cls])
        # loss += predict[:, lack_cls].mean()
    predict = torch.sigmoid(predict)

    predict = torch.clamp(predict * (1-target), min=0.01) + predict * target

    # predict = torch.where((predict < 0.01) * (target == 0),
    #                       torch.zeros_like(predict), predict)

    tp = predict * target
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + 1e-8)
    recall = tp / (target.sum(dim=0) + 1e-8)
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    # f1 *= _weight
    return 1 - f1.mean() + loss


class Loss_1728:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, predict, target):
        n_crop = predict.shape[0] // target.shape[0]
        target = target.view(target.shape[0], 1,
                             target.shape[1]).expand(target.shape[0], n_crop, target.shape[1])
        target = target.contiguous().view(-1, target.shape[-1])
        return self.loss_fn(predict, target)
