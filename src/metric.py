from lightai.core import *
from lightai.torch_core import *


class F1:
    def __init__(self, threshold=0.5):
        self.predicts = []
        self.targets = []
        self.threshold = threshold

    def __call__(self, predict, target):
        """
        predict and target are in batch
        """
        self.predicts.append(predict)
        self.targets.append(target)

    def res(self):
        origin_predict = torch.cat(self.predicts).sigmoid()
        target = torch.cat(self.targets)
        scores = []
        predict = (origin_predict > self.threshold).float()
        tp = (predict*target).sum(dim=0)  # shape (28,)
        precision = tp/(predict.sum(dim=0) + 1e-8)
        recall = tp/(target.sum(dim=0) + 1e-8)
        f1 = 2*(precision*recall/(precision+recall+1e-8))
        self.predicts = []
        self.targets = []
        return f1.mean().item()


class Metric_1728:
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, predict, target):
        n_crop = predict.shape[0] // target.shape[0]
        target = target.view(target.shape[0], 1,
                             target.shape[1]).expand(target.shape[0], n_crop, target.shape[1])
        target = target.contiguous().view(-1, target.shape[-1])
        self.metric(predict, target)

    def res(self):
        return self.metric.res()


class CropEvaluator:
    def __init__(self):
        self.scores = []

    def __call__(self, predict, target):
        predict = predict.view(target.shape[0], -1, 28)
        target = target.view(target.shape[0], 1,
                             target.shape[1]).expand_as(predict)
        predict = predict.sigmoid()
        tp = predict * target
        score = tp.sum(dim=-1) / target.sum(dim=-1)
        self.scores.append(score)

    def res(self):
        self.scores = torch.cat(self.scores)
