from lightai.core import *

class F1:
    def __init__(self, threshold):
        self.predicts = []
        self.targets = []
        self.threshold = threshold

    def __call__(self, predict, target):
        """
        predict and target are in batch
        """
        predict = torch.sigmoid(predict)
        predict = predict > self.threshold
        self.predicts.append(predict)
        self.targets.append(target)
        
    def res(self):
        predict = torch.cat(self.predicts)
        target = torch.cat(self.targets)
        tp = (predict*target).sum(dim=0).float()
        precision = tp/(predict.sum(dim=0).float() + 1e-8)
        recall = tp/(target.sum(dim=0).float() + 1e-8)
        f1 = 2*(precision*recall/(precision+recall+1e-8))
        self.predicts = []
        self.targets = []
        return f1.mean().item()