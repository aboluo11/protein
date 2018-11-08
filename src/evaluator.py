from lightai.core import *

class Evaluator:
    def __init__(self, loss_fn, metric, val_dl, model):
        self.metric = metric
        self.val_dl = val_dl
        self.model = model
        self.loss_fn = loss_fn

    def __call__(self):
        losses = []
        with torch.no_grad():
            for img, target in val_dl:
                img = img.cuda()
                target = target.cuda()
                predict = self.model(img)
                self.metric(predict, target)
                losses.append(self.loss_fn(predict, target).mean(dim=1))
        loss = torch.cat(losses).mean().item()
        return loss, self.metric.res()