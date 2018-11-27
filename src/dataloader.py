from lightai.core import *
from torch.utils.data.dataloader import default_collate
import torch.multiprocessing as mp
from .dataset import *


def worker_loop(dss, ds_idx_q, row_idx_q, out_q):
    while True:
        ds_idx = ds_idx_q.get()
        row_idx = row_idx_q.get()
        batch = default_collate([dss[ds_i][row_i] for ds_i, row_i in zip(ds_idx, row_idx)])
        out_q.put(batch)


class MyDataLoader:
    def __init__(self, bs, num_workers, fold, iters_per_epoch, tsfm):
        dfs = [pd.read_csv(f'inputs/28_classes/df/class_{i}.csv') for i in range(28)]
        dss = [Dataset(df, fold, train=True, tsfm=tsfm) for df in dfs]
        self.bs = bs
        self.iters_per_epoch = iters_per_epoch
        self.workers = []
        self.ds_idx_q = mp.Queue()
        self.row_idx_q = mp.Queue()
        self.out_q = mp.Queue()
        self.ds_samplers = [RandomSampler(dss[i]) for i in range(28)]
        self.ds_sampler_iters = [iter(self.ds_samplers[i]) for i in range(28)]
        for i in range(num_workers):
            w = mp.Process(target=worker_loop, args=(dss, self.ds_idx_q, self.row_idx_q, self.out_q))
            w.daemon = True
            w.start()
            self.workers.append(w)
        self.batch_idx_iter = self.get_batch_idx()
        for i in range(num_workers):
            self.put_idx()

    def __iter__(self):
        for _ in range(len(self)):
            batch = self.out_q.get()
            self.put_idx()
            yield batch

    def put_idx(self):
        ds_idxes, row_idxes = next(self.batch_idx_iter)
        self.ds_idx_q.put(ds_idxes)
        self.row_idx_q.put(row_idxes)

    def get_batch_idx(self):
        ds_idxes = []
        row_idxes = []
        while True:
            for ds_idx in range(28):
                row_idx = next(self.ds_sampler_iters[ds_idx], None)
                if row_idx is None:
                    self.ds_sampler_iters[ds_idx] = iter(self.ds_samplers[ds_idx])
                    row_idx = next(self.ds_sampler_iters[ds_idx], None)
                ds_idxes.append(ds_idx)
                row_idxes.append(row_idx)
                if len(row_idxes) == self.bs:
                    yield ds_idxes, row_idxes
                    ds_idxes = []
                    row_idxes = []

    def __len__(self):
        return 28*self.iters_per_epoch//self.bs