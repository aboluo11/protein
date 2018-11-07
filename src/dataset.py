from lightai.core import *

class Dataset:
    def __init__(self, df, fold, train, tsfm):
        if train:
            self.df = df[df['fold'] != fold]
        else:
            self.df = df[df['fold'] == fold]
        self.tsfm = tsfm

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = self.tsfm(row)
        return sample

    def __len__(self):
        return len(self.df)