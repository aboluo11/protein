from lightai.core import *

def create_k_fold(k):
    trn_df = pd.read_csv('inputs/train.csv')
    trn_df = trn_df.sample(frac=1)
    trn_df['fold'] = (list(range(k)) * (len(trn_df) // k + 1))[:len(trn_df)]
    trn_df.to_csv(f'inputs/{k}_fold.csv', index=False)