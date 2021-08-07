"""
@Time   :   2021-01-21 14:58:30
@File   :   csc.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
from torch.utils.data import DataLoader

from bbcm.data.datasets.csc import CscDataset, PureTextDataset


def get_csc_loader(fp, _collate_fn, **kwargs):
    dataset = CscDataset(fp)
    if kwargs.get('pure_text_dataset'):
        dataset = PureTextDataset(fp)
        kwargs['shuffle'] = False
    loader = DataLoader(dataset, collate_fn=_collate_fn, **kwargs)
    return loader
