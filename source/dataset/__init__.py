from omegaconf import DictConfig, open_dict
from .abcd import load_abcd_data
from .abide import load_abide_data
from .dataloader import init_my_dataloader
from typing import List

import torch.utils as utils
import torch.nn.functional as F
import torch

from .ts_data import load_dataset
from .preprocess import StandardScaler

import numpy as np
from scipy import stats


# def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

#     assert cfg.dataset.name in []

#     datasets = eval(f"load_{cfg.dataset.name}_data")(cfg)

#     dataloaders = (
# init_stratified_dataloader(cfg, *datasets)
#         if cfg.dataset.stratified
# else init_dataloader(cfg, *datasets)
#     )

#     return dataloaders


def my_dataset_factory(
    cfg: DictConfig, k: int, trial: int
) -> List[utils.data.DataLoader]:
    choices = [
        "oasis",
        "abide",
        "fbirn",
        "cobre",
        "abide_869",
        "ukb",
        "bsnip",
        "time_fbirn",
        "fbirn_100",
        "fbirn_200",
        "fbirn_400",
        "fbirn_1000",
        "hcp",
        "hcp_roi",
        "abide_roi",
    ]

    assert cfg.dataset.name in choices

    # [n_features; n_channels; time_len]
    final_timeseires, labels = load_dataset(cfg.dataset.name)

    for t in range(final_timeseires.shape[0]):
        for r in range(final_timeseires.shape[1]):
            final_timeseires[t, r, :] = stats.zscore(final_timeseires[t, r, :])

    final_pearson = np.zeros(
        (
            final_timeseires.shape[0],
            final_timeseires.shape[1],
            final_timeseires.shape[1],
        )
    )
    for i in range(final_timeseires.shape[0]):
        final_pearson[i, :, :] = np.corrcoef(final_timeseires[i, :, :])

    # final_timeseires, final_pearson, labels = [
    #     torch.from_numpy(data).float()
    #     for data in (final_timeseires, final_pearson, labels)
    # ]

    print("Data shape: ")
    print("TSeries: ", final_timeseires.shape)
    print("CorrMat: ", final_pearson.shape)
    print("Labels: ", labels.shape)

    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = (
            torch.from_numpy(final_pearson).float().shape[1:]
        )
        cfg.dataset.timeseries_sz = torch.from_numpy(final_timeseires).float().shape[2]

    dataloaders = init_my_dataloader(
        cfg, k, trial, final_timeseires, final_pearson, labels
    )

    return dataloaders
