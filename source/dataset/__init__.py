import torch
import torch.utils.data as utils
from torch.utils.data import TensorDataset
import torch.nn.functional as F

from omegaconf import DictConfig, open_dict
from .abcd import load_abcd_data
from .abide import load_abide_data
from .dataloader import init_my_dataloader
from typing import List


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


def my_dataset_factory(cfg: DictConfig, k: int, trial: int) -> List[utils.DataLoader]:
    choices = [
        "oasis",
        "adni",
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

    tcs = []
    tc = final_timeseires.shape[2]
    if cfg.model.name == "FBNETGEN":
        tc = tc - tc % cfg.model.window_size
    tcs.append(tc)

    if cfg.dataset.name == "fbirn":
        extra_ds = ["bsnip", "cobre"]
    elif cfg.dataset.name == "bsnip":
        extra_ds = ["fbirn", "cobre"]
    elif cfg.dataset.name == "cobre":
        extra_ds = ["fbirn", "bsnip"]
    elif cfg.dataset.name == "oasis":
        extra_ds = ["adni"]
    elif cfg.dataset.name == "adni":
        extra_ds = ["oasis"]
    else:
        extra_ds = []

    fbnet = False
    if cfg.model.name == "FBNETGEN":
        fbnet = True
        for ds in extra_ds:
            local_timeseires, _ = load_dataset(ds)
            tc = (
                local_timeseires.shape[2]
                - local_timeseires.shape[2] % cfg.model.window_size
            )
            tcs.append(local_timeseires.shape[2])

    # find minimal time-course
    min_tc = np.min(tcs)

    if fbnet:
        final_timeseires = final_timeseires[:, :, :min_tc]

    for t in range(final_timeseires.shape[0]):
        for r in range(final_timeseires.shape[1]):
            final_timeseires[t, r, :] = stats.zscore(final_timeseires[t, r, :])

    good_indices = []
    for i, ts in enumerate(final_timeseires):
        if np.sum(np.isnan(ts)) == 0:
            good_indices += [i]

    final_timeseires = final_timeseires[good_indices]
    labels = labels[good_indices]

    final_pearson = np.zeros(
        (
            final_timeseires.shape[0],
            final_timeseires.shape[1],
            final_timeseires.shape[1],
        )
    )
    for i in range(final_timeseires.shape[0]):
        final_pearson[i, :, :] = np.corrcoef(final_timeseires[i, :, :])

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

    extra_dataloaders = {}
    for ds in extra_ds:
        final_timeseires, labels = load_dataset(ds)
        final_timeseires[final_timeseires != final_timeseires] = 0
        if fbnet:
            final_timeseires = final_timeseires[:, :, :min_tc]

        for t in range(final_timeseires.shape[0]):
            for r in range(final_timeseires.shape[1]):
                final_timeseires[t, r, :] = stats.zscore(final_timeseires[t, r, :])
        good_indices = []
        for i, ts in enumerate(final_timeseires):
            if np.sum(np.isnan(ts)) == 0:
                good_indices += [i]

        final_timeseires = final_timeseires[good_indices]
        labels = labels[good_indices]

        final_pearson = np.zeros(
            (
                final_timeseires.shape[0],
                final_timeseires.shape[1],
                final_timeseires.shape[1],
            )
        )
        for i in range(final_timeseires.shape[0]):
            final_pearson[i, :, :] = np.corrcoef(final_timeseires[i, :, :])

        labels = torch.from_numpy(labels)
        labels = F.one_hot(labels.to(torch.int64))
        test_ds = TensorDataset(
            torch.tensor(final_timeseires, dtype=torch.float32),
            torch.tensor(final_pearson, dtype=torch.float32),
            labels,
        )

        extra_dataloaders[ds] = utils.DataLoader(
            test_ds,
            batch_size=cfg.dataset.batch_size,
            shuffle=True,
            drop_last=False,
        )

    dataloaders.append(extra_dataloaders)

    return dataloaders
