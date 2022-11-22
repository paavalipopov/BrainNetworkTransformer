import torch
import torch.utils.data as utils
from torch.utils.data import TensorDataset
import torch.nn.functional as F

from omegaconf import DictConfig, open_dict
from typing import List

# from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


# def init_dataloader(cfg: DictConfig,
#                     final_timeseires: torch.tensor,
#                     final_pearson: torch.tensor,
#                     labels: torch.tensor) -> List[utils.DataLoader]:
#     labels = F.one_hot(labels.to(torch.int64))
#     length = final_timeseires.shape[0]
#     train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
#     val_length = int(length*cfg.dataset.val_set)
#     if cfg.datasz.percentage == 1.0:
#         test_length = length-train_length-val_length
#     else:
#         test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

#     with open_dict(cfg):
#         # total_steps, steps_per_epoch for lr schedular
#         cfg.steps_per_epoch = (
#             train_length - 1) // cfg.dataset.batch_size + 1
#         cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

#     dataset = utils.TensorDataset(
#         final_timeseires[:train_length+val_length+test_length],
#         final_pearson[:train_length+val_length+test_length],
#         labels[:train_length+val_length+test_length]
#     )

#     train_dataset, val_dataset, test_dataset = utils.random_split(
#         dataset, [train_length, val_length, test_length])
#     train_dataloader = utils.DataLoader(
#         train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

#     val_dataloader = utils.DataLoader(
#         val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

#     test_dataloader = utils.DataLoader(
#         test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

#     return [train_dataloader, val_dataloader, test_dataloader]


# def init_stratified_dataloader(cfg: DictConfig,
#                                final_timeseires: torch.tensor,
#                                final_pearson: torch.tensor,
#                                labels: torch.tensor,
#                                stratified: np.array) -> List[utils.DataLoader]:
#     labels = F.one_hot(labels.to(torch.int64))
#     length = final_timeseires.shape[0]
#     train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
#     val_length = int(length*cfg.dataset.val_set)
#     if cfg.datasz.percentage == 1.0:
#         test_length = length-train_length-val_length
#     else:
#         test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

#     with open_dict(cfg):
#         # total_steps, steps_per_epoch for lr schedular
#         cfg.steps_per_epoch = (
#             train_length - 1) // cfg.dataset.batch_size + 1
#         cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

#     split = StratifiedShuffleSplit(
#         n_splits=1, test_size=val_length+test_length, train_size=train_length, random_state=42)
#     for train_index, test_valid_index in split.split(final_timeseires, stratified):
#         final_timeseires_train, final_pearson_train, labels_train = final_timeseires[
#             train_index], final_pearson[train_index], labels[train_index]
#         final_timeseires_val_test, final_pearson_val_test, labels_val_test = final_timeseires[
#             test_valid_index], final_pearson[test_valid_index], labels[test_valid_index]
#         stratified = stratified[test_valid_index]

#     split2 = StratifiedShuffleSplit(
#         n_splits=1, test_size=test_length)
#     for test_index, valid_index in split2.split(final_timeseires_val_test, stratified):
#         final_timeseires_test, final_pearson_test, labels_test = final_timeseires_val_test[
#             test_index], final_pearson_val_test[test_index], labels_val_test[test_index]
#         final_timeseires_val, final_pearson_val, labels_val = final_timeseires_val_test[
#             valid_index], final_pearson_val_test[valid_index], labels_val_test[valid_index]

#     train_dataset = utils.TensorDataset(
#         final_timeseires_train,
#         final_pearson_train,
#         labels_train
#     )

#     val_dataset = utils.TensorDataset(
#         final_timeseires_val, final_pearson_val, labels_val
#     )

#     test_dataset = utils.TensorDataset(
#         final_timeseires_test, final_pearson_test, labels_test
#     )

#     train_dataloader = utils.DataLoader(
#         train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

#     val_dataloader = utils.DataLoader(
#         val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

#     test_dataloader = utils.DataLoader(
#         test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

#     return [train_dataloader, val_dataloader, test_dataloader]


def init_my_dataloader(
    cfg: DictConfig,
    k,
    trial,
    final_timeseires,
    final_pearson,
    labels,
) -> List[utils.DataLoader]:

    length = final_timeseires.shape[0]
    train_length = length // 5 * 3

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_index, test_index = list(skf.split(np.zeros(labels.shape[0]), labels))[k]

    X_train, X_test = final_timeseires[train_index], final_timeseires[test_index]
    M_train, M_test = final_pearson[train_index], final_pearson[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # train/val split
    X_train, X_val, M_train, M_val, y_train, y_val = train_test_split(
        X_train,
        M_train,
        y_train,
        test_size=X_train.shape[0] // 4,
        random_state=42 + trial,
        stratify=y_train,
    )

    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_val)
    y_test = torch.from_numpy(y_test)
    y_train = F.one_hot(y_train.to(torch.int64))
    y_val = F.one_hot(y_val.to(torch.int64))
    y_test = F.one_hot(y_test.to(torch.int64))

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(M_train, dtype=torch.float32),
        y_train,
    )
    valid_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(M_val, dtype=torch.float32),
        y_val,
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(M_test, dtype=torch.float32),
        y_test,
    )

    train_dataloader = utils.DataLoader(
        train_ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        drop_last=cfg.dataset.drop_last,
    )

    val_dataloader = utils.DataLoader(
        valid_ds, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False
    )

    test_dataloader = utils.DataLoader(
        test_ds, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False
    )

    return [train_dataloader, val_dataloader, test_dataloader]
