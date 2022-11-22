from datetime import datetime
import wandb
import hydra
from omegaconf import DictConfig, open_dict
from .dataset import my_dataset_factory
from .models import model_factory
from .components import lr_scheduler_factory, optimizers_factory, logger_factory
from .training import training_factory
from datetime import datetime

import argparse


def model_training(cfg: DictConfig, k: int, trial: int):

    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")

    dataloaders = my_dataset_factory(cfg, k, trial)
    logger = logger_factory(cfg)
    model = model_factory(cfg)
    optimizers = optimizers_factory(model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
    training = training_factory(
        cfg, model, optimizers, lr_schedulers, dataloaders, logger
    )

    training.train()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    group_name = f"{cfg.dataset.name}_{cfg.model.name}_{cfg.datasz.percentage}_{cfg.preprocess.name}"
    # _{cfg.training.name}\
    # _{cfg.optimizer[0].lr_scheduler.mode}"

    for k in range(5):
        for trial in range(10):
            for _ in range(cfg.repeat_time):
                run = wandb.init(project=group_name, name=f"k_{k}-trial_{trial:04d}")
                model_training(cfg, k, trial)

                run.finish()


if __name__ == "__main__":
    main()
