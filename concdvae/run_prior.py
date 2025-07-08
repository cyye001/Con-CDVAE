from pathlib import Path
from typing import List
import sys
import os
import argparse
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)
import logging

import hydra
import random
import numpy as np
import torch
import omegaconf
from hydra.experimental import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import hydra
import numpy as np
import torch
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

from concdvae.common.utils import PROJECT_ROOT, log_hyperparameters
from concdvae.run import build_callbacks
from scripts.eval_utils import load_model
from concdvae.pl_prior.utils import update_prior_cfg


def main(args):
    # load the CDVAE and cfg
    model_path = args.model_path # Path(args.model_path)
    model_file = args.model_file
    model, _, cfg = load_model(model_path, model_file)


    # set the message for logging
    log = hydra.utils.log
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    file_handler = logging.FileHandler(model_path + "/run.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.handlers.clear()
    log.addHandler(console_handler)
    log.addHandler(file_handler)
    log.setLevel(logging.INFO)


    # update hydra config
    cfg = update_prior_cfg(cfg, args)


    # set the random seed
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)
        np.random.seed(cfg.train.random_seed)
        random.seed(cfg.train.random_seed)
        torch.manual_seed(cfg.train.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.train.random_seed)
            torch.cuda.manual_seed_all(cfg.train.random_seed)


    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    logger = None
    if "csvlogger" in cfg.logging:
        logger = CSVLogger(save_dir=model_path, name=cfg.logging.csvlogger.name+"-"+args.prior_label)
    
    callbacks: List[Callback] = build_callbacks(cfg=cfg, outpath=model_path, filename=args.prior_label + "-{epoch}-{step}")

    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (Path(model_path) / ("hparams_"+args.prior_label+".yaml")).write_text(yaml_conf)

    prior = hydra.utils.instantiate(
        cfg.prior.prior_model,
        optim=cfg['optim'],
        data=cfg['data'],
        logging=cfg['logging'],
        _recursive_=False,
    )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    prior._model = model
    

    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=model_path,
        logger=logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        **cfg.train.pl_trainer,
    )

    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    ckpts = list(Path(model_path).glob(f'{args.prior_label}*.ckpt'))
    print(f"ckpts: {ckpts}")
    if len(ckpts) > 0 and cfg.train.use_exit:
        last_ckpt = [ckpt for ckpt in ckpts if ckpt.name == f"{args.prior_label}-last.ckpt"]
        print(f"last_ckpt: {last_ckpt}")
        if last_ckpt:
            ckpt = str(last_ckpt[0])
            hydra.utils.log.info(f"found last checkpoint: {args.prior_label}-last.ckpt")
        else:
            try:
                ckpt_epochs = np.array([
                    int(ckpt.stem.split('=')[1].split('-')[0])
                    for ckpt in ckpts if "epoch=" in ckpt.stem
                ])
                ckpt = str(ckpts[np.argmax(ckpt_epochs)])
                hydra.utils.log.info(f"found checkpoint by max epoch: {ckpt}")
            except Exception as e:
                hydra.utils.log.warning(f"failed to parse epoch checkpoints: {e}")
                ckpt = None
    else:
        ckpt = None
    print(f"ckpt: {ckpt}")

    hydra.utils.log.info("Starting training!")
    trainer.fit(model=prior, datamodule=datamodule, ckpt_path=ckpt)

    if cfg.train.model_checkpoints.save_last:
        last_ckpt_path = os.path.join(model_path, f"{args.prior_label}-last.ckpt")
        trainer.save_checkpoint(last_ckpt_path)

    # test_dataloaders = datamodule.test_dataloader()
    hydra.utils.log.info("Starting testing!")
    trainer.test(datamodule=datamodule)

    hydra.utils.log.info("End")
    return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model_file', default='epoch=330-step=17543.ckpt', type=str)
    parser.add_argument('--prior_label', default='prior', type=str)

    parser.add_argument('--prior_file', default='default', type=str)
    parser.add_argument('--train_file', default='prior_default', type=str)
    parser.add_argument('--optim_file', default='prior_default', type=str)
    parser.add_argument('--priorcondition_file', default=None, type=str)  # if use full strategy, set the conditionmodel file
    parser.add_argument('--data_file', default=None, type=str)            # if use full strategy, set the new dataset
    args = parser.parse_args()

    main(args)