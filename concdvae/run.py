"""
Main training script for Con-CDVAE (Conditional Diffusion Variational Autoencoder).

This script handles the training pipeline for the crystal structure generation model,
including data loading, model instantiation, training, and evaluation.
"""

from pathlib import Path
from typing import List
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)

import hydra
import random
import numpy as np
import torch
import omegaconf
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


def build_callbacks(cfg: DictConfig, outpath=None, filename=None) -> List[Callback]:
    """
    Build PyTorch Lightning callbacks based on configuration.
    
    Args:
        cfg: Configuration dictionary containing callback settings
        outpath: Output path for model checkpoints
        filename: Filename for model checkpoints
        
    Returns:
        List of configured callbacks
    """
    callbacks: List[Callback] = []

    # Learning rate monitoring callback
    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    # Early stopping callback
    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    # Model checkpointing callback
    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        if outpath is not None:
            dirpath = Path(outpath)
        else:
            dirpath = Path(HydraConfig.get().run.dir)
        callbacks.append(
            ModelCheckpoint(
                dirpath=dirpath,
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
                save_last=False,
                filename=filename
            )
        )

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Generic training loop for CDVAE model.
    
    This function handles the complete training pipeline including:
    - Setting random seeds for reproducibility
    - Instantiating datamodule and model
    - Setting up callbacks and logging
    - Loading checkpoints if available
    - Training and testing the model
    
    Args:
        cfg: Run configuration, defined by Hydra in /conf
    """
    # Set random seeds for reproducibility
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)
        np.random.seed(cfg.train.random_seed)
        random.seed(cfg.train.random_seed)
        torch.manual_seed(cfg.train.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.train.random_seed)
            torch.cuda.manual_seed_all(cfg.train.random_seed)

    # Get Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule for data loading and preprocessing
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate the CDVAE model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # Pass lattice scaler from datamodule to model for normalization
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.lattice_scaler}>")
    model.lattice_scaler = datamodule.lattice_scaler.copy()
    torch.save(datamodule.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    datamodule.lattice_scaler.save_to_txt(hydra_dir / "lattice_scaler.txt")
    
    # Build training callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Store the YAML configuration for reproducibility
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    # Load checkpoint if it exists and resume training is enabled
    ckpts = list(hydra_dir.glob('*.ckpt'))
    if len(ckpts) > 0 and cfg.train.use_exit:
        last_ckpt = [ckpt for ckpt in ckpts if ckpt.name == "last.ckpt"]
        if last_ckpt:
            ckpt = str(last_ckpt[0])
            hydra.utils.log.info(f"found last checkpoint: {ckpt}")
        else:
            try:
                # Find checkpoint with highest epoch number
                ckpt_epochs = np.array([
                    int(ckpt.stem.split('-')[0].split('=')[1])
                    for ckpt in ckpts if "epoch=" in ckpt.stem
                ])
                ckpt = str(ckpts[np.argmax(ckpt_epochs)])
                hydra.utils.log.info(f"found checkpoint by max epoch: {ckpt}")
            except Exception as e:
                hydra.utils.log.warning(f"failed to parse epoch checkpoints: {e}")
                ckpt = None
    else:
        ckpt = None

    # Set up logging
    logger = None
    if "csvlogger" in cfg.logging:
        logger = CSVLogger(save_dir=hydra_dir, name=cfg.logging.csvlogger.name)

    # Instantiate PyTorch Lightning trainer
    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        **cfg.train.pl_trainer,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    # Start training
    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)
    
    # Save final checkpoint if configured
    if cfg.train.model_checkpoints.save_last:
        last_ckpt_path = os.path.join(hydra_dir, f"last.ckpt")
        trainer.save_checkpoint(last_ckpt_path)

    # Run testing/evaluation
    # test_dataloaders = datamodule.test_dataloader()
    hydra.utils.log.info("Starting testing!")
    trainer.test(datamodule=datamodule)

    hydra.utils.log.info("End")
    return 0


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    """
    Main entry point for the training script.
    
    Args:
        cfg: Configuration loaded by Hydra
    """
    # print(cfg)
    run(cfg)


if __name__ == "__main__":
    main()