from pathlib import Path
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

from concdvae.common.utils import PROJECT_ROOT, param_statistics
from concdvae.PT_train.training import train



def run(cfg: DictConfig) -> None:
    """
    Generic train loop
    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        np.random.seed(cfg.train.random_seed)
        random.seed(cfg.train.random_seed)
        torch.manual_seed(cfg.train.random_seed)
        # torch.backends.cudnn.deterministic = True
        if(cfg.accelerator != 'cpu'):
            torch.cuda.manual_seed(cfg.train.random_seed)
            torch.cuda.manual_seed_all(cfg.train.random_seed)

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)
    
    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    param_statistics(model)

    best_loss_old = None
    if(cfg.train.PT_train.start_epochs>1):
        filename = 'model_' + cfg.expname + '.pth'
        model_root = Path(hydra_dir) / filename
        if os.path.exists(model_root):
            checkpoint = torch.load(model_root, map_location=torch.device('cpu'))
            model_state_dict = checkpoint['model']
            model.load_state_dict(model_state_dict)
            cfg.train.PT_train.start_epochs = int(checkpoint['epoch'])
            best_loss_old = checkpoint['val_loss']

            print('use old model with loss=',best_loss_old,',and epoch = ',cfg.train.PT_train.start_epochs)


    model.lattice_scaler = datamodule.lattice_scaler.copy()
    torch.save(datamodule.lattice_scaler, hydra_dir / 'lattice_scaler.pt')

    if cfg.accelerator == 'DDP':
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        model.device = device
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif cfg.accelerator == 'gpu':
        model.device = 'cuda'
        model.cuda()

    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Device: {param.device}", file=sys.stdout)


    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    optimizer = hydra.utils.instantiate(
        cfg.optim.optimizer, params=model.parameters(), _convert_="partial"
    )
    scheduler = hydra.utils.instantiate(
        cfg.optim.lr_scheduler, optimizer=optimizer
    )

    hydra.utils.log.info('Start Train')
    test_losses, train_loss_epoch, val_loss_epoch = train(cfg, model, datamodule, optimizer, scheduler, hydra_dir, best_loss_old)


    hydra.utils.log.info('END')


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    #YCY
    if cfg.accelerator == 'DDP':
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        print(local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)

    run(cfg)


if __name__ == "__main__":
    main()