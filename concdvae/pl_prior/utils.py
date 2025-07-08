import os
from concdvae.common.utils import PROJECT_ROOT
from omegaconf import OmegaConf

def update_prior_cfg(cfg, args):
    if args.prior_file != None:
        prior_path = os.path.join(PROJECT_ROOT, 'conf', 'prior', args.prior_file)
        if not prior_path.endswith('.yaml'):
            prior_path += '.yaml'
        if not os.path.isfile(prior_path):
            raise FileNotFoundError(f"can not find: {prior_path}")
        new_prior_cfg = OmegaConf.load(prior_path)
        cfg['prior'] = new_prior_cfg
    
    if args.train_file != None:
        train_path = os.path.join(PROJECT_ROOT, 'conf', 'train', args.train_file)
        if not train_path.endswith('.yaml'):
            train_path += '.yaml'
        if not os.path.isfile(train_path):
            raise FileNotFoundError(f"can not find: {train_path}")
        new_train_cfg = OmegaConf.load(train_path)
        cfg['train'] = new_train_cfg

    if args.optim_file != None:
        optim_path = os.path.join(PROJECT_ROOT, 'conf', 'optim', args.optim_file)
        if not optim_path.endswith('.yaml'):
            optim_path += '.yaml'
        if not os.path.isfile(optim_path):
            raise FileNotFoundError(f"can not find: {optim_path}")
        new_optim_cfg = OmegaConf.load(optim_path)
        cfg['optim'] = new_optim_cfg

    if args.priorcondition_file != None:
        priorcondition_path = os.path.join(PROJECT_ROOT, 'conf', 'model', 'conditionmodel', args.priorcondition_file)
        if not priorcondition_path.endswith('.yaml'):
            priorcondition_path += '.yaml'
        if not os.path.isfile(priorcondition_path):
            raise FileNotFoundError(f"can not find: {priorcondition_path}")
        new_priorcondition_cfg = OmegaConf.load(priorcondition_path)
        cfg['prior']['prior_model']['conditionmodel'] = new_priorcondition_cfg

    if args.data_file != None:
        data_path = os.path.join(PROJECT_ROOT, 'conf', 'data', args.data_file)
        if not data_path.endswith('.yaml'):
            data_path += '.yaml'
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"can not find: {data_path}")
        new_data_cfg = OmegaConf.load(data_path)
        cfg['data'] = new_data_cfg

    OmegaConf.resolve(cfg)
    return cfg