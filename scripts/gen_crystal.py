import time
import argparse
import torch
import hydra
import random
import yaml
import numpy as np
import pandas as pd
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from pymatgen.core.structure import Structure
from types import SimpleNamespace
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from omegaconf import OmegaConf

from eval_utils import load_model, generation
from condition_diff_z import condition_diff_z
from concdvae.pl_data.dataset import AtomCustomJSONInitializer, formula2atomnums
from concdvae.common.utils import PROJECT_ROOT


def main(cfg):

    if cfg.deterministic:
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)

    if str(cfg.prior_path).lower() == 'none':
        cfg.prior_path = cfg.model_path
    if str(cfg.prior_label).lower() == 'none':
        cfg.prior_label = cfg.prior_file.split('-epoch')[0]


    ## prepare the model
    model, _, _ = load_model(cfg.model_path, cfg.model_file)
    prior, _, cfg_train = load_model(cfg.prior_path, cfg.prior_file, prior_label=cfg.prior_label)
    for param in model.parameters():
        param.requires_grad = False
    for param in prior.parameters():
        param.requires_grad = False
    model.eval()
    prior.eval()
    need_props = [x.condition_name for x in cfg_train.prior.prior_model.conditionmodel.condition_embeddings]
    print('need_props:', need_props)

    if torch.cuda.is_available():
        model.to('cuda')
        prior.to('cuda')

    ## prepare generation
    input_path = os.path.join(cfg.model_path, cfg.input_path)
    input_data = pd.read_csv(input_path)

    label_list = list(input_data['label'])
    column_names = input_data.columns.tolist()
    try:
        refdata = pd.read_csv(cfg.refdata_path)
        prior._refdata = refdata
        print(prior._refdata.shape)
    except Exception as e:
        pass

    ld_kwargs = SimpleNamespace(num_samples_per_z=cfg.num_samples_per_z,
                                save_traj=cfg.save_traj,
                                down_sample_traj_step=cfg.down_sample_traj_step,
                                use_one=cfg.use_one,
                                disable_bar=cfg.disable_bar,
                                min_sigma=cfg.min_sigma,
                                step_lr=cfg.step_lr,
                                n_step_each=cfg.n_step_each,)

    # start generating
    for i_prop in range(len(label_list)):
        start_time = time.time()
        print('No.', i_prop + 1, ' in ', len(label_list), 'with label = ', label_list[i_prop], flush=True)

        if torch.cuda.is_available():
            prop_dict = {k: torch.Tensor([input_data[k][i_prop]]).float().cuda() for k in column_names if k in need_props}
        else:
            prop_dict = {k: torch.Tensor([input_data[k][i_prop]]).float() for k in column_names if k in need_props}

        (frac_coords, num_atoms, atom_types, lengths, angles,
        all_frac_coords_stack, all_atom_types_stack) = generation(model, prior, prop_dict ,
                       batch_size=cfg.batch_size, 
                       down_sample=cfg.down_sample, 
                       num_batches_to_sample=cfg.num_batches_to_samples,
                       ld_kwargs=ld_kwargs)
        
        if cfg.label == '':
            gen_out_name = f'eval_gen_{label_list[i_prop]}.pt'
        else:
            gen_out_name = f'eval_gen_{cfg.label}_{label_list[i_prop]}.pt'

        torch.save({
                'eval_setting': args,
                'frac_coords': frac_coords,
                'num_atoms': num_atoms,
                'atom_types': atom_types,
                'lengths': lengths,
                'angles': angles,
                'all_frac_coords_stack': all_frac_coords_stack,
                'all_atom_types_stack': all_atom_types_stack,
                'time': time.time() - start_time
            }, Path(cfg.model_path) / gen_out_name)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='crystal generation')
    parser.add_argument('--config', default='/data/work/cyye/0-project/15-con_cdvae/Con-CDVAE/conf/gen/default.yaml', type=str, metavar='N')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)
    # print('config:', config, flush=True)

    main(config)