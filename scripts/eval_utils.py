import itertools
import hydra
from pathlib import Path
import numpy as np
import os
import torch
from torch_geometric.loader import DataLoader

from omegaconf import DictConfig, OmegaConf
from hydra.experimental import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import smact
from smact.screening import pauling_test

from concdvae.pl_data.datamodule import worker_init_fn
from concdvae.common.utils import PROJECT_ROOT
from concdvae.common.data_utils import chemical_symbols


def load_model(model_path, model_file, load_data=False):
    GlobalHydra.instance().clear()  # 清除之前的初始化
    with initialize_config_dir(str(model_path)):
        cfg = compose(config_name='hparams')
        ckpts = list(Path(model_path).glob('*.ckpt'))
        if len(ckpts) > 0:
            ckpt_epochs = np.array(
                [int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        if model_file != None:
            ckpt = os.path.join(model_path, model_file)


    model = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    state_dict = torch.load(ckpt, map_location="cpu")
    state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    # model_root = Path(model_path) / model_file
    # checkpoint = torch.load(model_root, map_location=torch.device('cpu'))
    # model_state_dict = checkpoint['model']
    # model.load_state_dict(model_state_dict)
    lattice_scaler = torch.load(Path(model_path) / 'lattice_scaler.pt')
    model.lattice_scaler = lattice_scaler

    if load_data :
        test_datasets = [hydra.utils.instantiate(dataset_cfg)
                         for dataset_cfg in cfg.data.datamodule.datasets.test]
        for test_dataset in test_datasets:
            test_dataset.lattice_scaler = lattice_scaler

        test_dataloaders = [
            DataLoader(
                test_datasets[i],
                shuffle=False,
                batch_size=cfg.data.datamodule.batch_size.test,
                num_workers=cfg.data.datamodule.num_workers.test,
                worker_init_fn=worker_init_fn,
            )
            for i in range(len(test_datasets))]
        test_loader = test_dataloaders[0]
    else:
        test_loader = None

    return model, test_loader, cfg


def load_data(file_path):
    if file_path[-3:] == 'npy':
        data = np.load(file_path, allow_pickle=True).item()
        for k, v in data.items():
            if k == 'input_data_batch':
                for k1, v1 in data[k].items():
                    data[k][k1] = torch.from_numpy(v1)
            else:
                data[k] = torch.from_numpy(v).unsqueeze(0)
    else:
        data = torch.load(file_path)
    return data


def load_config(model_path):
    with initialize_config_dir(str(model_path)):
        cfg = compose(config_name='hparams')
    return cfg


def get_crystals_list(
        frac_coords, atom_types, lengths, angles, num_atoms):
    """
    args:
        frac_coords: (num_atoms, 3)
        atom_types: (num_atoms)
        lengths: (num_crystals)
        angles: (num_crystals)
        num_atoms: (num_crystals)
    """
    assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
    assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

    start_idx = 0
    crystal_array_list = []
    for batch_idx, num_atom in enumerate(num_atoms.tolist()):
        cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
        cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]

        crystal_array_list.append({
            'frac_coords': cur_frac_coords.detach().cpu().numpy(),
            'atom_types': cur_atom_types.detach().cpu().numpy(),
            'lengths': cur_lengths.detach().cpu().numpy(),
            'angles': cur_angles.detach().cpu().numpy(),
        })
        start_idx = start_idx + num_atom
    return crystal_array_list


def smact_validity(comp, count,
                   use_pauling_test=True,
                   include_alloys=True):
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(
                        tuple([elem_symbols, ox_states, ratio]))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    if len(compositions) > 0:
        return True
    else:
        return False


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True