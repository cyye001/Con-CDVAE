import itertools
import hydra
from pathlib import Path
import numpy as np
import os
import torch
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
from omegaconf import DictConfig, OmegaConf
from hydra.experimental import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import smact
from smact.screening import pauling_test

from concdvae.pl_data.datamodule import worker_init_fn
from concdvae.common.utils import PROJECT_ROOT
from concdvae.common.data_utils import chemical_symbols


def load_model(model_path, model_file=None, load_data=False, prior_label=None):
    GlobalHydra.instance().clear()  # 清除之前的初始化
    with initialize_config_dir(str(model_path)):
        if prior_label is not None:
            config_name = f'hparams_{prior_label}'
            last_ckpt = Path(model_path) / f'{prior_label}-last.ckpt'
            ckpt_candidates = list(Path(model_path).glob(f"{prior_label}-epoch=*.ckpt"))
        else:
            config_name = 'hparams'
            last_ckpt = Path(model_path) / 'last.ckpt'
            ckpt_candidates = list(Path(model_path).glob("epoch=*.ckpt"))

        cfg = compose(config_name=config_name)
        if model_file is not None:
            ckpt = os.path.join(model_path, model_file)
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(f"can not find: {ckpt}")
        else:
            ckpt = None
            if last_ckpt.exists():
                ckpt = str(last_ckpt)
            else:
                if ckpt_candidates:
                    try:
                        # 提取 epoch 数字并排序
                        ckpt_epochs = np.array([
                            int(ckpt.stem.split('=')[1].split('-')[0]) for ckpt in ckpt_candidates
                        ])
                        latest_idx = ckpt_epochs.argsort()[-1]
                        ckpt = str(ckpt_candidates[latest_idx])
                    except Exception as e:
                        raise RuntimeError(f"Err: {e}")

        if ckpt is None:
            raise FileNotFoundError("can not find checkpoint")

    print('load ckpt:',ckpt, flush=True)
    if prior_label is not None:
        model = hydra.utils.instantiate(
            cfg.prior.prior_model,
            optim=cfg['optim'],
            data=cfg['data'],
            logging=cfg['logging'],
            _recursive_=False,
        )
    else:
        model = hydra.utils.instantiate(
            cfg.model,
            optim=cfg.optim,
            data=cfg.data,
            logging=cfg.logging,
            _recursive_=False,
        )


    state_dict = torch.load(ckpt, map_location="cpu")
    state_dict = state_dict["state_dict"]
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_model.")}
    model.load_state_dict(state_dict)
    
    if prior_label is None:
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
    

def generation(model, prior, input_dict, 
               batch_size=512, down_sample=1, num_batches_to_sample=1, 
               ld_kwargs=None):

    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    if down_sample>1:
        prior_batch_size = int(batch_size / down_sample)
    else:
        prior_batch_size = int(batch_size)


    with torch.no_grad():
        for z_idx in range(num_batches_to_sample):
            print('No.', z_idx + 1, ' in ', num_batches_to_sample, flush=True)
            batch_all_frac_coords = []
            batch_all_atom_types = []
            batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
            batch_lengths, batch_angles = [], []

            randan_z = torch.randn(batch_size, model.hparams.latent_dim, device=model.device)
            z = prior.gen(input_dict, randan_z, ld_kwargs)

            ## down sample
            mae_list = [0] * z.size(0)
            if down_sample > 1:
                for j in range(len(model.conditions_predict)):
                    pre = model.conditions_predict[j].mlp(z)
                    prop_name = model.hparams.conditionpre.condition_predictp[j].condition_name
                    if pre.size(-1) == 1:  ##regression
                        a = model.conditions_predict[j].condition_max
                        b = model.conditions_predict[j].condition_min
                        # pre = (pre * (a - b)) + b
                        mae_tensor = torch.abs(pre - (input_dict[prop_name].item() - b) / (a - b))

                    else:  ##classification
                        pre = F.softmax(pre, dim=1)
                        pre = pre[:, int(input_dict[prop_name].item())]
                        pre = pre.view(-1, 1)
                        mae_tensor = torch.abs(1 - pre)

                    for k in range(batch_size):
                        sub_tensor = mae_tensor[k:k + 1, :].cpu()
                        mae_list[k] += sub_tensor.item()

                sorted_indices = sorted(range(len(mae_list)), key=lambda k: mae_list[k])
                min_n_indices = sorted_indices[:prior_batch_size]
                z = z[min_n_indices, :]

                print('loss', mae_list[sorted_indices[0]], mae_list[sorted_indices[prior_batch_size]])
                print('after down sample', z.shape)


            condition_emb = model.condition_model(input_dict)
            condition_emb = condition_emb.repeat(prior_batch_size, 1).float()
            z_con = torch.cat((z, condition_emb), dim=1)
            z_con = model.z_condition(z_con)

            for sample_idx in range(ld_kwargs.num_samples_per_z):
                samples = model.langevin_dynamics(z_con, ld_kwargs)

                # collect sampled crystals in this batch.
                batch_frac_coords.append(samples['frac_coords'].detach().cpu())
                batch_num_atoms.append(samples['num_atoms'].detach().cpu())
                batch_atom_types.append(samples['atom_types'].detach().cpu())
                batch_lengths.append(samples['lengths'].detach().cpu())
                batch_angles.append(samples['angles'].detach().cpu())
                if ld_kwargs.save_traj:
                    batch_all_frac_coords.append(
                        samples['all_frac_coords'][::ld_kwargs.down_sample_traj_step].detach().cpu())
                    batch_all_atom_types.append(
                        samples['all_atom_types'][::ld_kwargs.down_sample_traj_step].detach().cpu())
            
            # collect sampled crystals for this z.
            frac_coords.append(torch.stack(batch_frac_coords, dim=0))
            num_atoms.append(torch.stack(batch_num_atoms, dim=0))
            atom_types.append(torch.stack(batch_atom_types, dim=0))
            lengths.append(torch.stack(batch_lengths, dim=0))
            angles.append(torch.stack(batch_angles, dim=0))
            if ld_kwargs.save_traj:
                all_frac_coords_stack.append(
                    torch.stack(batch_all_frac_coords, dim=0))
                all_atom_types_stack.append(
                    torch.stack(batch_all_atom_types, dim=0))
                
    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack)