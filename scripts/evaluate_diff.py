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

from eval_utils import load_model
from condition_diff_z import condition_diff_z
from concdvae.common.data_utils import GaussianDistance
from concdvae.pl_data.datamodule import worker_init_fn
from concdvae.pl_data.dataset import AtomCustomJSONInitializer, formula2atomnums

def generation(model, conz_model, ld_kwargs, num_batches_to_sample, num_samples_per_z, #GDF,
               i_prop, prop_dict, batch_size=512, down_sample_traj_step=1,down_sample=1):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    if down_sample>1:
        real_batch_size = int(batch_size / down_sample)
    else:
        real_batch_size = int(batch_size)
    condition_emb = model.condition_model(prop_dict)

    condition_emb = condition_emb.repeat(real_batch_size, 1).float()

    model.eval()
    conz_model.eval()

    model_props = []
    dict_props = []
    for i in range(len(model.conditions_name)):
        model_props.append(model.conditions_name[i])
        dict_props.append(model.conditions_name[i])

    for i in range(len(dict_props)):
        if dict_props[i] == 'band_gap':
            dict_props[i] == 'bandgap'
        elif dict_props[i] == 'formation_energy_per_atom':
            dict_props[i] == 'formation'

    with torch.no_grad():
        print('in no_grad!!!!')
        for z_idx in range(num_batches_to_sample):
            
            print('No.', z_idx + 1, ' in ', num_batches_to_sample, file=sys.stdout)
            sys.stdout.flush()
            batch_all_frac_coords = []
            batch_all_atom_types = []
            batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
            batch_lengths, batch_angles = [], []

            randan_z = torch.randn(batch_size, model.hparams.latent_dim, device=model.device)

            z = conz_model.gen(prop_dict, randan_z)
            mae_list = [0] * z.size(0)
            if down_sample > 1:
                for j in range(len(model_props)):
                    pre = model.conditions_predict[j].mlp(z)
                    if pre.size(-1) == 1:  ##regression
                        a = model.conditions_predict[j].condition_max
                        b = model.conditions_predict[j].condition_min
                        # pre = (pre * (a - b)) + b
                        mae_tensor = torch.abs(pre - (prop_dict[dict_props[j]].item() - b) / (a - b))

                    else:  ##classification
                        pre = F.softmax(pre, dim=1)
                        pre = pre[:, int(prop_dict[dict_props[j]].item())]
                        pre = pre.view(-1, 1)
                        mae_tensor = torch.abs(1 - pre)

                    for k in range(batch_size):
                        sub_tensor = mae_tensor[k:k + 1, :].cpu()
                        mae_list[k] += sub_tensor.item()

                sorted_indices = sorted(range(len(mae_list)), key=lambda k: mae_list[k])
                min_n_indices = sorted_indices[:real_batch_size]
                z = z[min_n_indices, :]

                print('mae', mae_list[sorted_indices[0]], mae_list[sorted_indices[real_batch_size]])
                print('after down sample', z.shape)

            z_con = torch.cat((z, condition_emb), dim=1)
            z_con = model.z_condition(z_con)

            for sample_idx in range(num_samples_per_z):
                samples = model.langevin_dynamics(z_con, ld_kwargs)

                # collect sampled crystals in this batch.
                batch_frac_coords.append(samples['frac_coords'].detach().cpu())
                batch_num_atoms.append(samples['num_atoms'].detach().cpu())
                batch_atom_types.append(samples['atom_types'].detach().cpu())
                batch_lengths.append(samples['lengths'].detach().cpu())
                batch_angles.append(samples['angles'].detach().cpu())
                if ld_kwargs.save_traj:
                    batch_all_frac_coords.append(
                        samples['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                    batch_all_atom_types.append(
                        samples['all_atom_types'][::down_sample_traj_step].detach().cpu())

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


def main(args):

    if args.deterministic:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    model_path = Path(args.model_path)
    # load_data if do reconstruction.
    model, test_loader, cfg = load_model(args.model_path, args.model_file,
                                         load_data=('recon' in args.tasks) or
                              ('opt' in args.tasks and args.start_from == 'data'))
    # conz_path = Path(args.conz_path)
    conz_model_root = model_path / args.conz_file
    checkpoint = torch.load(conz_model_root, map_location=torch.device('cpu'))
    args_conz = checkpoint['args']
    args_conz = SimpleNamespace(**args_conz)
    if args_conz.fullfea == 1:
        print('use full feature')
        cfg.data.root_path = '${oc.env:PROJECT_ROOT}/data/'+args_conz.newdata
        atom_init_file = os.path.join(cfg.data.root_path, 'atom_init.json')
        ari = AtomCustomJSONInitializer(atom_init_file)
        cfg.data.prop = ['bandgap','formation','e_above_hull','a','b','c','alpha','beta','gamma','density',
                         'coor_number','n_atom','spacegroup','crystal_system']
        cfg.data.use_prop = 'formation'
        condition_root = args_conz.newcond
        if not torch.cuda.is_available():
            cfg.data.preprocess_workers = 1
        with open(condition_root, 'r') as file:
            new_condition = yaml.safe_load(file)##

    else:
        print('use default feature')
        new_condition = None
        ari = None


    ld_kwargs_conz = SimpleNamespace(fc_num_layers=args_conz.fc_num_layers,
                                hidden_dim=args_conz.hidden_dim,
                                resnet=args_conz.resnet,
                                ddpm_noise_start=args_conz.ddpm_noise_start,
                                ddpm_noise_end=args_conz.ddpm_noise_end,
                                ddpm_n_noise=args_conz.ddpm_n_noise,
                                time_emb_dim=args_conz.time_emb_dim,
                                n_UNet_lay=args_conz.n_UNet_lay,
                                new_condition=new_condition,
                                ari=ari,
                                EMB_way=args.EMB_way,
                                tolerate=args.tolerate,
                                use_one=args.use_one,
                                data_root=cfg.data.root_path,
                                )
    conz_model = condition_diff_z(cfg, ld_kwargs_conz)
    print(conz_model)
    model_state_dict = checkpoint['model']
    conz_model.load_state_dict(model_state_dict)
    for param in model.parameters():
        param.requires_grad = False
    for param in conz_model.parameters():
        param.requires_grad = False
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)

    if torch.cuda.is_available():
        model.to('cuda')
        model.device = 'cuda'
        conz_model.to('cuda')
        conz_model.device = 'cuda'


    if 'gen' in args.tasks:
        print('Evaluate model on the generation task.')
        prop_path = model_path / args.prop_path

        prop_data = pd.read_csv(prop_path)
        label_list = list(prop_data['label'])
        column_names = prop_data.columns.tolist()
        start_time = time.time()
        for i_prop in range(len(label_list)):
            print('No.', i_prop + 1, ' in ', len(label_list), 'with label = ', label_list[i_prop], file=sys.stdout)

            if torch.cuda.is_available():
                prop_dict = {k: torch.Tensor([prop_data[k][i_prop]]).float().cuda() for k in column_names if k not in ['label','cif','formula']}
            else:
                prop_dict = {k: torch.Tensor([prop_data[k][i_prop]]).float() for k in column_names if k not in ['label','cif','formula']}
            if args_conz.fullfea == 1 and 'formula' in prop_data.keys():
                atom_list = formula2atomnums(prop_data['formula'][i_prop])
                atom_fea = np.vstack([ari.get_atom_fea(atom_list[k])
                                      for k in range(len(atom_list))])
                atom_fea = torch.Tensor(atom_fea)
                atom_fea = torch.mean(atom_fea, dim=0)
                atom_fea = atom_fea.reshape(1, 92)
                if torch.cuda.is_available():
                    prop_dict.update({'formula':atom_fea.cuda()})
                else:
                    prop_dict.update({'formula': atom_fea})



            (frac_coords, num_atoms, atom_types, lengths, angles,
             all_frac_coords_stack, all_atom_types_stack) = generation(
                model, conz_model, ld_kwargs, args.num_batches_to_samples, args.num_evals,#GDF,
                i_prop, prop_dict, args.batch_size, args.down_sample_traj_step, args.down_sample)

            if args.label == '':
                gen_out_name = f'eval_gen_{label_list[i_prop]}.pt'
            else:
                gen_out_name = f'eval_gen_{args.label}_{label_list[i_prop]}.pt'

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
            }, model_path / gen_out_name)



    print('end')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    # parser.add_argument('--conz_path', required=True)
    parser.add_argument('--model_file', default='model_test.pth', type=str)
    parser.add_argument('--conz_file', default='conz_model_ABC_diffu.pth', type=str)
    parser.add_argument('--tasks', nargs='+', default='gen')
    parser.add_argument('--n_step_each', default=100, type=int)  #default=100
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_batches_to_samples', default=2, type=int)
    parser.add_argument('--start_from', default='data', type=str)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--down_sample', default=10, type=int)
    parser.add_argument('--force_num_atoms', action='store_true')
    parser.add_argument('--force_atom_types', action='store_true')
    parser.add_argument('--down_sample_traj_step', default=10, type=int)
    parser.add_argument('--label', default='default')
    parser.add_argument('--deterministic', default=True)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--prop_path', default='general_full.csv', type=str)

    parser.add_argument('--EMB_way', default='train_EMB', type=str,
                        help='you can chooe train_EMB, or None')
    parser.add_argument('--tolerate', default=0.1, type=float,
                        help='use in train_EMB default 0.1')
    parser.add_argument('--use_one', default='False', type=str,
                        help='use in train_EMB, only use one data to fix missing condition or not')


    args = parser.parse_args()

    main(args)
