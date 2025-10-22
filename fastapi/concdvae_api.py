import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)

from typing import Dict
import torch
from pathlib import Path
import pandas as pd
import asyncio
from fastapi import FastAPI

from scripts.eval_utils  import load_model
from concdvae.common.utils import PROJECT_ROOT
from utils_api import full_pre, tensor2cif
from scripts.eval_utils import generation
from concdvae.pl_data.dataset import AtomCustomJSONInitializer
# from common.utils import load_model, load_conz_models, gen_with_prop_z, default_genz_with_prop, full_pre, formula2atomnums
# from common.utils import incomplete_prop_to_z_con, gen_with_z_con
# from common.data_utils import AtomCustomJSONInitializer
from types import SimpleNamespace

app = FastAPI()

model_path = os.path.join(PROJECT_ROOT, 'src/model/mp20_format')#'/data/work/cyye/0-project/15-con_cdvae/new_repo/src/model/mp20_format'
model_file = 'epoch=330-step=17543.ckpt'
default_prior_file = 'prior_default-epoch=95-step=10176.ckpt'
full_prior_file = 'prior_full-epoch=95-step=10176.ckpt'
# os.environ['PROJECT_ROOT'] = model_path
lock = asyncio.Lock()

# model, cfg = load_model(model_path, model_file)
model, _, _ = load_model(model_path, model_file)
default_prior, _, _ = load_model(model_path, default_prior_file, prior_label='prior_default')
full_prior, _, cfg_train = load_model(model_path, full_prior_file, prior_label='prior_full')


# default_conz_model, full_conz_model = load_conz_models(model_path, default_conz_file, full_conz_file, cfg)

# n_step_each = 1          #for debug in CPU
# disable_bar = False
if torch.cuda.is_available():
    model.to('cuda')
    if default_prior != None:
        default_prior.to('cuda')
    if full_prior != None:
        full_prior.to('cuda')
    n_step_each = 50
    disable_bar = True


for param in model.parameters():
        param.requires_grad = False
for param in default_prior.parameters():
    param.requires_grad = False
for param in full_prior.parameters():
    param.requires_grad = False
model.eval()
default_prior.eval()
full_prior.eval()

atom_init_file = os.path.join(PROJECT_ROOT, 'cgcnn/data/mptest/atom_init.json')
ari = AtomCustomJSONInitializer(atom_init_file)


# for param in model.parameters():
#     param.requires_grad = False
# for param in default_conz_model.parameters():
#     param.requires_grad = False
# for param in full_conz_model.parameters():
#     param.requires_grad = False
# ld_kwargs = SimpleNamespace(n_step_each=n_step_each, ###
#                             step_lr=1e-4,
#                             min_sigma=0,
#                             save_traj=False,
#                             disable_bar=False)

# atom_init_file = os.path.join(model_path, 'atom_init.json')
# ari = AtomCustomJSONInitializer(atom_init_file)

# data_root = os.path.join(model_path, 'train.csv')
# traindata = pd.read_csv(data_root)

@app.get("/")
def read_root():
    intro = 'Load model from '+model_file+', the model in running in '+ str(model.device)
    return {"Introdution": intro}


# @app.get("/random/{n_cry}")
# def random_crystal(n_cry):
#     if n_cry == 'nan':
#         n_cry = 1
#     try:
#         n_cry = int(n_cry)
#     except Exception as e:
#         return {'wrong': e}

#     # random material_z
#     randan_z = torch.randn(n_cry, model.hparams.latent_dim, device=model.device)

#     # prop from material_z
#     model_props = []
#     props_value = []
#     for i in range(len(model.conditions_name)):
#         model_props.append(model.conditions_name[i])
#         pre = model.conditions_predict[i].mlp(randan_z)
#         if pre.size(-1) == 1:  ##regression
#             a = model.conditions_predict[i].condition_max
#             b = model.conditions_predict[i].condition_min
#             pre = pre * (a-b) + b
#             pre = pre.reshape(pre.size(0),)
#         else:  ##classification
#             #TODO no classification
#             print('error')

#         props_value.append(pre)

#     if torch.cuda.is_available():
#         prop_dict = {k:v.cuda() for k, v in zip(model_props, props_value)}
#     else:
#         prop_dict = {k: v for k, v in zip(model_props, props_value)}

#     # gen with prop and material_z
#     cif_list = gen_with_prop_z(model, prop_dict, randan_z, ld_kwargs)

#     return {'cif_list': cif_list}


# @app.get("/bgfe/{n_cry}/{bg}/{fe}")
# def default_bgfe_crystal(n_cry, bg, fe):
#     if n_cry == 'nan':
#         n_cry = 1
#     try:
#         n_cry = int(n_cry)
#         bg = float(bg)
#         fe = float(fe)
#     except Exception as e:
#         return {'wrong': e}

#     if torch.cuda.is_available():
#         prop_dict = {'bandgap':torch.Tensor([bg]).float().cuda(),'formation':torch.Tensor([fe]).float().cuda(),
#                      'band_gap':torch.Tensor([bg]).float().cuda(),'formation_energy_per_atom':torch.Tensor([fe]).float().cuda()}
#     else:
#         prop_dict = {'bandgap': torch.Tensor([bg]).float(), 'formation': torch.Tensor([fe]).float(),
#                      'band_gap': torch.Tensor([bg]).float(), 'formation_energy_per_atom': torch.Tensor([fe]).float()}

#     material_z = default_genz_with_prop(model, default_conz_model, prop_dict, n_cry)

#     cif_list = gen_with_prop_z(model, prop_dict, material_z, ld_kwargs)

#     return {'cif_list': cif_list}


# # @app.get("/fullbgfe/{n_cry}/{bg}/{fe}/{n_atom}/{formula}")
# # async def full_bgfe_crystal(n_cry, bg, fe, n_atom, formula):
# #     loop = asyncio.get_running_loop()
# #     warnings = []
# #     output = {}
# #     n_cry, prop_dict = full_pre(n_cry, bg, fe, n_atom, formula, ari, model.device)
# #     if 'err' in prop_dict:
# #         return {'err': prop_dict['err']}
# #     if 'warning' in prop_dict:
# #         warnings.extend(prop_dict['warning'])

# #     z_con = await loop.run_in_executor(None, incomplete_prop_to_z_con, model, full_conz_model, prop_dict, traindata, n_cry, ari) 
# #     if z_con=='err':
# #         return {'err': 'This model can not deal with this condition, please try other input. This may happen when you set the input too big or too small.'}

# #     cif_list = await loop.run_in_executor(None, gen_with_z_con, model, z_con, prop_dict, ld_kwargs)
# #     output.update({'cif_list': cif_list})

# #     if len(warnings)>0:
# #         output.update({'warning':warnings})

# #     return output


# @app.get("/fullbgfe/{n_cry}/{bg}/{fe}/{n_atom}/{formula}")
# async def full_bgfe_crystal(n_cry, bg, fe, n_atom, formula):
#     warnings = []
#     output = {}
#     n_cry, prop_dict = full_pre(n_cry, bg, fe, n_atom, formula, ari, model.device)
#     if 'err' in prop_dict:
#         return {'err': prop_dict['err']}
#     if 'warning' in prop_dict:
#         warnings.extend(prop_dict['warning'])

#     z_con = incomplete_prop_to_z_con(model, full_conz_model, prop_dict, traindata, n_cry, ari) 
#     if z_con=='err':
#         return {'err': 'This model can not deal with this condition, please try other input. This may happen when you set the input too big or too small.'}
#     async with lock:  # 确保每次只有一个请求使用 GPU
#         cif_list = gen_with_z_con(model, z_con, prop_dict, ld_kwargs)
#     output.update({'cif_list': cif_list})

#     if len(warnings)>0:
#         output.update({'warning':warnings})

#     return output


@app.post("/fullfe_post/")
def full_fe_crystal_post(data: Dict):
    warnings = []
    output = {}
    if 'n_cry' in data.keys():
        n_cry = int(data['n_cry'])
    else:
        n_cry = 'None'
#     if 'bg' in data.keys():
#         bg = float(data['bg'])
#     else:
#         bg = 'None'
    if 'fe' in data.keys():
        fe = float(data['fe'])
    else:
        fe = 'None'
    if 'n_atom' in data.keys():
        n_atom = int(data['n_atom'])
    else:
        n_atom = 'None'
    if 'formula' in data.keys():
        formula = str(data['formula'])
    else:
        formula = 'None'
    n_cry, prop_dict = full_pre(n_cry, fe, n_atom, formula, ari, model.device)
    # print(prop_dict)

    if 'err' in prop_dict:
        return {'err': prop_dict['err']}
    if 'warning' in prop_dict:
        warnings.extend(prop_dict['warning'])

    ld_kwargs = SimpleNamespace(num_samples_per_z=1,
                                save_traj=False,
                                down_sample_traj_step=1,
                                use_one=True,
                                disable_bar=False,
                                min_sigma=0.0,
                                step_lr=1e-4 ,
                                n_step_each=50,)
    
    (frac_coords, num_atoms, atom_types, lengths, angles,
    all_frac_coords_stack, all_atom_types_stack) = generation(model, default_prior, prop_dict ,
                    batch_size=n_cry, 
                    down_sample=1, 
                    num_batches_to_sample=1,
                    ld_kwargs=ld_kwargs)

    # frac_coords = torch.cat(frac_coords, dim=1)
    # num_atoms = torch.cat(num_atoms, dim=1)
    # atom_types = torch.cat(atom_types, dim=1)
    # lengths = torch.cat(lengths, dim=1)
    # angles = torch.cat(angles, dim=1)

    cif_list = tensor2cif(frac_coords, num_atoms, atom_types, lengths, angles)
#     z_con = incomplete_prop_to_z_con(model, full_conz_model, prop_dict, traindata, n_cry, ari)
#     if z_con=='err':
#         return {'err': 'This model can not deal with this condition, please try other input. This may happen when you set the input too big or too small.'}

#     cif_list = gen_with_z_con(model, z_con, prop_dict, ld_kwargs)
    output.update({'cif_list': cif_list})

    if len(warnings)>0:
        output.update({'warning':warnings})

    return output

if __name__ == "__main__":
    data = {'n_cry':'2',
            'fe':-3.5,
            'n_atom':4,
            'formula':'Li2O2'}
    a = full_fe_crystal_post(data)
    print(a)
#     # a = random_crystal(4)
#     # print(a)
#     #
#     # a = default_bgfe_crystal(4,1,0)
#     # print(a)

#     a = full_bgfe_crystal(1, 1, 'None', 4000, 'None')
#     print(a)