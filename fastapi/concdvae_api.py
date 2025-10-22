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
from types import SimpleNamespace

app = FastAPI()

model_path = os.path.join(PROJECT_ROOT, 'src/model/mp20_format')
model_file = 'epoch=330-step=17543.ckpt'
default_prior_file = 'prior_default-epoch=95-step=10176.ckpt'
full_prior_file = 'prior_full-epoch=95-step=10176.ckpt'

lock = asyncio.Lock()


model, _, _ = load_model(model_path, model_file)
default_prior, _, _ = load_model(model_path, default_prior_file, prior_label='prior_default')
full_prior, _, cfg_train = load_model(model_path, full_prior_file, prior_label='prior_full')


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



@app.get("/")
def read_root():
    intro = 'Load model from '+model_file+', the model in running in '+ str(model.device)
    return {"Introdution": intro}



@app.post("/fullfe_post/")
def full_fe_crystal_post(data: Dict):
    warnings = []
    output = {}
    if 'n_cry' in data.keys():
        n_cry = int(data['n_cry'])
    else:
        n_cry = 'None'
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


    cif_list = tensor2cif(frac_coords, num_atoms, atom_types, lengths, angles)
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
