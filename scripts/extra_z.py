import time
import argparse
import torch
import hydra
import random
import numpy as np
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch


from eval_utils import load_model
from concdvae.common.data_utils import GaussianDistance


def main(args):
    print('strat with data:', args.data)
    model_path = Path(args.model_path)
    # load_data if do reconstruction.
    model, test_loader, cfg = load_model(args.model_path, args.model_file,
                                         load_data=True)
    if torch.cuda.is_available():
        cfg.data.datamodule.accelerator='gpu'
    else:
        cfg.data.datamodule.accelerator = 'cpu'

    if(args.data!='test'):
        datamodule = hydra.utils.instantiate(
            cfg.data.datamodule, _recursive_=False
        )
        if(args.data=='train'):
            test_loader=datamodule.train_dataloader
        elif(args.data=='val'):
            test_loader = datamodule.val_dataloaders[0]
        else:
            print('warng in data:', args.data)


    if torch.cuda.is_available():
        model.to('cuda')
        model.device = 'cuda'

    condition_names = []
    condition_list = []
    id_list = []
    mu_list = []
    for con_emb in cfg.model.conditionmodel.condition_embeddings:
        condition_names.append(con_emb['condition_name'])
        condition_list.append([])

    for idx, batch in enumerate(test_loader):
        print(idx, 'in', len(test_loader), file=sys.stdout)
        sys.stdout.flush()
        if torch.cuda.is_available():
            batch = batch.cuda()

        mu, log_var, z = model.encode(batch)

        for i in range(len(condition_names)):
            condition = batch[condition_names[i]]
            condition = condition.cpu().detach().numpy()
            condition = list(condition)
            condition_list[i].extend(condition)
            # print('he')
        id_list.extend(batch['mp_id'])
        mu = mu.cpu().detach().numpy()
        for i in range(mu.shape[0]):
            mu_cry = mu[i,:].tolist()
            mu_list.append(mu_cry)

    condition_dict = {k:v for k, v in zip(condition_names,condition_list)}
    output = {'material_id': id_list, 'material_z': mu_list}
    output.update(condition_dict)

    outputfile = model_path / args.output_file
    torch.save(output, outputfile)
    print('end')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model_file', default='model_perov.pth', type=str)
    parser.add_argument('--output_file', default='material_z.pt', type=str)
    parser.add_argument('--data', default='test', type=str)

    args = parser.parse_args()

    main(args)