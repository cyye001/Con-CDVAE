import time
import argparse

import pandas as pd
import torch
import hydra
import random
import yaml
import numpy as np
import sys
import math
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)
import joblib

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch
import torch.nn as nn
from hydra.experimental import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F


from eval_utils import load_model
from concdvae.common.data_utils import GaussianDistance
from concdvae.common.utils import param_statistics
from concdvae.PT_train.training import AverageMeter
from concdvae.pl_data.dataset import AtomCustomJSONInitializer, formula2atomnums

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

class condition_diff_z(nn.Module):
    def __init__(self, cfg, ld_kwargs) -> None:
        super(condition_diff_z, self).__init__()
        self.cfg = cfg
        self.ld_kwargs = ld_kwargs

        # noise for ddpm
        betas = torch.linspace(self.ld_kwargs.ddpm_noise_start, self.ld_kwargs.ddpm_noise_end, self.ld_kwargs.ddpm_n_noise)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.betas = nn.Parameter(betas, requires_grad=False)
        self.sqrt_recip_alphas = nn.Parameter(sqrt_recip_alphas, requires_grad=False)
        self.sqrt_alphas_cumprod = nn.Parameter(sqrt_alphas_cumprod, requires_grad=False)
        self.sqrt_one_minus_alphas_cumprod = nn.Parameter(sqrt_one_minus_alphas_cumprod, requires_grad=False)
        self.posterior_variance = nn.Parameter(posterior_variance, requires_grad=False)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.ld_kwargs.time_emb_dim),
            nn.Linear(self.ld_kwargs.time_emb_dim, self.ld_kwargs.time_emb_dim), nn.ReLU()
        )

        if ld_kwargs.new_condition==None:
            self.condition_model = hydra.utils.instantiate(self.cfg.model.conditionmodel, _recursive_=False)
            self.conditon_need = self.cfg.model.conditionmodel['condition_embeddings']
            condition_dim = self.cfg.model.conditionmodel.n_features
        else:
            self.condition_model = hydra.utils.instantiate(ld_kwargs.new_condition, _recursive_=False)
            self.conditon_need = ld_kwargs.new_condition['condition_embeddings']
            condition_dim = ld_kwargs.new_condition['n_features']


        self.decoder = UNet(latent_dim=self.cfg.model.hidden_dim,
                            con_dim=condition_dim,
                            time_dim=self.ld_kwargs.time_emb_dim,
                            n_UNet_lay=self.ld_kwargs.n_UNet_lay)

        self.device = 'cpu'


    def forward(self, input, true_z):
        condition_emb = self.condition_model(input)

        # sample noise levels.
        noise_level = torch.randint(0, self.betas.size(0), (input.num_atoms.size(0),), device=self.device).long()#~~
        used_sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[noise_level]#.repeat_interleave(batch.num_atoms, dim=0)
        used_sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[noise_level]#.repeat_interleave(batch.num_atoms, dim=0)
        time_emb = self.time_mlp(noise_level)

        # add noise
        noises = torch.randn_like(true_z.float())
        used_sqrt_alpha_cumprod = used_sqrt_alpha_cumprod.unsqueeze(1).expand(true_z.size(0), true_z.size(1))
        used_sqrt_one_minus_alpha_cumprod = used_sqrt_one_minus_alpha_cumprod.unsqueeze(1).expand(true_z.size(0), true_z.size(1))
        noisy_z = used_sqrt_alpha_cumprod * true_z + used_sqrt_one_minus_alpha_cumprod * noises

        pre_diff_z = self.decoder(noisy_z, condition_emb, time_emb)


        return noises, pre_diff_z

    def gen(self,input, randan_z):
        if self.ld_kwargs.EMB_way=='train_EMB':
            train_data = self.pre_fromtrain()
            condition_emb = self.fromtrain_EMB(input, randan_z, train_data)
        else:
            condition_emb = self.condition_EMB(input, randan_z)
        input_z = randan_z
        for noise_level in tqdm(range(self.betas.shape[0] - 1, -1, -1), total=self.betas.shape[0], disable=True):
            noise_level_tensor = [noise_level] * randan_z.size(0)
            noise_level_tensor = torch.tensor(noise_level_tensor, device=randan_z.device)
            time_emb = self.time_mlp(noise_level_tensor)

            diff_z = self.decoder(input_z, condition_emb, time_emb)

            z_mean = self.sqrt_recip_alphas[noise_level] * \
                         (input_z - self.betas[noise_level] * diff_z / self.sqrt_one_minus_alphas_cumprod[noise_level])

            if noise_level != 0:
                noise = torch.randn_like(diff_z)
                input_z = z_mean + torch.sqrt(self.posterior_variance[noise_level]) * noise

        output_z = z_mean

        return output_z


    def pre_fromtrain(self):
        data_root = os.path.join(self.ld_kwargs.data_root, 'train.csv')
        traindata = pd.read_csv(data_root)
        
        return traindata


    def fromtrain_EMB(self, input, randan_z, traindata):
        # chack if full condition
        full_condition = True
        need_idxs = []
        have_idxs = []
        for i in range(len(self.conditon_need)):
            if self.conditon_need[i]['condition_name'] not in input.keys():
                full_condition = False
                need_idxs.append(i)
            else:
                have_idxs.append(i)

        # if full condition
        if full_condition:
            # print('full_condition')
            condition_emb = self.condition_model(input)
            condition_emb = condition_emb.repeat(randan_z.size(0), 1).float()
            return condition_emb
        else:
            # print('need idx', need_idxs)
            # print('have idx', have_idxs)
            train_data = traindata.copy()
            for idx in have_idxs:
                if self.conditon_need[idx]['condition_name'] == 'formula':
                    continue
                elif self.conditon_need[idx]['_target_'] == 'concdvae.pl_modules.ConditionModel.ClassConditionEmbedding':
                    value = input[self.conditon_need[idx]['condition_name']].to('cpu').item()
                    value = int(value)
                    conditon = train_data[self.conditon_need[idx]['condition_name']] == value
                    train_data = train_data[conditon]
                    # print('condition name',self.conditon_need[idx]['condition_name'],',value',value)
                elif self.conditon_need[idx]['_target_'] == 'concdvae.pl_modules.ConditionModel.ScalarConditionEmbedding':
                    value = input[self.conditon_need[idx]['condition_name']].to('cpu').item()
                    change_value = self.ld_kwargs.tolerate/2 * (self.conditon_need[idx]['condition_max']-self.conditon_need[idx]['condition_min'])
                    value_min = value - change_value
                    value_max = value + change_value
                    conditon = train_data[self.conditon_need[idx]['condition_name']] >= value_min
                    train_data = train_data[conditon]
                    conditon = train_data[self.conditon_need[idx]['condition_name']] <= value_max
                    train_data = train_data[conditon]
                    # print('condition name',self.conditon_need[idx]['condition_name'],',max',value_max,',min',value_min)

            #chack the len of tain_Data
            num = len(train_data['formula'])
            if num < 1:
                print('use a larger tolerate!!!')
                sys.exit()
            else:
                print('after condition there are ',num,' item in train_Data')

            # use train_data to random choice the missing condition
            condition_emb_list = []
            # random fix for every crystal
            for j in range(randan_z.size(0)):
                new_input = input.copy()
                if self.ld_kwargs.use_one=='True': #use the same traindata to fix the missing condition
                    print('use the same traindata to fix')
                    choice_traindata_idx = random.choice(train_data.axes[0])
                    choice_traindata_idxs = [choice_traindata_idx]*len(need_idxs)
                else:  # use many traindata to fix
                    choice_traindata_idxs = [random.choice(train_data.axes[0]) for idx in need_idxs]
                print(choice_traindata_idxs)

                for idx, choice_idx in zip(need_idxs,choice_traindata_idxs):
                    condition_name = self.conditon_need[idx]['condition_name']
                    if condition_name != 'formula':
                        condition_value = train_data[condition_name][choice_idx]
                        condition_value = torch.Tensor([condition_value])
                        condition_value = condition_value.to(randan_z.device)
                        new_input.update({condition_name: condition_value})
                    else:
                        condition_value = train_data[condition_name][choice_idx]
                        atom_list = formula2atomnums(condition_value)
                        atom_fea = np.vstack([self.ld_kwargs.ari.get_atom_fea(atom_list[k])
                                              for k in range(len(atom_list))])
                        atom_fea = torch.Tensor(atom_fea)
                        # atom_fea = torch.sum(atom_fea, dim=0)
                        atom_fea = torch.mean(atom_fea, dim=0)
                        atom_fea = atom_fea.reshape(1, 92)
                        atom_fea = atom_fea.to(randan_z.device)
                        new_input.update({condition_name: atom_fea})

                condition_emb = self.condition_model(new_input)
                condition_emb_list.append(condition_emb)

            condition_emb = torch.cat(condition_emb_list, dim=0)
            return condition_emb


    def condition_EMB(self, input, randan_z):
        # chack if full condition
        full_condition = True
        a = randan_z.size(0)
        need_idxs = []
        for i in range(len(self.conditon_need)):
            if self.conditon_need[i]['condition_name'] not in input.keys():
                full_condition = False
                need_idxs.append(i)

        # if full condition
        if full_condition:
            print('full_condition')
            condition_emb = self.condition_model(input)
            condition_emb = condition_emb.repeat(randan_z.size(0), 1).float()
            return condition_emb
        else:
            print('need idx', need_idxs)
            condition_emb_list = []
            # random fix for every crystal
            for j in range(randan_z.size(0)):
                new_input = input.copy()
                for idx in need_idxs:
                    if self.conditon_need[idx]['_target_'] == 'concdvae.pl_modules.ConditionModel.ClassConditionEmbedding':
                        types = list(range(self.conditon_need[idx]['n_type']))
                        type = torch.Tensor([random.choice(types)])
                        type = type.to(randan_z.device)
                        new_input.update({self.conditon_need[idx]['condition_name']: type})
                    elif self.conditon_need[idx]['_target_'] == 'concdvae.pl_modules.ConditionModel.ScalarConditionEmbedding':
                        condition_min = self.conditon_need[idx]['condition_min']
                        condition_max = self.conditon_need[idx]['condition_max']
                        random_con = random.uniform(condition_min, condition_max)
                        random_con = torch.Tensor([random_con])
                        random_con = random_con.to(randan_z.device)
                        new_input.update({self.conditon_need[idx]['condition_name']: random_con})
                    elif self.conditon_need[idx]['condition_name'] == 'formular':
                        if 'n_atom' not in new_input.keys():
                            n_atom = list(range(1,20))
                            n_atom = random.choice(n_atom)
                        else:
                            n_atom = new_input['n_atom'].tolist()
                            n_atom = int(n_atom[0])
                            if n_atom == 0:
                                n_atom += 1
                        atom_list = []
                        for k in range(n_atom):
                            all_atom = list(range(1,82))
                            atom_list.append(random.choice(all_atom))
                        atom_fea = np.vstack([self.ld_kwargs.ari.get_atom_fea(atom_list[k])
                                              for k in range(len(atom_list))])
                        atom_fea = torch.Tensor(atom_fea)
                        atom_fea = torch.mean(atom_fea, dim=0)
                        atom_fea = atom_fea.reshape(1, 92)
                        atom_fea = atom_fea.to(randan_z.device)
                        new_input.update({'formular': atom_fea})


                condition_emb = self.condition_model(new_input)
                condition_emb_list.append(condition_emb)

            condition_emb = torch.cat(condition_emb_list, dim=0)
            return condition_emb


class UNet(nn.Module):
    def __init__(self, latent_dim=256,con_dim=128,time_dim=128,n_UNet_lay=3) -> None:
        super(UNet, self).__init__()
        self.latent_dim = latent_dim
        self.con_dim = con_dim
        self.time_dim = time_dim
        self.n_UNet_lay = n_UNet_lay
        down_list = []
        down_act = []
        up_list = []
        up_act = []
        now_dim = latent_dim
        for n_lay in range(self.n_UNet_lay):
            assert (now_dim / 2) >2
            out_dim = int(now_dim / 2)
            down_list.append(nn.Linear(now_dim+self.con_dim+self.time_dim, out_dim))
            down_act.append(nn.ReLU())
            up_list.append(nn.Linear(out_dim+self.con_dim+self.time_dim, now_dim))
            up_act.append(nn.ReLU())
            now_dim = out_dim
        self.downmodel = nn.ModuleList(down_list)
        self.downact = nn.ModuleList(down_act)
        self.upmodel = nn.ModuleList(up_list[::-1])
        self.upact = nn.ModuleList(up_act)
        self.middle_mpl = nn.Linear(now_dim+self.con_dim+self.time_dim, out_dim)
        self.middle_act = nn.ReLU()
        self.output_mlp = nn.Linear(latent_dim+self.con_dim+self.time_dim, latent_dim)

    def forward(self,latent, condition, time):
        input = torch.cat((latent, condition, time), dim=1)
        # down
        down_out_list = []
        for i in range(len(self.downmodel)):
            input = self.downact[i](self.downmodel[i](input))
            down_out_list.append(input)
            input = torch.cat((input, condition, time), dim=1)
        down_out_list = down_out_list[::-1]

        input = self.middle_act(self.middle_mpl(input))

        for i in range(len(self.upmodel)):
            input = input + down_out_list[i]
            input = torch.cat((input, condition, time), dim=1)
            input = self.upact[i](self.upmodel[i](input))

        input = input + latent
        input = torch.cat((input, condition, time), dim=1)
        output = self.output_mlp(input)

        return output



def main(args):
    best_loss = 10000000
    if args.deterministic:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    model_path = args.model_path # Path(args.model_path)
    initialize_config_dir(config_dir=model_path)
    cfg: DictConfig = compose(config_name="hparams")
    if args.fullfea == 1:
        print('use full feature')
        cfg.data.root_path = '${oc.env:PROJECT_ROOT}/data/'+args.newdata
        cfg.data.prop = ['bandgap','formation','e_above_hull','a','b','c','alpha','beta','gamma','density',
                         'coor_number','n_atom','spacegroup','crystal_system']
        cfg.data.use_prop = 'formation'
        condition_root = args.newcond

        with open(condition_root, 'r') as file:
            new_condition = yaml.safe_load(file)
    else:
        print('use default feature')
        new_condition = None
    if torch.cuda.is_available():
        cfg.accelerator = 'gpu'
        cfg.data.datamodule.accelerator = 'gpu'
        print('use gpu')
    else:
        cfg.accelerator = 'cpu'
        cfg.data.datamodule.accelerator = 'cpu'
        print('use cpu')



    ld_kwargs = SimpleNamespace(fc_num_layers=args.fc_num_layers,
                                hidden_dim=args.hidden_dim,
                                resnet=args.resnet,
                                ddpm_noise_start=args.ddpm_noise_start,
                                ddpm_noise_end=args.ddpm_noise_end,
                                ddpm_n_noise=args.ddpm_n_noise,
                                time_emb_dim=args.time_emb_dim,
                                n_UNet_lay=args.n_UNet_lay,
                                new_condition=new_condition,
                                )

    conz_model = condition_diff_z(cfg, ld_kwargs)


    model = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    model_root = Path(model_path) / args.model_file
    checkpoint = torch.load(model_root, map_location=torch.device('cpu'))
    model_state_dict = checkpoint['model']
    model.load_state_dict(model_state_dict)
    lattice_scaler = torch.load(Path(model_path) / 'lattice_scaler.pt')
    model.lattice_scaler = lattice_scaler
    for param in model.parameters():
        param.requires_grad = False

    print('Load model', file=sys.stdout)

    cfg.data.datamodule.batch_size.train=args.batch_size
    cfg.data.datamodule.batch_size.val=args.batch_size
    cfg.data.datamodule.batch_size.test=args.batch_size
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)

    print('Load data', file=sys.stdout)
    sys.stdout.flush()

    print('model param:')
    param_statistics(model)
    print('conz_model param:')
    param_statistics(conz_model)

    if torch.cuda.is_available():
        conz_model.to('cuda')
        model.to('cuda')
        model.device = 'cuda'
        conz_model.device = 'cuda'

    optimizer = torch.optim.Adam(lr=args.step_lr,
                                 betas=[ 0.9, 0.999 ],
                                 eps=1e-08,
                                 weight_decay=0,
                                 params=conz_model.parameters(),)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(factor=args.factor,
                                                           patience=args.patience,
                                                           min_lr=args.min_lr,
                                                           optimizer=optimizer)

    print('start train', file=sys.stdout)

    train_loss_epoch = []
    val_loss_epoch = []
    for epoch in range(args.epochs):
        end = time.time()
        train_loss = AverageMeter()
        batch_time = AverageMeter()
        model.train()
        for i, batch in enumerate(datamodule.train_dataloader):
            if torch.cuda.is_available():
                batch = batch.cuda()

            true_mu, true_log_var, true_z = model.encode(batch)
            noisy_z, diff_z = conz_model(batch, true_mu)
            loss = mse_loss(noisy_z, diff_z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.data.cpu(), diff_z.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.train.PT_train.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time}\t'
                      'Loss {train_loss}'.format(epoch, i, len(datamodule.train_dataloader),
                                                 batch_time=batch_time,
                                                 train_loss=train_loss), file=sys.stdout)
                sys.stdout.flush()

        batch_time = AverageMeter()
        val_loss = AverageMeter()
        model.eval()
        for i, batch in enumerate(datamodule.val_dataloaders[0]):
            if torch.cuda.is_available():
                batch = batch.cuda()

            true_mu, true_log_var, true_z = model.encode(batch)
            noisy_z, diff_z = conz_model(batch, true_mu)
            loss = mse_loss(noisy_z, diff_z)
            val_loss.update(loss.data.cpu(), noisy_z.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.train.PT_train.print_freq == 0:
                print('{3}: [{0}][{1}/{2}]\t'
                      'Time {batch_time}\t'
                      'Loss {val_loss}'.format(epoch, i, len(datamodule.val_dataloaders[0]), 'val',
                                                 batch_time=batch_time,
                                                 val_loss=val_loss), file=sys.stdout)
                sys.stdout.flush()

        scheduler.step(metrics=val_loss.avg)
        if (val_loss.avg < best_loss):
            best_loss = val_loss.avg
            filename = 'conz_model_' + args.label + '_diffu' + '.pth'
            path = Path(model_path) / filename
            data = {'model': conz_model.state_dict(),
                    'epoch': epoch + 1,
                    'val_loss': val_loss.avg,
                    'args': vars(args)}
            torch.save(data, path)
            print('save model with loss = ', val_loss, file=sys.stdout)

        train_loss_epoch.append(train_loss.avg.cpu().detach().numpy())
        val_loss_epoch.append(val_loss.avg.cpu().detach().numpy())

    loss_dict = {
        'train_loss_epoch': train_loss_epoch,
        'val_loss_epoch': val_loss_epoch, }
    loss_df = pd.DataFrame(loss_dict)
    filename = 'conz_loss_file_' + args.label + '.xlsx'
    excel_file = Path(model_path) / filename
    loss_df.to_excel(excel_file, index=False)
    print('end')


def mse_loss(noisy_z, diff_z):
    loss1 = F.mse_loss(noisy_z, diff_z)
    return loss1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model_file', default='model_test.pth', type=str)
    parser.add_argument('--step_lr', default=1e-3, type=float)
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--factor', default=0.6, type=float)
    parser.add_argument('--patience', default=30, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--deterministic', default=True)
    parser.add_argument('--seed', default=456, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--fc_num_layers', default=2, type=int)
    parser.add_argument('--resnet', default=True)
    parser.add_argument('--label', default='ABC', type=str)
    parser.add_argument('--fullfea', default=1, type=int)
    parser.add_argument('--newdata', default='mptest4conz', type=str)
    parser.add_argument('--newcond', default='conz_2.yaml', type=str)

    parser.add_argument('--ddpm_n_noise', default=300, type=int)
    parser.add_argument('--ddpm_noise_start', default=0.001, type=float)
    parser.add_argument('--ddpm_noise_end', default=0.02, type=float)
    parser.add_argument('--time_emb_dim', default=64, type=int)
    parser.add_argument('--n_UNet_lay', default=3, type=int)

    args = parser.parse_args()

    main(args)