from typing import Any, Dict

import hydra
import math
import numpy as np
import omegaconf
import random
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from concdvae.pl_modules.model import SinusoidalPositionEmbeddings


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
    

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


class prior(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # noise for ddpm
        betas = torch.linspace(self.hparams.ddpm_noise_start, self.hparams.ddpm_noise_end, self.hparams.ddpm_n_noise)
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
            SinusoidalPositionEmbeddings(self.hparams.time_emb_dim),
            nn.Linear(self.hparams.time_emb_dim, self.hparams.time_emb_dim), nn.ReLU()
        )


        # condition model
        self.condition_model = hydra.utils.instantiate(self.hparams.conditionmodel, _recursive_=False)
        self.conditon_need = self.hparams.conditionmodel['condition_embeddings']
        condition_dim = self.hparams.conditionmodel.n_features

        # prior main model
        self.decoder = UNet(latent_dim=self.hparams.hidden_dim,
                            con_dim=condition_dim,
                            time_dim=self.hparams.time_emb_dim,
                            n_UNet_lay=self.hparams.n_UNet_layers)
        

        self._model = None
        self._refdata = None

    def forward(self, input):
        true_z, _, _ = self._model.encode(input)
        true_z = true_z.detach()

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
        # noisy_z.requires_grad_()

        pre_diff_z = self.decoder(noisy_z, condition_emb, time_emb)

        loss = F.mse_loss(noises, pre_diff_z)
        

        return loss
    

    def condition_emb_box(self, input, randan_z, ld_kwargs=None):
        need_props = [x.condition_name for x in self.conditon_need]
        if not all(prop in input.keys() for prop in need_props):
            condition_emb_list = []
            for j in range(randan_z.size(0)):
                new_input = input.copy()
                choice_traindata_idx = random.choice(range(len(self._refdata)))
                for prop in need_props:
                    if prop not in input.keys():
                        if str(ld_kwargs.use_one).lower() == 'true':
                            choice_traindata_idx = random.choice(self._refdata.axes[0])
                        condition_value = self._refdata[prop][choice_traindata_idx]
                        condition_value = torch.Tensor([condition_value])
                        condition_value = condition_value.to(randan_z.device)
                        new_input.update({prop: condition_value})
            
                condition_emb = self.condition_model(new_input)
                condition_emb_list.append(condition_emb)
            
            condition_emb = torch.cat(condition_emb_list, dim=0)
        else:
            condition_emb = self.condition_model(input)
            condition_emb = condition_emb.repeat(randan_z.size(0), 1).float()
        return condition_emb

    def gen(self, input, randan_z, ld_kwargs=None):
        # assuming the input have not loss props
        condition_emb = self.condition_emb_box(input, randan_z, ld_kwargs)
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


    def compute_stats(self, batch, outputs, prefix):
        loss = outputs
        log_dict = {
            f'{prefix}_loss': loss,
        }
        return log_dict, loss


    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss