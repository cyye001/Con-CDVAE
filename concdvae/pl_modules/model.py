from typing import Any, Dict

import hydra
import math
import numpy as np
import omegaconf
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from concdvae.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc
)
from concdvae.pl_modules.embeddings import MAX_ATOMIC_NUM
from concdvae.pl_modules.embeddings import KHOT_EMBEDDINGS

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for diffusion time steps.
    This is used to encode the noise level (time step) in the diffusion process.
    """
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

def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim, drop=-1, norm=True):
    """
    Build a multi-layer perceptron (MLP) with configurable parameters.
    
    Args:
        in_dim: Input dimension
        hidden_dim: Hidden layer dimension
        fc_num_layers: Number of fully connected layers
        out_dim: Output dimension
        drop: Dropout rate (if > 0 and < 1)
        norm: Whether to use batch normalization
    
    Returns:
        nn.Sequential: The constructed MLP
    """
    if norm:
        mods = [nn.Linear(in_dim, hidden_dim),nn.BatchNorm1d(num_features=hidden_dim), nn.ReLU()]
    else:
        mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        singlelist = []
        singlelist.append(nn.Linear(hidden_dim, hidden_dim))
        if drop >0 and drop < 1:
            singlelist.append(nn.Dropout(p=drop))
        singlelist.append(nn.ReLU())
        mods += singlelist
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


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


class CDVAE(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Initialize condition prediction modules
        conditions_predict = []
        self.conditions_name = []
        for pre in self.hparams.conditionpre.condition_predict:
            conditions_predict.append(hydra.utils.instantiate(pre, _recursive_=False))
            self.conditions_name.append(pre.condition_name)
        self.conditions_predict = nn.ModuleList(conditions_predict)

        # Condition model for embedding external conditions
        self.condition_model = hydra.utils.instantiate(self.hparams.conditionmodel, _recursive_=False)

        # MLP to combine latent vector with condition embeddings
        self.z_condition = build_mlp(self.hparams.latent_dim+self.hparams.conditionmodel.n_features,
                                    self.hparams.hidden_dim,
                                    self.hparams.fc_num_layers,
                                    self.hparams.latent_dim)

        # Time embedding for diffusion process
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hparams.time_emb_dim),
            nn.Linear(self.hparams.time_emb_dim, self.hparams.time_emb_dim), nn.ReLU()
        )

        # Encoder and decoder networks
        self.encoder = hydra.utils.instantiate(
            self.hparams.encoder, num_targets=self.hparams.latent_dim)
        self.decoder = hydra.utils.instantiate(self.hparams.decoder)

        # VAE components: mean and variance predictors
        self.fc_mu = nn.Linear(self.hparams.latent_dim,
                               self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.latent_dim,
                                self.hparams.latent_dim)

        # Predictors for crystal structure components
        self.fc_num_atoms = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                      self.hparams.fc_num_layers, self.hparams.max_atoms + 1)
        self.fc_lattice = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                    self.hparams.fc_num_layers, 6)
        self.fc_composition = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                        self.hparams.fc_num_layers, MAX_ATOMIC_NUM)

        # Noise schedule for coordinate diffusion
        sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.sigma_begin),
            np.log(self.hparams.sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        # Noise schedule for atom type diffusion
        type_sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.type_sigma_begin),
            np.log(self.hparams.type_sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

        # self.embedding = torch.zeros(100, 92)
        # for i in range(100):
        #     self.embedding[i] = torch.tensor(KHOT_EMBEDDINGS[i + 1])

        # obtain from datamodule.
        self.lattice_scaler = None


    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        
        Args:
            mu: Mean of the latent Gaussian [B x D]
            logvar: Log variance of the latent Gaussian [B x D]
            
        Returns:
            Sampled latent vector [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, batch):
        """
        Encode crystal structures to latent representations.
        
        Args:
            batch: Input batch containing crystal structure data
            
        Returns:
            tuple: (mu, log_var, z) - mean, log variance, and sampled latent vector
        """
        hidden = self.encoder(batch)  
        if self.hparams.smooth == True:
            hidden = torch.tanh(hidden)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    def decode_stats(self, z, gt_num_atoms=None, gt_lengths=None, gt_angles=None,
                     teacher_forcing=False):
        """
        Decode key statistics from latent embeddings.
        
        Args:
            z: Latent vector
            gt_num_atoms: Ground truth number of atoms (for teacher forcing)
            gt_lengths: Ground truth lattice lengths (for teacher forcing)
            gt_angles: Ground truth lattice angles (for teacher forcing)
            teacher_forcing: Whether to use teacher forcing
            
        Returns:
            tuple: (num_atoms, lengths_and_angles, lengths, angles, composition_per_atom)
        """
        if gt_num_atoms is not None:
            num_atoms = self.predict_num_atoms(z)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, gt_num_atoms))
            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = self.predict_num_atoms(z).argmax(dim=-1)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, num_atoms))
            composition_per_atom = self.predict_composition(z, num_atoms)
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom


    def forward(self, batch, teacher_forcing, training):
        """
        Forward pass through the CDVAE model.
        
        Args:
            batch: Input batch containing crystal structure data
            teacher_forcing: Whether to use teacher forcing
            training: Whether in training mode
            
        Returns:
            dict: Dictionary containing all losses and predictions
        """
        # Embed the conditions
        condition_emb = self.condition_model(batch)

        # hacky way to resolve the NaN issue. Will need more careful debugging later.
        # You can set smooth == True
        # Encode to get latent representation
        mu, log_var, z = self.encode(batch)

        # Use latent vector to predict conditions for better latent space
        pre_losses = {}
        for con_pre in self.conditions_predict:
            loss = con_pre(batch, z)
            pre_losses.update({con_pre.condition_name+'_loss': loss})

        # Combine latent vector with condition embeddings
        z_con = torch.cat((z,condition_emb),dim=1)
        z_con = self.z_condition(z_con)

        # Decode crystal structure statistics
        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles,
         pred_composition_per_atom) = self.decode_stats(
            z_con, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing)

        # Sample noise levels for diffusion
        noise_level = torch.randint(0, self.sigmas.size(0),
                                    (batch.num_atoms.size(0),),
                                    device=self.device)
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(
            batch.num_atoms, dim=0)

        type_noise_level = noise_level
        used_type_sigmas_per_atom = (
            self.type_sigmas[type_noise_level].repeat_interleave(
                batch.num_atoms, dim=0))

        # Add noise to atom types and sample atom types
        pred_composition_probs = F.softmax(
            pred_composition_per_atom.detach(), dim=-1)
        atom_type_probs = (
                F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM) +
                pred_composition_probs * used_type_sigmas_per_atom[:, None])
        rand_atom_types = torch.multinomial(
            atom_type_probs, num_samples=1).squeeze(1) + 1

        # Add noise to the cartesian coordinates
        cart_noises_per_atom = (
                torch.randn_like(batch.frac_coords) *
                used_sigmas_per_atom[:, None])
        cart_coords = frac_to_cart_coords(
            batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms)
        cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords(
            cart_coords, pred_lengths, pred_angles, batch.num_atoms)

        # Prevent decoder from affecting encoder if nograd is True
        if self.hparams.nograd == True: 
            z_nograd = z.detach()
            z_nograd = torch.cat((z_nograd,condition_emb),dim=1)
            z_con = self.z_condition(z_nograd)

        # Add time embedding for diffusion
        time_emb = self.time_mlp(noise_level)
        z_con_time = torch.cat((z_con, time_emb), dim=1)

        # Decode coordinate differences and atom types
        pred_cart_coord_diff, pred_atom_types = self.decoder(
            z_con_time, noisy_frac_coords, rand_atom_types, batch.num_atoms, pred_lengths, pred_angles)

        # Compute all losses
        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch) #cross
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch) #MSE
        composition_loss = self.composition_loss(
            pred_composition_per_atom, batch.atom_types, batch)  #cross
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch)  #MSE 
        type_loss = self.type_loss(pred_atom_types, batch.atom_types,  #cross  
                                   used_type_sigmas_per_atom, batch)

        kld_loss = self.kld_loss(mu, log_var)

        # Prepare output dictionary
        output = {
            'num_atom_loss': num_atom_loss,
            'lattice_loss': lattice_loss,
            'composition_loss': composition_loss,
            'coord_loss': coord_loss,
            'type_loss': type_loss,
            'kld_loss': kld_loss,
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'pred_atom_types': pred_atom_types,
            'pred_composition_per_atom': pred_composition_per_atom,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            'rand_atom_types': rand_atom_types,
            'z': z,
        }
        output.update(pre_losses)
        predict_loss = pre_losses[self.conditions_name[0]+'_loss']
        for con_name in self.conditions_name:
            if con_name != self.conditions_name[0]:
                predict_loss += pre_losses[con_name+'_loss']
        output.update({'predict_loss': predict_loss})

        return output

    def predict_num_atoms(self, z):
        """Predict number of atoms from latent vector."""
        return self.fc_num_atoms(z)

    def predict_lattice(self, z, num_atoms):
        """
        Predict lattice parameters (lengths and angles) from latent vector.
        
        Args:
            z: Latent vector
            num_atoms: Number of atoms per structure
            
        Returns:
            tuple: (lengths_and_angles, lengths, angles)
        """
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.hparams.data.lattice_scale_method == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float() ** (1 / 3)
        # <pred_lengths_and_angles> is scaled.
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_composition(self, z, num_atoms):
        """
        Predict atomic composition from latent vector.
        
        Args:
            z: Latent vector
            num_atoms: Number of atoms per structure
            
        Returns:
            Composition probabilities per atom
        """
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom

    def num_atom_loss(self, pred_num_atoms, batch):
        """Cross-entropy loss for number of atoms prediction."""
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)

    def property_loss(self, z, batch):
        """MSE loss for property prediction."""
        return F.mse_loss(self.fc_property(z), batch.y)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        """
        MSE loss for lattice parameter prediction.
        
        Args:
            pred_lengths_and_angles: Predicted lattice parameters
            batch: Ground truth batch data
            
        Returns:
            MSE loss between predicted and target lattice parameters
        """
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        if self.hparams.data.lattice_scale_method == 'scale_length':
            target_lengths = batch.lengths / \
                             batch.num_atoms.view(-1, 1).float() ** (1 / 3)
        target_lengths_and_angles = torch.cat(
            [target_lengths, batch.angles], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        """
        Cross-entropy loss for composition prediction.
        
        Args:
            pred_composition_per_atom: Predicted composition probabilities
            target_atom_types: Target atom types
            batch: Batch data for scatter operation
            
        Returns:
            Mean cross-entropy loss per structure
        """
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        return scatter(loss, batch.batch, reduce='mean').mean()

    def coord_loss(self, pred_cart_coord_diff, noisy_frac_coords,
                   used_sigmas_per_atom, batch):
        """
        MSE loss for coordinate prediction in diffusion process.
        
        Args:
            pred_cart_coord_diff: Predicted coordinate differences
            noisy_frac_coords: Noisy fractional coordinates
            used_sigmas_per_atom: Noise levels per atom
            batch: Ground truth batch data
            
        Returns:
            Weighted MSE loss for coordinate prediction
        """
        noisy_cart_coords = frac_to_cart_coords(
            noisy_frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        target_cart_coords = frac_to_cart_coords(
            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords, noisy_cart_coords, batch.lengths, batch.angles,
            batch.num_atoms, self.device, return_vector=True)

        target_cart_coord_diff = target_cart_coord_diff / \
                                 used_sigmas_per_atom[:, None] ** 2
        pred_cart_coord_diff = pred_cart_coord_diff / \
                               used_sigmas_per_atom[:, None]

        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff) ** 2, dim=1)

        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom ** 2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def type_loss(self, pred_atom_types, target_atom_types,
                  used_type_sigmas_per_atom, batch):
        """
        Cross-entropy loss for atom type prediction in diffusion process.
        
        Args:
            pred_atom_types: Predicted atom type probabilities
            target_atom_types: Target atom types
            used_type_sigmas_per_atom: Noise levels for atom types
            batch: Batch data for scatter operation
            
        Returns:
            Weighted cross-entropy loss for atom type prediction
        """
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_atom_types, target_atom_types, reduction='none')
        # rescale loss according to noise
        loss = loss / used_type_sigmas_per_atom
        return scatter(loss, batch.batch, reduce='mean').mean()

    def kld_loss(self, mu, log_var):
        """
        KL divergence loss for VAE regularization.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            KL divergence loss
        """
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss


    def compute_stats(self, batch, outputs, prefix):
        """
        Compute training/validation statistics and losses.
        
        Args:
            batch: Input batch
            outputs: Model outputs
            prefix: Prefix for logging ('train', 'val', or 'test')
            
        Returns:
            tuple: (log_dict, loss) - logging dictionary and total loss
        """
        num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        coord_loss = outputs['coord_loss']
        type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        composition_loss = outputs['composition_loss']
        predict_loss = outputs['predict_loss']
        if not self.hparams.predict_property:
            predict_loss = predict_loss * 0.0

        # Compute total loss with weighted components
        loss = (
                self.hparams.cost_natom * num_atom_loss +
                self.hparams.cost_lattice * lattice_loss +
                self.hparams.cost_coord * coord_loss +
                self.hparams.cost_type * type_loss +
                self.hparams.beta * kld_loss +
                self.hparams.cost_composition * composition_loss +
                self.hparams.cost_property * predict_loss)

        # Prepare logging dictionary
        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_type_loss': type_loss,
            f'{prefix}_kld_loss': kld_loss,
            f'{prefix}_composition_loss': composition_loss,
            f'{prefix}_predict_loss': predict_loss,
        }

        if prefix != 'train':
            # Validation/test loss only has coord and type
            loss = (
                    self.hparams.cost_coord * coord_loss +
                    self.hparams.cost_type * type_loss)

            # Evaluate num_atom prediction accuracy
            pred_num_atoms = outputs['pred_num_atoms'].argmax(dim=-1)
            num_atom_accuracy = (
                                        pred_num_atoms == batch.num_atoms).sum() / batch.num_graphs

            # Evaluate lattice prediction accuracy
            pred_lengths_and_angles = outputs['pred_lengths_and_angles']
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles)
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * \
                               batch.num_atoms.view(-1, 1).float() ** (1 / 3)
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)

            # Evaluate atom type prediction accuracy
            pred_atom_types = outputs['pred_atom_types']
            target_atom_types = outputs['target_atom_types']
            type_accuracy = pred_atom_types.argmax(
                dim=-1) == (target_atom_types - 1)
            type_accuracy = scatter(type_accuracy.float(
            ), batch.batch, dim=0, reduce='mean').mean()

            log_dict.update({
                f'{prefix}_loss': loss,
                f'{prefix}_natom_accuracy': num_atom_accuracy,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_volumes_mard': volumes_mard,
                f'{prefix}_type_accuracy': type_accuracy,
            })

        return log_dict, loss
    

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training step for PyTorch Lightning.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        teacher_forcing = (
            self.current_epoch <= self.hparams.teacher_forcing_max_epoch)
        outputs = self(batch, teacher_forcing, training=True)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Validation step for PyTorch Lightning.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Validation loss
        """
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Test step for PyTorch Lightning.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Test loss
        """
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
        )
        return loss
    

    @torch.no_grad()
    def langevin_dynamics(self, z, ld_kwargs, gt_num_atoms=None, gt_atom_types=None):
        """
        Decode crystal structure from latent embeddings using annealed Langevin dynamics.
        
        Args:
            z: Latent vector
            ld_kwargs: Arguments for Langevin dynamics:
                n_step_each: Number of steps for each sigma level
                step_lr: Step size parameter
                min_sigma: Minimum sigma to use in annealed Langevin dynamics
                save_traj: If True, save the entire LD trajectory
                disable_bar: Disable the progress bar of Langevin dynamics
            gt_num_atoms: If not None, use the ground truth number of atoms
            gt_atom_types: If not None, use the ground truth atom types
            
        Returns:
            dict: Generated crystal structure with optional trajectory
        """
        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        # Obtain key statistics from latent vector
        num_atoms, _, lengths, angles, composition_per_atom = self.decode_stats(
            z, gt_num_atoms)
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        # Obtain atom types
        composition_per_atom = F.softmax(composition_per_atom, dim=-1)
        if gt_atom_types is None:
            cur_atom_types = self.sample_composition(
                composition_per_atom, num_atoms)
        else:
            cur_atom_types = gt_atom_types

        # Initialize coordinates randomly
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device)

        # Annealed Langevin dynamics
        time = 0
        for sigma in tqdm(self.sigmas, total=self.sigmas.size(0), disable=ld_kwargs.disable_bar):
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            # Add time embedding
            time_tensor = torch.Tensor([time])
            time_tensor = time_tensor.to(self.device)
            time_emb = self.time_mlp(time_tensor)
            time_emb = time_emb.repeat(z.size(0), 1)
            z_time = torch.cat((z, time_emb), dim=1)
            time += 1

            for step in range(ld_kwargs.n_step_each):
                # Add noise
                noise_cart = torch.randn_like(
                    cur_frac_coords) * torch.sqrt(step_size * 2)
                pred_cart_coord_diff, pred_atom_types = self.decoder(
                    z_time, cur_frac_coords, cur_atom_types, num_atoms, lengths, angles)
                cur_cart_coords = frac_to_cart_coords(
                    cur_frac_coords, lengths, angles, num_atoms)
                pred_cart_coord_diff = pred_cart_coord_diff / sigma  # This is the S(x,t) in the paper
                cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart
                cur_frac_coords = cart_to_frac_coords(
                    cur_cart_coords, lengths, angles, num_atoms)

                if gt_atom_types is None:
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1

                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(
                        step_size * pred_cart_coord_diff)
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

        output_dict = {'num_atoms': num_atoms, 'lengths': lengths, 'angles': angles,
                       'frac_coords': cur_frac_coords, 'atom_types': cur_atom_types,
                       'is_traj': False}

        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_frac_coords=torch.stack(all_frac_coords, dim=0),
                all_atom_types=torch.stack(all_atom_types, dim=0),
                all_pred_cart_coord_diff=torch.stack(
                    all_pred_cart_coord_diff, dim=0),
                all_noise_cart=torch.stack(all_noise_cart, dim=0),
                is_traj=True))

        return output_dict

    def sample_composition(self, composition_prob, num_atoms):
        """
        Sample composition such that it exactly satisfies composition_prob.
        
        Args:
            composition_prob: Target composition probabilities
            num_atoms: Number of atoms per structure
            
        Returns:
            Sampled atom types that match the target composition
        """
        batch = torch.arange(
            len(num_atoms), device=num_atoms.device).repeat_interleave(num_atoms)
        assert composition_prob.size(0) == num_atoms.sum() == batch.size(0)
        composition_prob = scatter(
            composition_prob, index=batch, dim=0, reduce='mean')

        all_sampled_comp = []

        for comp_prob, num_atom in zip(list(composition_prob), list(num_atoms)):
            comp_num = torch.round(comp_prob * num_atom)
            atom_type = torch.nonzero(comp_num, as_tuple=True)[0] + 1
            atom_num = comp_num[atom_type - 1].long()

            sampled_comp = atom_type.repeat_interleave(atom_num, dim=0)

            # If the rounded composition gives less atoms, sample the rest
            if sampled_comp.size(0) < num_atom:
                left_atom_num = num_atom - sampled_comp.size(0)

                left_comp_prob = comp_prob - comp_num.float() / num_atom

                left_comp_prob[left_comp_prob < 0.] = 0.
                left_comp = torch.multinomial(
                    left_comp_prob, num_samples=left_atom_num, replacement=True)
                # Convert to atomic number
                left_comp = left_comp + 1
                sampled_comp = torch.cat([sampled_comp, left_comp], dim=0)

            sampled_comp = sampled_comp[torch.randperm(sampled_comp.size(0))]
            sampled_comp = sampled_comp[:num_atom]
            all_sampled_comp.append(sampled_comp)

        all_sampled_comp = torch.cat(all_sampled_comp, dim=0)
        assert all_sampled_comp.size(0) == num_atoms.sum()
        return all_sampled_comp

