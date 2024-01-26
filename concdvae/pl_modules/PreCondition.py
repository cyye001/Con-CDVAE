import torch
import torch.nn as nn
from concdvae.pl_modules.model import build_mlp
from torch.nn import functional as F

class ScalarConditionPredict(nn.Module):
    def __init__(
            self,
            condition_name: str,
            condition_min: float,
            condition_max: float,
            latent_dim: int,
            hidden_dim: int,
            out_dim: int,
            n_layers: int,
            drop: float = -1,
    ):
        super(ScalarConditionPredict, self).__init__()
        self.condition_name = condition_name
        self.condition_min = condition_min
        self.condition_max = condition_max
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.drop = drop


        self.mlp = build_mlp(in_dim=self.latent_dim,
                             hidden_dim=self.hidden_dim,
                             fc_num_layers=self.n_layers,
                             out_dim=self.out_dim,
                             drop=self.drop)


    def forward(self, inputs, z):
        predict = self.mlp(z)
        loss = self.property_loss(inputs, predict)
        return loss


    def property_loss(self, inputs, predict):
        true = torch.Tensor(inputs[self.condition_name]).float()
        true = (true - self.condition_min) / (self.condition_max - self.condition_min)
        true = true.view(true.size(0), 1)
        return F.mse_loss(predict, true)


class ClassConditionPredict(nn.Module):
    def __init__(
            self,
            condition_name: str,
            n_type: int,
            latent_dim: int,
            hidden_dim: int,
            n_layers: int,
            drop: float = -1,
    ):
        super(ClassConditionPredict, self).__init__()
        self.condition_name = condition_name
        self.n_type = n_type
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop = drop

        if drop > 0 and drop < 1:
            list_sqe = [nn.Dropout(p=drop), nn.Linear(self.latent_dim, self.n_type)]
        else:
            list_sqe = [nn.Linear(self.latent_dim, self.n_type)]
        self.mlp = nn.Sequential(*list_sqe)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, z):
        predict = self.mlp(z)
        loss = self.property_loss(inputs, predict)
        return loss


    def property_loss(self, inputs, predict):
        true = torch.Tensor(inputs[self.condition_name]).long()
        return self.criterion(predict,true)
