##This part of the code refers to https://github.com/atomistic-machine-learning/SchNet

import torch.nn as nn
from concdvae.pl_modules.model import build_mlp
import hydra
import torch
import math
import sys
from typing import Dict, Optional, List, Callable, Union, Sequence
from omegaconf import ListConfig

class ConditioningModule(nn.Module):
    def __init__(
        self,
        n_features,
        n_layers,
        condition_embeddings,
        ):

        super(ConditioningModule, self).__init__()
        self.n_features = n_features
        self.condition_embeddings = condition_embeddings
        condition_embModel = []
        # self.condition_embModel = condition_embeddings
        n_in = 0
        for condition_emb in self.condition_embeddings:
            condition_embModel.append(hydra.utils.instantiate(condition_emb))
            n_in += condition_emb.n_features
        self.condition_embModel = nn.ModuleList(condition_embModel)

        self.dense_net = build_mlp(
            in_dim=n_in,
            out_dim=self.n_features,
            hidden_dim=self.n_features,
            fc_num_layers=n_layers,
            norm=False,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # embed all conditions
        emb_features = []
        for emb in self.condition_embModel:
            emb_features += [emb(inputs)]
        # concatenate the features
        emb_features = torch.cat(emb_features, dim=-1)
        # mix the concatenated features
        conditional_features = self.dense_net(emb_features)
        return conditional_features



class ConditionEmbedding(nn.Module):

    def __init__(
        self,
        condition_name: str,
        n_features: int,
        required_data_properties: Optional[List[str]] = [],
        condition_type: str = "trajectory",
    ):

        super().__init__()
        if condition_type not in ["trajectory", "step", "atom"]:
            raise ValueError(
                f"`condition_type` is {condition_type} but needs to be `trajectory`, "
                f"`step`, or `atom` for trajectory-wise, step-wise, or atom-wise "
                f"conditions, respectively."
            )
        self.condition_name = condition_name
        self.condition_type = condition_type
        self.n_features = n_features
        self.required_data_properties = required_data_properties

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError


# class ScalarConditionEmbedding(ConditionEmbedding):
class ScalarConditionEmbedding(nn.Module):
    def __init__(
        self,
        condition_name: str,
        condition_min: float,
        condition_max: float,
        grid_spacing: float,
        n_features: int,
        n_layers: int,
        required_data_properties: Optional[List[str]] = [],
        condition_type: str = "trajectory",
    ):
        super(ScalarConditionEmbedding, self).__init__()
        # super().__init__(
        #     condition_name, n_features, required_data_properties, condition_type
        # )
        self.condition_name = condition_name
        self.condition_type = condition_type
        self.n_features = n_features
        self.required_data_properties = required_data_properties
        # compute the number of rbfs
        n_rbf = math.ceil((condition_max - condition_min) / grid_spacing) + 1
        # compute the position of the last rbf
        _max = condition_min + grid_spacing * (n_rbf - 1)
        # initialize Gaussian rbf expansion network
        self.gaussian_expansion = GaussianRBF(
            n_rbf=n_rbf, cutoff=_max, start=condition_min
        )
        # initialize fully connected network
        self.dense_net = build_mlp(
            in_dim=n_rbf,
            hidden_dim=n_features,
            fc_num_layers = n_layers,
            out_dim=n_features,
            norm=False,
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # print('here is input:',input.keys(),flush=True)
        # get the scalar condition value
        scalar_condition = torch.Tensor(inputs[self.condition_name]).float()
        # expand the scalar value with Gaussian rbfs
        expanded_condition = self.gaussian_expansion(scalar_condition)
        # feed through fully connected network
        embedded_condition = self.dense_net(expanded_condition)
        return embedded_condition


class ClassConditionEmbedding(nn.Module):
    def __init__(
        self,
        condition_name: str,
        n_type: int,
        n_emb: int,
        n_features: int,
        n_layers: int,
        required_data_properties: Optional[List[str]] = [],
        condition_type: str = "trajectory",
    ):
        super(ClassConditionEmbedding, self).__init__()
        self.condition_name = condition_name
        self.n_type = n_type
        self.embedding_layer = nn.Embedding(n_type, n_emb)

        self.dense_net = build_mlp(
            in_dim=n_emb,
            hidden_dim=n_features,
            fc_num_layers=n_layers,
            out_dim=n_features,
            norm=False,
        )

    def forward(
            self,
            inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        emb_input = inputs[self.condition_name].int()
        emb_condition = self.embedding_layer(emb_input)
        embedded_condition = self.dense_net(emb_condition)
        return embedded_condition


class VectorialConditionEmbedding(nn.Module):
    """
    An embedding network for vectorial conditions (e.g. a fingerprint). The vector is
    mapped to the final embedding with a fully connected neural network.
    """

    def __init__(
        self,
        condition_name: str,
        n_in: int,
        n_features: int,
        n_layers: int,
        required_data_properties: Optional[List[str]] = [],
        condition_type: str = "trajectory",
    ):

        super(VectorialConditionEmbedding, self).__init__()
        self.condition_name = condition_name
        # initialize fully connected network
        self.dense_net = build_mlp(
            in_dim=n_in,
            hidden_dim=n_features,
            fc_num_layers=n_layers,
            out_dim=n_features,
            norm=False,
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # get the vectorial condition value
        vectorial_condition = inputs[self.condition_name]
        # feed through fully connected network
        embedded_condition = self.dense_net(vectorial_condition)
        return embedded_condition


class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            # self.register_buffer("widths", widths)
            # self.register_buffer("offsets", offset)
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
            self.widths.requires_grad = False
            self.offsets.requires_grad = False

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)


def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    # print('input de:', inputs.device, file=sys.stdout)
    # print('offset de:', offsets.device, file=sys.stdout)
    # print('widths de:', widths.device, file=sys.stdout)
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y.float()