import hydra
import omegaconf
import torch
import pandas as pd
import numpy as np
import os
import json
from omegaconf import ValueNode
from torch.utils.data import Dataset

from torch_geometric.data import Data
from pymatgen.core.structure import Structure

from concdvae.common.utils import PROJECT_ROOT
from concdvae.common.data_utils import (
    preprocess, add_scaled_lattice_prop,chemical_symbols)

class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, use_prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.use_prop = use_prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method



        self.cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=list(prop))

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None

        atom_init_file = os.path.dirname(self.path)
        atom_init_file = os.path.join(atom_init_file, 'atom_init.json')
        if os.path.exists(atom_init_file):
            self.ari = AtomCustomJSONInitializer(atom_init_file)
            for i in range(len(self.cached_data)):
                crystal = Structure.from_str(self.cached_data[i]['cif'], fmt="cif")

                atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                                      for i in range(len(crystal))])
                atom_fea = torch.Tensor(atom_fea)
                atom_fea = torch.mean(atom_fea, dim=0)
                atom_fea = atom_fea.reshape(1, 92)
                self.cached_data[i].update({'formula':atom_fea})
        else:
            self.ari = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )

        exclude_keys = ['cif', 'graph_arrays', 'scaled_lattice']
        filtered_data = {key: value for key, value in data_dict.items() if key not in exclude_keys}
        data.update(filtered_data)

        if self.ari != None:
            data.update({'formula': self.cached_data[index]['formula']})

        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


def formula2atomnums(formula):
    elements = []
    current_element = ""
    current_count = ""

    for char in formula:
        if char.isupper():
            if current_element:
                elements.append((current_element, int(current_count) if current_count else 1))
            current_element = char
            current_count = ""
        elif char.islower():
            current_element += char
        elif char.isdigit():
            current_count += char

    if current_element:
        elements.append((current_element, int(current_count) if current_count else 1))

    ele_list = []
    for data in elements:
        for time in range(data[1]):
            ele_list.append(data[0])


    index_list = []
    for ele in ele_list:
        index = chemical_symbols.index(ele)
        index_list.append(index)


    return index_list