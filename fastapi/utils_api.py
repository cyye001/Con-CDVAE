import torch
import numpy as np
from concdvae.common.data_utils import chemical_symbols
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
# from concdvae.pl_data.dataset import formula2atomnums
# from concdvae.pl_data.dataset import AtomCustomJSONInitializer

def full_pre(n_cry, fe, n_atom, formula, ari, device):
    prop_dict = {}
    warnings = []

    if n_cry == 'None':
        n_cry = 1
    n_cry = int(n_cry)

    # if bg != 'None':
    #     bg = float(bg)
    #     prop_dict.update({'bandgap': torch.Tensor([bg]).float().to(device),'band_gap': torch.Tensor([bg]).float().to(device)})

    if fe != 'None':
        fe = float(fe)
        prop_dict.update({'formation': torch.Tensor([fe]).float().to(device), 'formation_energy_per_atom': torch.Tensor([fe]).float().to(device)})

    if formula != 'None':
        # TODO if formula is not right
        atom_list, n_atom2 = formula2atomnums(formula)
        if n_atom2 == 'Err':
            prop_dict.update({'err': 'Please check whether the formula is right. The input \''+formula+'\' can not be read.'})
            return n_cry, prop_dict
        atom_fea = np.vstack([ari.get_atom_fea(atom_list[k])
                              for k in range(len(atom_list))])
        atom_fea = torch.Tensor(atom_fea)
        atom_fea = torch.mean(atom_fea, dim=0)
        atom_fea = atom_fea.reshape(1, 92)
        atom_fea = atom_fea.to(device)

        if n_atom == 'None' :
            n_atom = int(n_atom2)
        elif abs(float(n_atom) % n_atom2) > 0.0001:
            warnings.append('\'n_atom\'='+str(n_atom)+' is not matching the formula, and reset to ' + str(int(n_atom2)))
            n_atom = int(n_atom2)
        else:
            atom_list = atom_list * int(float(n_atom)/n_atom2)

        prop_dict.update({'formula': atom_fea.float().to(device)})
        prop_dict.update({'gt_atom_types': torch.Tensor(atom_list * n_cry).int().to(device)})

    if n_atom != 'None':
        n_atom = int(n_atom)
        prop_dict.update({'n_atom': torch.Tensor([n_atom]).int().to(device)})
        prop_dict.update({'gt_num_atoms': torch.Tensor([n_atom] * n_cry).int().to(device)})

    if len(warnings)>0:
        prop_dict.update({'warning': warnings})
    return n_cry, prop_dict


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
    n_atom = 0
    for data in elements:
        n_atom += data[1]
        for time in range(data[1]):
            ele_list.append(data[0])

    index_list = []
    try:
        for ele in ele_list:
            index = chemical_symbols.index(ele)
            index_list.append(index)
    except:
        return index_list, 'Err'

    return index_list, n_atom


def tensor2cif(frac_coords, num_atoms, atom_types, lengths, angles):
    lengths_list = lengths.numpy().tolist()
    angles_list = angles.numpy().tolist()
    num_atoms_list = num_atoms.tolist()
    frac_coors_list = frac_coords.numpy().tolist()
    atom_types_list = atom_types.tolist()

    num_materal = 0
    cif_list = []
    for i in range(len(num_atoms_list)):  # 第i个batch
        now_atom = 0
        for a in range(len(num_atoms_list[i])):  # 第a个材料
            length = lengths_list[i][a]
            angle = angles_list[i][a]
            atom_num = num_atoms_list[i][a]

            atom_type = atom_types_list[i][now_atom: now_atom + atom_num]
            frac_coord = frac_coors_list[i][now_atom: now_atom + atom_num][:]
            lattice = Lattice.from_parameters(a=length[0], b=length[1], c=length[2], alpha=angle[0],
                                              beta=angle[1], gamma=angle[2])

            structure = Structure(lattice, atom_type, frac_coord, to_unit_cell=True)

            cif_text = structure.to(fmt="cif")
            cif_list.append(cif_text)
            now_atom += atom_num
            num_materal += 1

    return cif_list