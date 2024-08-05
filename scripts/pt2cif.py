import torch
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.composition import Composition
import os

dataroot = 'YOUR_PATH_TO_.PT'
datafile = 'eval_gen_abc.pt'

datafile_read = os.path.join(dataroot, datafile)
data = torch.load(datafile_read,map_location=torch.device('cpu'))
cif_path = os.path.join(dataroot, 'ciffile/')
if not os.path.exists(cif_path):
    os.makedirs(cif_path)
lengths = data['lengths']
angles = data['angles']
num_atoms = data['num_atoms']
frac_coors = data['frac_coords']
atom_types = data['atom_types']

lengths_list = lengths.numpy().tolist()
angles_list = angles.numpy().tolist()
num_atoms_list = num_atoms.tolist()
frac_coors_list = frac_coors.numpy().tolist()
atom_types_list = atom_types.tolist()

num_materal = 0
for i in range(len(num_atoms_list)): #第i个batch？
    now_atom = 0
    for a in range(len(num_atoms_list[i])): #第a个材料
        cif_mat_path = os.path.join(cif_path, str(num_materal))
        length = lengths_list[i][a]
        angle = angles_list[i][a]
        atom_num = num_atoms_list[i][a]

        atom_type = atom_types_list[i][now_atom: now_atom + atom_num]
        frac_coord = frac_coors_list[i][now_atom: now_atom + atom_num][:]
        lattice = Lattice.from_parameters(a=length[0], b=length[1], c=length[2], alpha=angle[0],
                                          beta=angle[1], gamma=angle[2])

        structure = Structure(lattice, atom_type, frac_coord, to_unit_cell=True)
        filename = datafile[:-3]+'__' + str(num_materal) + '.cif'
        file_path = os.path.join(cif_path, filename)
        structure.to(filename=file_path)
        now_atom += atom_num
        num_materal += 1

print('end')

