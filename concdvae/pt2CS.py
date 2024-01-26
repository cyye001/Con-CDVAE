import torch
from pymatgen.core.lattice import Lattice
from collections import Counter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
import os

def determine_crystal_system(structure, tolerance=1.0, angle_tolerance=5):
    try:
        sga = SpacegroupAnalyzer(structure, symprec=tolerance,angle_tolerance=angle_tolerance)

        crystal_system = sga.get_crystal_system()

        return crystal_system

    except Exception as e:
        return "error" + str(e)

dataroot = 'D:/2-project/0-MaterialDesign/3-CDVAE/cdvae-pre4/output/hydra/singlerun/2023-12-25/mp20_CS/'
dataname = 'eval_gen_less_CS'
lastname = ['0.pt','1.pt','2.pt','3.pt','4.pt','5.pt','6.pt',]
for last in lastname:
    print('last name: ', last)
    datafile = dataname + last
    tolerance = 0.2
    angle_tolerance = 5
    datafile_read = os.path.join(dataroot, datafile)
    data = torch.load(datafile_read,map_location=torch.device('cpu'))
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
    CS_type = []
    cs_name = []
    for i in range(len(num_atoms_list)):
        now_atom = 0
        for a in range(len(num_atoms_list[i])):
            length = lengths_list[i][a]
            angle = angles_list[i][a]
            atom_num = num_atoms_list[i][a]

            atom_type = atom_types_list[i][now_atom: now_atom + atom_num]
            frac_coord = frac_coors_list[i][now_atom: now_atom + atom_num][:]
            lattice = Lattice.from_parameters(a=length[0], b=length[1], c=length[2], alpha=angle[0],
                                              beta=angle[1], gamma=angle[2])

            structure = Structure(lattice, atom_type, frac_coord, to_unit_cell=True)

            crystal_system = determine_crystal_system(structure, tolerance, angle_tolerance)
            cs_name.append(crystal_system)

    print('result')
    element_counts = Counter(cs_name)

    for element, count in element_counts.items():
        print(f"{element}: {count} time")

    print('-------------------------------------------------------------')

