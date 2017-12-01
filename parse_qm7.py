import sys
import pandas as pd
import numpy as np
import glob
import time
from qml.representations import generate_bob
from fns.fn import fastfn

def read_file(filename):
    atoms = ""
    coords = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines[2:]:
            tokens = line.split()
            atoms += tokens[0]
            coords.append(tokens[1:])

    return atoms, coords

def atoms_str2int(string):
    string = string.replace('C','6,')
    string = string.replace('H','1,')
    string = string.replace('O','8,')
    string = string.replace('N','7,')
    string = string.replace('S','16,')

    return np.asarray(string.split(',')[:-1], dtype=int)

def sort_by_distance(atoms, coordinates):
    n = len(atoms)
    # Get unique atom types
    unique_atoms = set("".join(atoms))
    unique_nuclear_charges = atoms_str2int("".join(unique_atoms))
    max_counts = {atom:0 for atom in unique_atoms}

    for mol in atoms:
        counts = dict()
        for a in mol:
            counts[a] = counts.get(a, 0) + 1
        max_counts = {key:max(max_counts[key],counts.get(key,0)) for key in unique_atoms}

    # generate descriptor
    for i in range(n):
        nuclear_charges = atoms_str2int(atoms[i])
        desc = generate_bob(nuclear_charges, coordinates[i], unique_nuclear_charges, asize=max_counts)
        if i == 0:
            descriptors = np.empty((n,desc.shape[0]))
        descriptors[i] = desc

    t = time.time()
    order = fastfn(descriptors, npartitions = 1, memory = 50)
    print(time.time() - t)

    return order


filenames = sorted(glob.glob('qm7/*.xyz'))
energies = np.loadtxt('qm7/hof_qm7.txt', usecols=[1,2])

molecule_ids = []
energy1 = []
energy2 = []
molecule_id = 0
all_atoms = []
all_coordinates = []
all_n = []
for f, filename in enumerate(filenames):
    atoms, coordinates = read_file(filename)
    n_atoms = len(atoms)
    energy1.append(energies[f,0])
    energy2.append(energies[f,1])
    all_atoms.append(atoms)
    all_coordinates.append(np.asarray(coordinates, dtype=float))
    all_n.append(n_atoms)

all_coordinates = np.asarray(all_coordinates, dtype=object)
all_n = np.asarray(all_n, dtype=int)
energy1 = np.asarray(energy1)
energy2 = np.asarray(energy2)


# sort by distance
order = sort_by_distance(all_atoms, all_coordinates)

# Get indices of first atom for lookup
atom_indices = np.empty(len(filenames), dtype=int)
n = 0
for i, j in enumerate(order):
    atom_indices[i] = n
    n += all_n[j]

# atom number
atomic_number = np.concatenate([atoms_str2int(x) for x in all_atoms])

all_coordinates = np.concatenate(all_coordinates[order].ravel())

adf = pd.DataFrame({'atomic_number': atomic_number, 'X': all_coordinates[:,0], 'Y': all_coordinates[:,1], 'Z': all_coordinates[:,2]})
mdf = pd.DataFrame({'N': all_n[order], 'atom_index': atom_indices, 'pbe_atomization_energy': energy1[order], 'dftb_atomization_energy': energy2[order]})

#print(adf.loc[mdf.loc[0,'atom_index']:mdf.loc[0,'atom_index']+mdf.loc[0,'N']-1,['X','Y','Z']].values)
print(adf.head(1))
print(mdf.head(1))
mdf.to_hdf('qm7.h5', 'molecules', mode='w', format='f')#, compression='blosc', complevel=9)
adf.to_hdf('qm7.h5', 'atoms', mode='a', format='f')#, compression='blosc', complevel=9)
