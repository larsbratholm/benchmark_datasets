import sys
import pandas as pd
import numpy as np
import glob
import time
from qml.representations import generate_bob
from fns.fn import brutefn, fastfn

def read_file(filename):
    atoms = ""
    coords = []
    mulliken = []
    with open(filename) as f:
        lines = f.readlines()
        no_atoms = int(lines[0])
        energy = float(lines[1].split()[-4])
        for i in range(no_atoms):
            line = lines[2+i]
            tokens = line.split()
            atoms += tokens[0]
            coords.append(tokens[1:4])
            mulliken.append(tokens[4])

    return atoms, coords, energy, mulliken

def atoms_str2int(string):
    string = string.replace('C','6,')
    string = string.replace('H','1,')
    string = string.replace('O','8,')
    string = string.replace('N','7,')
    string = string.replace('F','9,')
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

    order = fastfn(descriptors, npartitions = 1, memory = 50)

    return order

def pyfn(X):

    t = 0

    order = np.arange(0,X.shape[0], 1, dtype=int)
    order[0], order[t] = order[t], order[0]

    for n in range(1, X.shape[0]-1):
        maxd = 0
        idx = None
        for i,I in enumerate(order[n:]):
            x = X[I]
            d = np.inf
            for j,J in enumerate(order[:n]):
                y = X[J]
                d = min(d,sum((y-x)**2))
            if d > maxd:
                maxd = d
                idx = n+i
        order[n], order[idx] = order[idx], order[n]
    return order



filenames = sorted(glob.glob('qm9/*/*.xyz'))

energies = []
all_atoms = []
all_coordinates = []
all_n = []
partial_charges = []
for f, filename in enumerate(filenames):
 try:
    atoms, coordinates, energy, mulliken = read_file(filename)
    n_atoms = len(atoms)
    energies.append(energy)
    all_atoms.append(atoms)
    all_coordinates.append(np.asarray(coordinates, dtype=float))
    all_n.append(n_atoms)
    partial_charges.extend(mulliken)
 except:
     print(filename)

all_coordinates = np.asarray(all_coordinates, dtype=object)
partial_charges = np.asarray(partial_charges)
all_n = np.asarray(all_n, dtype=int)
energy = np.asarray(energies)

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

adf = pd.DataFrame({'atomic_number': atomic_number, 'X': all_coordinates[:,0], 'Y': all_coordinates[:,1], 'Z': all_coordinates[:,2], 'partial_charges': partial_charges})
mdf = pd.DataFrame({'N': all_n[order], 'atom_index': atom_indices, 'b3lyp_energy': energy[order]})

#print(adf.loc[mdf.loc[0,'atom_index']:mdf.loc[0,'atom_index']+mdf.loc[0,'N']-1,['X','Y','Z']].values)
print(adf.head(1))
print(mdf.head(1))
mdf.to_hdf('qm9.h5', 'molecules', mode='w', format='f')#, compression='blosc', complevel=9)
adf.to_hdf('qm9.h5', 'atoms', mode='a', format='f')#, compression='blosc', complevel=9)
