import sys
import os
import pandas as pd
import numpy as np
import glob
import ast
import time
from qml.representations import generate_bob
from fns.fn import fastfn

def read_file(filename):
    atoms = ""
    coords = []
    with open(filename) as f:
        lines = f.readlines()
        tokens = lines[1].split(';')
        energy = float(tokens[0])
        forces = ast.literal_eval(tokens[1])
        for line in lines[2:]:
            tokens = line.split()
            atoms += tokens[0]
            coords.append(tokens[1:])

    return atoms, coords, energy, forces

def atoms_str2int(string):
    string = string.replace('H','1,')
    string = string.replace('C','6,')
    string = string.replace('N','7,')
    string = string.replace('O','8,')
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

    return order

with pd.HDFStore('md.h5') as store:
    for folder in glob.glob('md_datasets/*/'):
        molecule_name = folder.split('/')[-2]
        print(molecule_name)
        filenames = sorted(glob.glob(folder + '/*.xyz'))[:20000]
        energies = []
        all_atoms = []
        all_coordinates = []
        energy = []
        all_forces = []
        all_n = []
        for f, filename in enumerate(filenames):
            atoms, coordinates, energy, forces = read_file(filename)
            n_atoms = len(atoms)
            energies.append(energy)
            all_forces.append(forces)
            all_atoms.append(atoms)
            all_coordinates.append(np.asarray(coordinates, dtype=float))
            all_n.append(n_atoms)

        all_coordinates = np.asarray(all_coordinates)
        all_n = np.asarray(all_n, dtype=int)
        energies = np.asarray(energies)
        all_forces = np.asarray(all_forces)

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

        all_coordinates = all_coordinates[order].reshape(-1, 3)
        all_forces = all_forces[order].reshape(-1, 3)

        adf = pd.DataFrame({'atomic_number': atomic_number, 'X': all_coordinates[:,0], 'Y': all_coordinates[:,1], 'Z': all_coordinates[:,2], 'FX': all_forces[:,0], 'FY': all_forces[:,1], 'FZ': all_forces[:,2]})
        mdf = pd.DataFrame({'N': all_n[order], 'atom_index': atom_indices, 'energies': energies})
        store.put('%s/molecules' % molecule_name, mdf, format = 't')
        store.put('%s/atoms' % molecule_name, adf, format = 't')
