import sys
import pandas as pd
import numpy as np
import glob
import time
from qml.representations import generate_bob
from fns.fn import fastfn
from collections import defaultdict

def read_file(filename):
    atoms = ""
    coords = []
    with open(filename) as f:
        lines = f.readlines()
        energy = float(lines[1].split()[1]) * 627.509
        for line in lines[2:]:
            tokens = line.split()
            atoms += tokens[0]
            coords.append(tokens[1:])

    id_ = int(filename.split("/")[-1].split(".")[0])

    return atoms, coords, energy, id_

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

    t = time.time()
    order = fastfn(descriptors, npartitions = 1, memory = 2000)
    print(time.time() - t)

    return order

def parse_ccsd(filenames):
    cc_energies, hf_energies = [], []
    cc_time, hf_time = [], []
    ids = []
    for filename in filenames:
        with open(filename) as f:
            lines = f.readlines()
            for i, line in enumerate(lines[::-1]):
                if "UCCSD" in line:
                    cc_energy, hf_energy = lines[-i].split()
                    cc_energies.append(float(cc_energy))
                    hf_energies.append(float(hf_energy))
                elif "CPU TIMES" in line:
                    cc_time.append(float(lines[-i-1].split()[3]))
                    hf_time.append(float(lines[-i-1].split()[6]))
                    break

        id_ = int(filename.split("/")[-1].split("_")[1])
        ids.append(id_)

    return cc_energies, hf_energies, cc_time, hf_time, ids

def parse_dft(filenames):
    energies = []
    time = []
    ids = []
    for filename in filenames:
        with open(filename) as f:
            lines = f.readlines()
            flag = False
            for i, line in enumerate(lines[::-1]):
                if "KS-SCF" in line:
                    energy = float(lines[-i])
                    energies.append(energy)
                    flag = True
                elif "CPU TIMES" in line:
                    if flag:
                        time.append(float(lines[-i-1].split()[3]))
                        break
            else:
                print(filename)

        id_ = int(filename.split("/")[-1].split("_")[1].split("-")[0])
        ids.append(id_)

    return energies, time, ids

filenames = sorted(glob.glob('trainingdata/dftb/*.xyz'))

# parse energies and whatnot
cc_energies, hf_energies, cc_time, hf_time, ccids = parse_ccsd(glob.glob('trainingdata/ccsd/*.out'))
b3lyp_energies, b3lyp_time, b3lypids = parse_dft(glob.glob('trainingdata/b3lyp/*z.out'))
ub3lyp_energies, ub3lyp_time, ub3lypids = parse_dft(glob.glob('trainingdata/b3lyp/*u.out'))
pbe_energies, pbe_time, pbeids = parse_dft(glob.glob('trainingdata/pbe/*g.out'))
upbe_energies, upbe_time, upbeids = parse_dft(glob.glob('trainingdata/pbe/*u.out'))

d = defaultdict(list)
# create dictionary with all the things
for i,idx in enumerate(ccids):
    d[idx].append(cc_energies[i])
    d[idx].append(hf_energies[i])
    d[idx].append(cc_time[i])
    d[idx].append(hf_time[i])
for i,idx in enumerate(b3lypids):
    d[idx].append(b3lyp_energies[i])
    d[idx].append(b3lyp_time[i])
for i,idx in enumerate(ub3lypids):
    d[idx].append(ub3lyp_energies[i])
    d[idx].append(ub3lyp_time[i])
for i,idx in enumerate(pbeids):
    d[idx].append(pbe_energies[i])
    d[idx].append(pbe_time[i])
for i,idx in enumerate(upbeids):
    d[idx].append(upbe_energies[i])
    d[idx].append(upbe_time[i])



molecule_ids = []
energy = []
all_atoms = []
all_coordinates = []
all_n = []
for f, filename in enumerate(filenames):
    atoms, coordinates, dftb_energy, id_ = read_file(filename)
    n_atoms = len(atoms)
    energy.append(dftb_energy)
    all_atoms.append(atoms)
    all_coordinates.append(np.asarray(coordinates, dtype=float))
    all_n.append(n_atoms)
    molecule_ids.append(id_)

all_coordinates = np.asarray(all_coordinates, dtype=float)
all_n = np.asarray(all_n, dtype=int)
energy = np.asarray(energy)

#x = sorted(molecule_ids)
#y = sorted(list(d.keys()))
#
#i = 0
#j = 0
#while i < len(x):
#    xi = x[i]
#    yi = y[i+j]
#    if xi == yi:
#        i += 1
#    elif xi < yi:
#        i += 1
#        j -= 1
#        print(xi)
#    else:
#        j += 1
#        print(yi)



# Initial ordering of data
data = np.zeros((all_n.size,12))
for i,idx in enumerate(molecule_ids):
    x = np.asarray(d[idx])
    data[i] = np.asarray(d[idx])

mean_time = np.mean(data, axis=0)/60

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


adf = pd.DataFrame({'atomic_number': atomic_number, 'X': all_coordinates[:,0], 'Y': all_coordinates[:,1], 'Z': all_coordinates[:,2]})
mdf = pd.DataFrame({'N': all_n[order], 'atom_index': atom_indices, 'uccsd': data[:,0], 'hf': data[:,1], 'b3lyp': data[:,4], 'ub3lyp': data[:,6], 'pbe': data[:,8], 'upbe': data[:,10]})

#print(adf.loc[mdf.loc[0,'atom_index']:mdf.loc[0,'atom_index']+mdf.loc[0,'N']-1,['X','Y','Z']].values)
print(adf.head(1))
print(mdf.head(1))
mdf.to_hdf('ch4cn.h5', 'molecules', mode='w', format='f')#, compression='blosc', complevel=9)
adf.to_hdf('ch4cn.h5', 'atoms', mode='a', format='f')#, compression='blosc', complevel=9)
