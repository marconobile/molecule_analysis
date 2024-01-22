import sys
sys.path.append("..")
from utils.data_utils import create_log, append_line_to_log
import os
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
import multiprocessing
from pathlib import Path
from multiprocessing import Pool
from itertools import islice
import warnings
from rdkit import RDLogger
import numpy as np
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")
log = create_log()


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def separate_tuples(tuples_list):
    first_elements = [item[0] for item in tuples_list]
    second_elements = [item[1] for item in tuples_list]
    return first_elements, second_elements


def flatten_list_of_lists(l):
    return [el for list_ in l for el in list_]


def to_pd(out, name):
    '''
    out is a list of MolStructs
    '''
    smiles = []
    n_atoms = []
    fps = []
    for el in out:
        smiles.append(el.smile)
        n_atoms.append(el.n_atoms)
        fps.append(el.fp)
    return pd.DataFrame({
        f"SMILES_{name}": smiles,
        f"NUM_ATOMS_{name}": n_atoms,
        "FPs": fps
    })


class MolStruct:
    def __init__(self, smile):
        self.smile = smile
        self.mol = Chem.MolFromSmiles(smile)
        self.fp = Chem.RDKFingerprint(self.mol)
        self.n_atoms = self.mol.GetNumAtoms()


def batch_ctors(l): return [MolStruct(smi) for smi in l]


p = Path("/home/nobilm@usi.ch/wd/data/synthesized_datasets/asinex/no_dup_asinex_03_Feb_2022_valid_properties.csv")
df = pd.read_csv(p)
asinex_df = df.drop(df.columns[0], axis=1)
p1 = Path("/home/nobilm@usi.ch/wd/data/training_data/moses_properties_properties.csv")
df1 = pd.read_csv(p1)
moses_df = df1.drop(df1.columns[0], axis=1)


moses_splits = np.array_split(moses_df, 10)
num_cores = 10  # multiprocessing.cpu_count() // 10
nels_per_split = 5000
treshold = 2
asinex_smiles = asinex_df["SMILES"].tolist()

for moses_part in range(len(moses_splits)):

    moses_df = moses_splits[moses_part]
    moses_smiles = moses_df["SMILES"].tolist()

    with Pool(processes=num_cores) as P:
        out = P.map(MolStruct, asinex_smiles)

    asinex_batches = batched(asinex_smiles, nels_per_split)
    with Pool(processes=num_cores) as P:
        out = P.map(batch_ctors, asinex_batches)
    asinex_out = flatten_list_of_lists(out)

    moses_batches = batched(moses_smiles, nels_per_split)
    with Pool(processes=num_cores) as P:
        out = P.map(batch_ctors, moses_batches)
    moses_out = flatten_list_of_lists(out)
    dn_name = "moses"
    df = to_pd(moses_out, dn_name)

    def compute_dist(molstruct):
        filter_ = df[f"NUM_ATOMS_{dn_name}"].between(
            molstruct.n_atoms-treshold, molstruct.n_atoms+treshold)
        possible_matches = df[filter_]
        possible_matches["tanimoto"] = DataStructs.BulkTanimotoSimilarity(
            molstruct.fp, possible_matches["FPs"].tolist())
        return possible_matches

    def compute_matches(molstruct):
        filter_ = df[f"NUM_ATOMS_{dn_name}"] == molstruct.n_atoms
        possible_matches = df[filter_]
        possible_matches["tanimoto"] = DataStructs.BulkTanimotoSimilarity(
            molstruct.fp, possible_matches["FPs"].tolist())

        filter_ = possible_matches[f"tanimoto"] == 1
        exact_matches = possible_matches[filter_]
        return exact_matches

    for i in range(len(asinex_out)):
        # out_df = compute_dist(asinex_out[i])
        out_df = compute_matches(asinex_out[i])

        if out_df.empty:
            continue

        append_line_to_log(
            log, f"Asinex smi: { asinex_out[i].smile}, moses smi: {out_df['SMILES_moses'].tolist()[0]}, tanimoto: {out_df['tanimoto'].tolist()[0]}, natoms: {out_df['NUM_ATOMS_moses'].tolist()[0]}")
