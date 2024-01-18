'''
Given a smile file it computes its intersection against moses in a slightly inefficient manner, 
not used to analyze generative modes
just used to compare train data and ref data
out file in same dir of input file 

Example
path = "/home/nobilm@usi.ch/wd/data/synthesized_datasets/maybridge/no_dup_Maybridge_HitDiscover_valid.smiles"
cmd: python get_intersection_against_moses_train.py /home/nobilm@usi.ch/wd/data/synthesized_datasets/maybridge/no_dup_Maybridge_HitDiscover_valid.smiles
'''

import os
from pathlib import Path
from utils.data_utils import get_dir, get_filename, mols_from_file
import argparse
from rdkit import Chem


# ----- Parser ----- #
parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()
path = args.path

moses_path = "./data/training_data/moses.txt"
moses_mols = mols_from_file(moses_path)
ref_mols = mols_from_file(path)

ref_mols = ref_mols[:5000]

exact_matches_mols = []
for m1 in moses_mols:
    nm1 = m1.GetNumAtoms()
    nbm1 = len(list(m1.GetBonds()))
    for m2 in ref_mols:
        if nm1 == m2.GetNumAtoms() and nbm1 == len(list(m2.GetBonds())):
            if m1.HasSubstructMatch(m2):
                exact_matches_mols.append((m1, m2))


dir = get_dir(path)
name = get_filename(get_dir(path))
filepath = os.path.join(dir, f"{str(name)}_vs_moses_train_duplicates.txt")
with open(filepath, "w+") as f:
    for i, (m1, m2) in enumerate(exact_matches_mols):
        f.write(
            f"{i} Moses: {Chem.MolToSmiles(m1)}     {str(name)}: {Chem.MolToSmiles(m2)}\n")
