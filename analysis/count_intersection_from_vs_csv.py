import os
import pandas as pd
import numpy as np
from rdkit import Chem
from pathlib import Path
from rdkit.Chem import Descriptors
from rdkit import Chem, DataStructs
from utils.data_utils import get_filename

import argparse

# ----- Parser ----- #
parser = argparse.ArgumentParser()
parser.add_argument("path_to_vs_file")
args = parser.parse_args()
path_join_file = Path(args.path_to_vs_file)

method_name = get_filename(path_join_file.parent)
df = pd.read_csv(path_join_file)

path_intersection = "/home/nobilm@usi.ch/wd/data/synthesized_datasets/maybridge/maybridge_vs_moses_train_duplicates.txt"
maybridge_smiles_in_moses_train = set()
with open(path_intersection) as file:
    for i, line in enumerate(file):
        maybridge_smiles_in_moses_train.add(
            line[line.find(":")+1: line.find("Maybridge")].strip())

maybridge_smiles_in_moses_train = list(maybridge_smiles_in_moses_train)
maybridge_mols_in_moses_train = [Chem.MolFromSmiles(
    smi) for smi in maybridge_smiles_in_moses_train]
filter_ = df["tanimoto"] == 1.
filtered = df[filter_]

already_present_in_train = []
intersection_fps = [Chem.RDKFingerprint(m)
                    for m in maybridge_mols_in_moses_train]
for smi_gen in filtered[f"SMILES_{method_name}"]:
    fp_gen = Chem.RDKFingerprint(Chem.MolFromSmiles(smi_gen))
    tanims = DataStructs.BulkTanimotoSimilarity(fp_gen, intersection_fps)
    if (np.array(tanims) == 1).any():
        already_present_in_train.append(smi_gen)

print(len(already_present_in_train))
