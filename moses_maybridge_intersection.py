import os
from data_utils import * 

moses_path = "/home/nobilm@usi.ch/wd/data/training_data/moses.txt"
maybridge_path = "/home/nobilm@usi.ch/wd/data/synthesized_datasets/maybridge/no_dup_Maybridge_HitDiscover_valid.smiles"

moses_mols = mols_from_file(moses_path)
maybridge_mols = mols_from_file(maybridge_path)

exact_matches_mols = []
for m1 in moses_mols:
    nm1 = m1.GetNumAtoms()
    nbm1 = len(list(m1.GetBonds()))
    for m2 in maybridge_mols:
        if nm1 == m2.GetNumAtoms() and nbm1 == len(list(m2.GetBonds())):
            if m1.HasSubstructMatch(m2):
                exact_matches_mols.append((m1, m2))


filepath = "/home/nobilm@usi.ch/wd/data/synthesized_datasets/maybridge/maybridge_vs_moses_train_duplicates.txt"
with open(filepath, "w+") as f:
    for i, (m1, m2) in enumerate(exact_matches_mols):
        f.write(
            f"{i} Moses: {Chem.MolToSmiles(m1)}     Maybridge: {Chem.MolToSmiles(m2)}\n")