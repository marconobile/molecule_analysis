import os
from pathlib import Path
from rdkit import Chem
from data_utils import read_smiles_from_file

cwd = os.getcwd()
moses_smiles = cwd + '/data/training_data/moses.txt'

# https://www.asinex.com/screening-libraries-(all-libraries)
asinex_sdf = cwd + '/data/synthesized_datasets/asinex_03_Feb_2022.sdf'
maybridge_sdf = cwd + '/data/synthesized_datasets/Maybridge_HitDiscover.sdf'
comparisons_df = [asinex_sdf, maybridge_sdf]

for sdf in comparisons_df:
    moses_gen = (Chem.MolFromSmiles(smi)
                 for smi in read_smiles_from_file(moses_smiles))
    exact_matches_mols = []
    for moses_mol in moses_gen:
        if moses_mol:
            gen = (mol for mol in Chem.SDMolSupplier(sdf))
            for mol in gen:
                if mol:
                    if moses_mol.GetNumAtoms() == mol.GetNumAtoms():
                        if moses_mol.HasSubstructMatch(mol):
                            exact_matches_mols.append((moses_mol, mol))

    filepath = cwd + "/" + Path(sdf).stem + '_duplicates.txt'
    with open(filepath, "w+") as f:
        for m1, m2 in exact_matches_mols:
            f.write(
                f"Moses: {Chem.MolToSmiles(m1)}    {Path(sdf).stem}: {Chem.MolToSmiles(m2)}\n")
