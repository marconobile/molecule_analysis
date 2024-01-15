from data_utils import *
import argparse
import numpy as np
from pathlib import Path
from rdkit import DataStructs
import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import dask.dataframe as dd

RDLogger.DisableLog('rdApp.*')


parser = argparse.ArgumentParser()
parser.add_argument("reference")
# path to csv file containing the properties of reference db eg: /home/marconobile/Desktop/wd/data/synthesized_datasets/maybridge/no_dup_Maybridge_HitDiscover_valid_properties.csv
parser.add_argument("generated")
# path to csv file containing the properties of generated db eg: /home/marconobile/Desktop/wd/data/generated_smiles/moses/aae/aae_properties.csv

# example: python compute_join_td.py /home/marconobile/Desktop/wd/data/synthesized_datasets/maybridge/no_dup_Maybridge_HitDiscover_valid_properties.csv /home/marconobile/Desktop/wd/data/generated_smiles/moses/aae/aae_properties.csv
args = parser.parse_args()

# reference
reference_path = args.reference
ref_df = pd.read_csv(reference_path)
ref_df = ref_df.drop(ref_df.columns[0], axis=1)
ref_dir, _ = os.path.split(Path(reference_path))
_, ref_name = os.path.split(Path(reference_path).parent)

# generated
generated_path = args.generated
gen_df = pd.read_csv(generated_path)
gen_df = gen_df.drop(gen_df.columns[0], axis=1)
_, gen_name = os.path.split(Path(generated_path).parent)

gen_df = dd.from_pandas(gen_df, npartitions=5000)
ref_df = dd.from_pandas(ref_df, npartitions=5000)
# out2 = dd.merge(gen_df1, ref_df, on=["NUM_ATOMS", "NUM_BONDS"]).compute()

out = dd.merge(gen_df, ref_df, on=["NUM_ATOMS", "NUM_BONDS"], how='inner', suffixes=(
    "_"+str(gen_name), "_"+str(ref_name))).compute()

# inner join on atom AND bond exact match
# out = pd.merge(gen_df, ref_df, on=["NUM_ATOMS", "NUM_BONDS"], how='inner', suffixes=(
#     "_"+str(gen_name), "_"+str(ref_name)), copy=False)

outpath = f"/storage_common/nobilm/data_comparisons/data/generated_smiles/moses/{gen_name}/{gen_name}_vs_{ref_name}.csv"
gen_smiles_colname = "SMILES_"+str(gen_name)
ref_smiles_colname = "SMILES_"+str(ref_name)
gen_weight_colname = "WEIGHT_"+str(gen_name)
ref_weight_colname = "WEIGHT_"+str(ref_name)

smiles_gen = []
NUM_ATOMS = []
NUM_BONDS = []
WEIGHT_gen = []
SMILES_ref = []
WEIGHT_ref = []
tanimoto = []

for i, (smi, df) in enumerate(out.groupby(gen_smiles_colname)):

    smiles_gen.extend(df[gen_smiles_colname])
    NUM_ATOMS.extend(df["NUM_ATOMS"])
    NUM_BONDS.extend(df["NUM_BONDS"])
    WEIGHT_gen.extend(df[gen_weight_colname])
    SMILES_ref.extend(df[ref_smiles_colname])
    WEIGHT_ref.extend(df[ref_weight_colname])

    # get fingerprint for gen mol
    # gen_mol = Chem.MolFromSmiles(smi)
    # gen_fps = Chem.RDKFingerprint(gen_mol)

    # get fingerprints for all db mols with same NumA, NumB of gen mol
    # ref_mol_matched = [Chem.MolFromSmiles(smi) for smi in list(df[ref_smiles_colname])]

    ref_mol_matched_fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(smi)) for smi in df[ref_smiles_colname]]

    # compute dists
    tanimoto.extend(DataStructs.BulkTanimotoSimilarity(Chem.RDKFingerprint(Chem.MolFromSmiles(smi)), ref_mol_matched_fps))


out_df  = pd.DataFrame(
    {gen_smiles_colname: smiles_gen,
     'NUM_ATOMS': NUM_ATOMS,
     'NUM_BONDS': NUM_BONDS,
     gen_weight_colname: WEIGHT_gen,
     ref_smiles_colname: SMILES_ref,
     ref_weight_colname: WEIGHT_ref,
     "tanimoto": tanimoto
    })

out_df.to_csv(outpath, index=False)




# # SMILES_aae,NUM_ATOMS,NUM_BONDS,WEIGHT_aae,SMILES_maybridge,WEIGHT_maybridge,TANIMOTO
# to_write = pd.DataFrame(columns=out.columns)

# for i, (smi, df) in enumerate(out.groupby(gen_smiles_colname)):

#     # get fingerprint for gen mol
#     # gen_mol = Chem.MolFromSmiles(smi)
#     # gen_fps = Chem.RDKFingerprint(gen_mol)

#     # get fingerprints for all db mols with same NumA, NumB of gen mol
#     ref_mol_matched = [Chem.MolFromSmiles(smi)
#                        for smi in list(df[ref_smiles_colname])]
#     ref_mol_matched_fps = [Chem.RDKFingerprint(m) for m in ref_mol_matched]

#     # compute dists, precision 3
#     df["TANIMOTO"] = np.around(
#         np.array(DataStructs.BulkTanimotoSimilarity(Chem.RDKFingerprint(Chem.MolFromSmiles(smi)), ref_mol_matched_fps)), 3)

#     pd.concat([to_write, df], ignore_index=True, copy=False)

#     # if i == 0:
#     #     df.to_csv(outpath, index=False)
#     # else:
#     #     df.to_csv(outpath, mode='a', header=False, index=False)

# to_write.to_csv(outpath, index=False)