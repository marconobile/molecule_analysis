import sys
sys.path.append("..")

from rdkit import RDLogger
from multiprocessing import Pool
from pathlib import Path
import multiprocessing
from rdkit import DataStructs
from rdkit import Chem
import pandas as pd
import os

RDLogger.DisableLog('rdApp.*')


p = Path("/home/nobilm@usi.ch/wd/data/synthesized_datasets/asinex/no_dup_asinex_03_Feb_2022_valid_properties.csv")
df = pd.read_csv(p)
asinex_df = df.drop(df.columns[0], axis=1)


p1 = Path("/home/nobilm@usi.ch/wd/data/training_data/moses_properties_properties.csv")
df1 = pd.read_csv(p1)
moses_df = df1.drop(df1.columns[0], axis=1)

asinex_df = asinex_df[:100]
moses_df = moses_df[:100]

num_cores = multiprocessing.cpu_count()
def f(smi): return smi, Chem.RDKFingerprint(Chem.MolFromSmiles(smi))

treshold = 2

def process_row(row):
    smi_gen = row[1]["SMILES"]
    mol_gen = Chem.MolFromSmiles(smi_gen)
    fp_gen = Chem.RDKFingerprint(mol_gen)
    n_atoms_gen = row[1]["NUM_ATOMS"]
    df2_filter = moses_df["NUM_ATOMS"].between(n_atoms_gen-treshold, n_atoms_gen+treshold)
    df2_filtered = moses_df[df2_filter]
    with Pool(processes=num_cores) as P:
        smi_fp_dict = {smi: fp for smi, fp in P.map(f, df2_filtered["SMILES"].tolist())}    
    tanimotos = DataStructs.BulkTanimotoSimilarity(fp_gen, list(smi_fp_dict.values()))
    for i, smi in enumerate(smi_fp_dict.keys()): smi_fp_dict[smi] = tanimotos[i]
    return smi_gen, smi_fp_dict


asinex_smiles, moses_smiles, tanimotos = [], [], []
for row in asinex_df.iterrows():  
    asinex_smi, result_dict = process_row(row)
    asinex_smiles += [asinex_smi]*len(result_dict)
    moses_smiles += list(result_dict.keys())
    tanimotos += list(result_dict.values())    


out_path = Path("/storage_common/nobilm/data_comparisons/data/synthesized_datasets/asinex/moses_intersection/intersection.csv")
df = pd.DataFrame({
    "SMILES_asinex": asinex_smiles,
    "SMILES_moses": asinex_smiles,
    "tanimoto": asinex_smiles,
}).to_csv(out_path)