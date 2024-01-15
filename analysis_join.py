from data_utils import *
from pathlib import Path
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from more_itertools import chunked
import argparse

def num_mols_over_treshold(df, t=.9):
    filter_ = df["tanimoto"] >= t
    filtered = out[filter_]
    return t, len(filtered)


parser = argparse.ArgumentParser()
parser.add_argument("path")
# path to: {technique}_vs_maybridge.csv file
# example: python analysis_join.py /home/nobilm@usi.ch/wd/data/generated_smiles/moses/hmm/hmm_vs_maybridge.csv
# example: python analysis_join.py /home/nobilm@usi.ch/wd/data/generated_smiles/moses/ngram/ngram_vs_maybridge.csv
args = parser.parse_args()
path = args.path


out_dir = os.path.split(path)[0]
log = create_log(out_dir, name="complete_analysis_log.txt")
# log = os.path.join(out_dir, "log.txt")
out_img_dir = out_dir+ "/img/"
Path(out_img_dir).mkdir(parents=True, exist_ok=True)

out = pd.read_csv(path)
technique_name = os.path.split(Path(path).parent)[-1]

append_line_to_log(log, f"Number of generated mols considered in the analysis: {len(set(out['SMILES_'+technique_name]))}")
append_line_to_log(log, f"Exact matches: {num_mols_over_treshold(out, 1.)[1]}")
append_line_to_log(log, f"Average Tanimoto dist: {out['tanimoto'].mean()}")

name_tanimoto_png = technique_name + '_tanimoto_hist.png'
fig, ax = plt.subplots()
out.hist(column="tanimoto", grid=False, ax=ax)
fig.savefig(os.path.join(out_img_dir, name_tanimoto_png))

treshold = .9
filter_ = out["tanimoto"] >= treshold
filtered = out[filter_]

t, n = num_mols_over_treshold(out)
append_line_to_log(log, f"Mols above treshold ({t}): {n}")

t, n = num_mols_over_treshold(out, .7)
append_line_to_log(log, f"Mols above treshold ({t}): {n}")

t, n = num_mols_over_treshold(out, .95)
append_line_to_log(log, f"Mols above treshold ({t}): {n}")

t, n = num_mols_over_treshold(out, .98)
append_line_to_log(log, f"Mols above treshold ({t}): {n}")

res = []
for _, row in filtered.iterrows():
    res.append({"TD": row["tanimoto"],
                "GEN_SMI": row["SMILES_"+technique_name],
                "REF_SMI": row["SMILES_maybridge"],
                "GEN_MOL": Chem.MolFromSmiles(row["SMILES_"+technique_name]),
                "REF_MOL": Chem.MolFromSmiles(row["SMILES_maybridge"]),
                })

res = sorted(res, key=lambda d: d["TD"], reverse=True)

mols = []
smiles = []
for el in res:
    mols.append(el["GEN_MOL"])
    mols.append(el["REF_MOL"])
    smiles.append("TD: " + str(el["TD"]) + " - " + el["GEN_SMI"])
    smiles.append(el["REF_SMI"])

mols = list(chunked(mols, 200))
smiles = list(chunked(smiles, 200))

for chunk_idx in range(len(mols)):
    # works only with ~ 200 mols and subImgSize=(300, 300)
    img = Draw.MolsToGridImage(mols[chunk_idx], molsPerRow=2, subImgSize=(300, 300), legends=smiles[chunk_idx])
    name_mols_png = f"{technique_name}__mols{chunk_idx}.png"
    img.save(os.path.join(out_img_dir, name_mols_png))    


