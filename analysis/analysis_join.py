'''
You can run this analysis over any {method}_vs_{ref_data}.csv file, it requires a single .csv file as input

performs the analysis of a method against a derired reference dataset
the data for the analysis must be stored in a 
{method}_vs_{ref_data}.csv eg: hmm_vs_maybridge.csv

example:
path to: {technique}_vs_maybridge.csv file
cmd: python analysis_join.py /home/nobilm@usi.ch/wd/data/generated_smiles/moses/hmm/hmm_vs_maybridge.csv

creates a log file with:
Number of generated mols considered in the analysis: 
Exact matches: 
Average Tanimoto dist: 
Mols above treshold (0.7): 
Mols above treshold (0.9): 
Mols above treshold (0.95): 
Mols above treshold (0.98): 

it creates also a .png where mols are displayed by their tanimoto simiarity
'''

import argparse
from more_itertools import chunked
import os
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from utils.data_utils import append_line_to_log, append_line_to_log, create_log, num_mols_over_treshold
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# ----- Parser paths ----- #
parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()
path = args.path

# ----- Processing paths ----- #
out_dir = os.path.split(path)[0]
log = create_log(out_dir, name="analysis_log.txt")
# log = os.path.join(out_dir, "log.txt") # if you want to log preprocessing steps and analysis in same log file
out_img_dir = out_dir + "/img/"
Path(out_img_dir).mkdir(parents=True, exist_ok=True)

# ----- Opening csv ----- #
out = pd.read_csv(path)
technique_name = os.path.split(Path(path).parent)[-1]

# ----- Log info ----- #
append_line_to_log(
    log, f"Number of generated mols considered in the analysis: {len(set(out['SMILES_'+technique_name]))}")
append_line_to_log(log, f"Exact matches: {num_mols_over_treshold(out, 1.)[1]}")
append_line_to_log(log, f"Average Tanimoto dist: {out['tanimoto'].mean()}")
t, n = num_mols_over_treshold(out, .7)
append_line_to_log(log, f"Mols above treshold ({t}): {n}")
t, n = num_mols_over_treshold(out)
append_line_to_log(log, f"Mols above treshold ({t}): {n}")
t, n = num_mols_over_treshold(out, .95)
append_line_to_log(log, f"Mols above treshold ({t}): {n}")
t, n = num_mols_over_treshold(out, .98)
append_line_to_log(log, f"Mols above treshold ({t}): {n}")

# ----- Tanimoto hist ----- #
name_tanimoto_png = technique_name + '_tanimoto_hist.png'
fig, ax = plt.subplots()
out.hist(column="tanimoto", grid=False, ax=ax)
fig.savefig(os.path.join(out_img_dir, name_tanimoto_png))

# ----- (sim, mol_i, mol_j) pic ----- #
treshold = .9
filter_ = out["tanimoto"] >= treshold
filtered = out[filter_]
res = []
for _, row in filtered.iterrows():
    res.append({
        "TD": row["tanimoto"],
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
    img = Draw.MolsToGridImage(mols[chunk_idx], molsPerRow=2, subImgSize=(
        300, 300), legends=smiles[chunk_idx])
    name_mols_png = f"{technique_name}__mols{chunk_idx}.png"
    img.save(os.path.join(out_img_dir, name_mols_png))
